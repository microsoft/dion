"""CUDA-graph capture for the optimizer step.

The megabatched Muon/NorMuon/Dion2/NorDion2 optimizer step issues hundreds of small host
ops per step for its per-matrix selection / error-feedback / normalization bookkeeping. On
a sharded step that host dispatch (tens of ms at multi-B scale) is a fixed overhead that
does not shrink with the data-parallel group, so it dominates the distributed step at
small per-GPU compute. The step is a fixed-shape, repeated computation -- an ideal
CUDA-graph target. Capturing it once and replaying collapses the host dispatch to a single
graph launch (GPU time unchanged), and the capture holds through the megabatch all-to-all
(NCCL) and the symmetric-GEMM (CuteDSL) kernels.

Usage (drop-in around any optimizer whose step() is graph-safe -- no host syncs, fixed
shapes, gradients living in stable ``.grad`` tensors):

    inner = Dion2(param_groups, ...)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(inner, T_max=...)  # NOT the wrapper
    opt = CudaGraphOptimizer(inner, warmup_steps=10)
    for batch in loader:
        loss = model(batch); loss.backward()
        opt.step()                    # eager for warmup_steps, then captured+replayed
        opt.zero_grad(set_to_none=False)   # MUST keep .grad tensors stable for replay
        sched.step()

Under a framework that drives automatic optimization with a closure (Lightning calls
``opt.step(closure)``, the closure running forward+backward), ``step`` runs the closure
first, then replays the captured update and returns its loss.

Requirements / caveats:
  * The wrapper is a ``torch.optim.Optimizer`` subclass (so ``isinstance`` holds for
    Lightning and torch's ``LRScheduler``) that delegates all state to the inner optimizer;
    a scheduler may bind to either, since they share ``param_groups``. Scheduling works with
    no wrapper plumbing -- ``group["lr"]`` is the device tensor the captured graph reads,
    and a stock ``torch.optim`` scheduler fills it in place outside the graph, so every
    replay re-reads the live value. Assigning a plain float (``group["lr"] = 0.05``, the
    idiom of a hand-rolled schedule) also works: before capturing or replaying, the wrapper
    asks the inner optimizer to push any such value back into the persistent tensor.
    Optimizers outside this family have no such hook, so any hyperparameter they read as a
    host-side python value -- an LR included -- is baked in at capture and stops changing.
  * ``lr`` is the only group hyperparameter carried live by default. Every other scalar
    (``weight_decay``, ``mu``, ``beta1``, ``epsilon``, ...) reaches the kernels as a host
    value and is baked in at capture, so a callback that rewrites one per step -- e.g.
    schedule-coupled weight decay, which scales ``weight_decay`` with the LR -- would freeze
    it at its capture-step value with no error. ``weight_decay`` can opt in: pass a Tensor
    for it (``weight_decay=torch.tensor(0.01)``) and it is carried as a persistent device
    tensor like the LR, so filling it in place drives a captured step. That is opt-in
    because on the AdamW scalar path it cannot ride ``torch._fused_adamw_``, whose
    ``weight_decay`` is a float in every overload, so it is applied as a separate pass and
    rounds ``X`` once more than the default. Opt in before the first capture -- afterwards
    the graph has already baked the host value, and the wrapper raises rather than let a
    late opt-in look like it took effect.
  * ``loss.backward()`` must accumulate into the same ``p.grad`` tensors each step, so
    call ``zero_grad(set_to_none=False)`` (the wrapper enforces this). The first
    backward allocates ``.grad``; capture pins those buffers.
  * The step must not do a host sync (``.item()``/``.cpu()``) or change tensor shapes
    across steps. Dion2/NorDion2's selection *count* is fixed; only indices/values vary,
    which replay handles (it re-reads the live tensors).
  * Which parameters take part is fixed at capture time: the step skips params whose
    ``.grad`` is None, so a param that gets no tokens during the warmup steps would be left
    out of the graph and stop being updated for the rest of the run. Capture refuses to run
    while any ``requires_grad`` param still has ``.grad=None``, so this fails loudly instead;
    raise ``warmup_steps`` past whatever schedule starves the param, or freeze it explicitly.
  * On the sharded path the graph holds the captured megabatch all-to-all, so the NCCL
    ops outlive the step. ``dist.destroy_process_group()`` blocks while they are alive:
    call ``release()`` before tearing the process group down at the end of a run, or
    shutdown hangs.
  * torch.compile is neither needed nor wanted under a graph (the graph already removes
    dispatch, and a fullgraph compile of the unrolled per-matrix loops is what makes the
    first step take minutes at many layers). Disable it for the wrapped optimizer.
  * Both the matrix (Muon/NorMuon/Dion2/NorDion2) path and the AdamW *scalar* path
    (embeddings / 1-D params) are capturable. Each group carries its learning rate as a
    device tensor -- ``group["lr"]`` *is* that tensor -- which the kernels read directly, so
    a ``torch.optim`` LR scheduler drives a captured step natively: it fills the tensor in
    place outside the graph and every replay re-reads the live value, no refresh plumbing.
    (Build the scheduler on the *inner* optimizer; ``group["lr"]`` is shared, so it updates
    the exact tensor the captured graph reads.) The AdamW step is likewise a device tensor,
    incremented on-device before the fused kernel so bias correction advances inside the
    graph. Verified bit-exact eager-vs-replay for a constant and a per-step-scheduled LR,
    including under a real ``CosineAnnealingLR`` (``tests/test_cuda_graph.py``).
"""

import warnings
from collections import OrderedDict
from typing import Callable, Optional

import torch


class CudaGraphOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 10,
        capture_error_mode: str = "thread_local",
    ):
        if warmup_steps < 1:
            raise ValueError(
                f"warmup_steps must be >= 1, got {warmup_steps}. Capture needs at least "
                "one eager step to have allocated the optimizer state, cuBLAS workspaces "
                "and NCCL buffers; capturing the first step instead allocates them inside "
                "the graph and fails deep in the backend."
            )
        # We deliberately do NOT call Optimizer.__init__: it would build a second, empty
        # param_groups/state on this object. Subclassing is only so isinstance(self,
        # Optimizer) holds (Lightning, torch's LRScheduler); every Optimizer attribute
        # delegates to the wrapped optimizer via the overrides below and __getattr__. We
        # still create the hook registries Optimizer.__init__ would, since torch reads them
        # on state_dict()/step().
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        # torch's cudaStreamCaptureMode. torch defaults to "global", which errors when
        # *any* thread makes a capture-unsafe CUDA call while capture is underway -- and in
        # a distributed run the ProcessGroupNCCL watchdog thread polls cudaEventQuery,
        # which is exactly such a call. "thread_local" is what inductor uses for all of its
        # own captures, for the same reason; it still catches unsafe calls made by the
        # capturing thread, i.e. by the step itself.
        self.capture_error_mode = capture_error_mode
        self._step_count = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        # How many persistent hyperparameter tensors the inner optimizer held when the
        # graph was captured; see _check_live_hyperparams_unchanged.
        self._captured_hyperparam_count = 0
        self._warned_set_to_none = False
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

    def __getattr__(self, name):
        # Delegate the rest of the Optimizer API (state, defaults, add_param_group, ...).
        # Only called for attributes not found normally, so the overrides below still win.
        # ``optimizer`` is guarded because __getattr__ runs before __init__ assigns it
        # (e.g. during unpickling), which would otherwise recurse forever.
        if name == "optimizer":
            raise AttributeError(name)
        return getattr(self.optimizer, name)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def add_param_group(self, param_group):
        # torch.optim.Optimizer defines add_param_group, so (unlike state/defaults) it is
        # found on the base class and __getattr__ never delegates it -- override explicitly.
        # A new group is not in the captured graph; drop it so the next step re-captures.
        # release() restarts the warmup as well: the new group's state, workspaces and comm
        # buffers have never been touched by an eager step, and re-capturing straight away
        # would allocate (or recompile) them inside the graph -- the failure warmup_steps
        # exists to prevent.
        self.release()
        return self.optimizer.add_param_group(param_group)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        # Loading new state invalidates any captured graph (buffers may move).
        self.release()
        return self.optimizer.load_state_dict(sd)

    def zero_grad(self, set_to_none: bool = False):
        # set_to_none=True would swap .grad for a fresh tensor each step, breaking the
        # graph's pinned grad buffers. Force the stable-buffer path; a framework that
        # defaults to True (e.g. Lightning) gets a one-time warning, not a hard failure.
        if set_to_none and not self._warned_set_to_none:
            self._warned_set_to_none = True
            warnings.warn(
                "CudaGraphOptimizer needs stable .grad buffers for graph replay; "
                "ignoring set_to_none=True and zeroing in place.",
                RuntimeWarning,
            )
        self.optimizer.zero_grad(set_to_none=False)

    def release(self) -> None:
        """Drop the captured graph, so the next step captures afresh after a new warmup.

        Call this before ``dist.destroy_process_group()``: on the sharded path the graph holds
        the captured megabatch all-to-all, and destroying the process group blocks while those
        NCCL ops are alive. A framework that keeps optimizers alive past the end of training
        (Lightning's ``Strategy.teardown`` only moves their state to CPU, and the process group
        is then destroyed from an ``atexit`` handler) hangs at exit without this.

        Also the single place a captured graph is dropped: ``add_param_group`` and
        ``load_state_dict`` invalidate the graph and go through here too, so they get the
        same synchronize-and-destroy instead of leaving live NCCL ops to a refcount.
        """
        graph, self._graph = self._graph, None
        self._captured_hyperparam_count = 0
        # The next capture needs its own warmup: whatever made the caller release may also
        # have invalidated the buffers and workspaces the previous warmup had touched.
        # Unconditional, so a release() during the warmup restarts it as documented.
        self._step_count = 0
        if graph is None:
            return
        # Do not tear a graph down underneath an in-flight replay.
        torch.cuda.synchronize()
        # Destroy the graph here rather than leaving it to refcounting. The whole point of
        # release() is that the captured NCCL ops are gone before destroy_process_group(),
        # and any stray reference -- an exception traceback holding the capturing frame, a
        # debugger, a profiler -- would otherwise keep them alive past this call.
        graph.reset()

    def _live_hyperparam_count(self) -> int:
        tensors = getattr(self.optimizer, "_hyperparam_tensors", None)
        return len(tensors) if tensors is not None else 0

    def _check_live_hyperparams_unchanged(self):
        # A group hyperparameter opted into being carried live (weight_decay=Tensor is the
        # way in) gets a fresh persistent device tensor the moment it opts in. Doing that
        # after capture is too late: the graph baked the old host value and never reads the
        # new tensor, so the schedule the caller just wired up would silently do nothing.
        # The count only ever grows, so comparing it is enough -- and it is an int compare
        # per replay rather than a host-side walk of the param groups.
        if self._live_hyperparam_count() != self._captured_hyperparam_count:
            raise RuntimeError(
                "a param group started carrying a hyperparameter as a live device tensor "
                "after the CUDA graph was captured (weight_decay=Tensor is the way in). "
                "The captured graph baked the previous host value and cannot read the new "
                "tensor. Opt in before the first capture, or call release() to re-capture."
            )

    def _check_params_have_grads(self):
        # The step only touches params with a gradient, so capture freezes the participating
        # set: a param whose .grad is still None here is left out of the graph and silently
        # stops being updated for the rest of the run. Catching that is worth one host-side
        # pass over the param groups, once, at capture. Params excluded on purpose
        # (requires_grad=False) are not the failure this guards against.
        missing = [
            f"group {index}[{position}] shape {tuple(param.shape)}"
            for index, group in enumerate(self.param_groups)
            for position, param in enumerate(group["params"])
            if param.requires_grad and param.grad is None
        ]
        if missing:
            # Params carry no names here, so point at where they sit in the groups -- enough
            # to find them again in whatever built the groups.
            shown = ", ".join(missing[:5])
            if len(missing) > 5:
                shown += f", ... (+{len(missing) - 5} more)"
            raise RuntimeError(
                f"{len(missing)} parameter(s) require a gradient but have .grad=None at CUDA "
                f"graph capture, so replay would never update them: {shown}. Capture after "
                "every parameter has taken part in a backward pass -- raise warmup_steps past "
                "whatever schedule (modality cadence, expert routing) starves them -- or set "
                "requires_grad=False on the ones that are meant to be frozen."
            )

    def _capture(self):
        # Capture exactly ONE step. Lazy allocations, cuBLAS workspaces and NCCL setup are
        # already done by the warmup_steps eager steps, so nothing allocates inside the
        # graph. Capture only RECORDS the ops onto the graph -- it does not execute them --
        # so we replay once immediately to actually perform this step's update. Without
        # that replay the trajectory would be one step short (params/state/step frozen at
        # their pre-capture values for this call).
        self._check_params_have_grads()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode=self.capture_error_mode):
            self.optimizer.step()
        # Publish only once capture has succeeded. A graph left over from a failed capture
        # is not replayable, and holding it here would turn one loud capture error into
        # every later step silently replaying garbage.
        self._graph = graph
        self._captured_hyperparam_count = self._live_hyperparam_count()
        graph.replay()

    def _sync_host_hyperparams(self):
        # The captured graph reads each group's LR from the device tensor that was live at
        # capture time. A stock LR scheduler fills that tensor in place, so it needs
        # nothing from us -- but the equally common `group["lr"] = 0.05` assignment
        # replaces it with a float the graph will never see. step() reconciles that on the
        # eager path; under capture/replay it either does not run at all (replay) or runs
        # with capture already underway (capture), so do it here, before either.
        sync = getattr(self.optimizer, "_sync_hyperparam_tensors", None)
        if sync is not None:
            sync()

    def step(self, closure: Optional[Callable] = None):
        # Automatic-optimization frameworks (Lightning) call step(closure); the closure runs
        # forward+backward, filling the stable .grad buffers the capture reads. Run it eagerly
        # first, then apply the captured update and return its loss. replay() launches on the
        # current stream -- where the closure's backward ran -- so the update is ordered after
        # the fresh gradients.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._step_count < self.warmup_steps:
            self.optimizer.step()
        else:
            self._sync_host_hyperparams()
            if self._graph is None:
                # step() runs (and so does its host-side bookkeeping) while being traced.
                self._capture()
            else:
                self._check_live_hyperparams_unchanged()
                self._graph.replay()
                # Replay executes only the recorded device work, so any host-side per-step
                # bookkeeping in step() has to be advanced here instead.
                advance = getattr(self.optimizer, "_advance_host_step_counters", None)
                if advance is not None:
                    advance()
        self._step_count += 1
        return loss
