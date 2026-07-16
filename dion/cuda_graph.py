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

Requirements / caveats:
  * LR schedulers must be constructed on the *inner* optimizer: torch's ``LRScheduler``
    rejects anything failing ``isinstance(opt, torch.optim.Optimizer)``, which this
    wrapper is not. Scheduling still works -- the scheduler writes ``group["lr"]`` and the
    wrapper pushes that into the device LR tensors before each replay.
  * ``loss.backward()`` must accumulate into the same ``p.grad`` tensors each step, so
    call ``zero_grad(set_to_none=False)`` (the wrapper enforces this). The first
    backward allocates ``.grad``; capture pins those buffers.
  * The step must not do a host sync (``.item()``/``.cpu()``) or change tensor shapes
    across steps. Dion2/NorDion2's selection *count* is fixed; only indices/values vary,
    which replay handles (it re-reads the live tensors).
  * On the sharded path the graph holds the captured megabatch all-to-all, so the NCCL
    ops outlive the step. ``dist.destroy_process_group()`` blocks while they are alive:
    drop the wrapper (and any graph it holds) before tearing the process group down at
    the end of a run, or shutdown hangs.
  * torch.compile is neither needed nor wanted under a graph (the graph already removes
    dispatch, and a fullgraph compile of the unrolled per-matrix loops is what makes the
    first step take minutes at many layers). Disable it for the wrapped optimizer.
  * Both the matrix (Muon/NorMuon/Dion2/NorDion2) path and the AdamW *scalar* path
    (embeddings / 1-D params) are capturable. Each group carries its learning rate as a
    device tensor (``megabatch_base._group_lr_tensor``) that the wrapper refreshes before
    each replay, so a *scheduled* LR takes effect under replay instead of freezing at the
    capture value; and ``scalar_opts.adamw_update_foreach``'s capturable form keeps each
    AdamW param's step as a device tensor and increments it on-device before the fused
    kernel, so bias correction advances inside the graph. Verified bit-exact eager-vs-
    replay for both a constant and a per-step-scheduled LR (``tests/test_cuda_graph.py``).
"""

from typing import Optional
import torch


class CudaGraphOptimizer:
    def __init__(self, optimizer, warmup_steps: int = 10):
        if warmup_steps < 1:
            raise ValueError(
                f"warmup_steps must be >= 1, got {warmup_steps}. Capture needs at least "
                "one eager step to have allocated the optimizer state, cuBLAS workspaces "
                "and NCCL buffers; capturing the first step instead allocates them inside "
                "the graph and fails deep in the backend."
            )
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None

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

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, sd):
        # Loading new state invalidates any captured graph (buffers may move).
        self._graph = None
        self._step_count = 0
        return self.optimizer.load_state_dict(sd)

    def zero_grad(self, set_to_none: bool = False):
        # set_to_none=True would swap .grad for a fresh tensor each step, breaking the
        # graph's pinned grad buffers. Force the stable-buffer path.
        if set_to_none:
            raise ValueError(
                "CudaGraphOptimizer requires stable .grad buffers; call "
                "zero_grad(set_to_none=False)."
            )
        self.optimizer.zero_grad(set_to_none=False)

    def _capture(self):
        # Capture exactly ONE step. Lazy allocations, cuBLAS workspaces and NCCL setup are
        # already done by the warmup_steps eager steps, so nothing allocates inside the
        # graph. Capture only RECORDS the ops onto the graph -- it does not execute them --
        # so we replay once immediately to actually perform this step's update. Without
        # that replay the trajectory would be one step short (params/state/step frozen at
        # their pre-capture values for this call).
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.optimizer.step()
        self._graph.replay()

    def step(self):
        # Push the current (possibly scheduled) LR into the wrapped optimizer's device LR
        # tensors here, OUTSIDE the graph, so the captured step reads the live value on
        # every replay. The optimizer's own in-step refresh is skipped while capturing, so
        # this is the only refresh the replay path gets.
        refresh = getattr(self.optimizer, "_refresh_lr_tensors", None)
        if refresh is not None:
            refresh()
        if self._step_count < self.warmup_steps:
            self.optimizer.step()
        elif self._graph is None:
            # step() runs (and so does its host-side bookkeeping) while being traced.
            self._capture()
        else:
            self._graph.replay()
            # Replay executes only the recorded device work, so any host-side per-step
            # bookkeeping in step() has to be advanced here instead.
            advance = getattr(self.optimizer, "_advance_host_step_counters", None)
            if advance is not None:
                advance()
        self._step_count += 1
