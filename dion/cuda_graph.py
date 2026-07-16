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
    rejects anything failing ``isinstance(opt, torch.optim.Optimizer)``, which this wrapper
    is not. Scheduling still works with no wrapper plumbing -- ``group["lr"]`` is the device
    tensor the captured graph reads, and the scheduler (bound to the inner optimizer, which
    shares the same param_groups) fills it in place outside the graph, so every replay
    re-reads the live value. A *scheduled* LR takes effect under replay only through such an
    in-place update; reassigning a python float (``group["lr"] = 0.05``) swaps the tensor
    out and leaves replay reading the stale one until the next eager step.
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
    device tensor -- ``group["lr"]`` *is* that tensor -- which the kernels read directly, so
    a ``torch.optim`` LR scheduler drives a captured step natively: it fills the tensor in
    place outside the graph and every replay re-reads the live value, no refresh plumbing.
    (Build the scheduler on the *inner* optimizer; ``group["lr"]`` is shared, so it updates
    the exact tensor the captured graph reads.) The AdamW step is likewise a device tensor,
    incremented on-device before the fused kernel so bias correction advances inside the
    graph. Verified bit-exact eager-vs-replay for a constant and a per-step-scheduled LR,
    including under a real ``CosineAnnealingLR`` (``tests/test_cuda_graph.py``).
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
        # No LR plumbing here: the wrapped optimizer carries each group's LR as the device
        # tensor ``group["lr"]`` that the kernels read, and a scheduler fills it in place
        # outside the graph, so every replay already re-reads the live value.
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
