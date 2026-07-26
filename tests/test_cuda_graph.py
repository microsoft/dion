"""CUDA-graph capture of the optimizer step (``dion.cuda_graph.CudaGraphOptimizer``).

The megabatched Muon/NorMuon/Dion2/NorDion2 step is a fixed-shape computation (top-k
selection changes which rows are chosen, not the shapes), so capturing it once and
replaying collapses the per-matrix host dispatch to a single graph launch. These tests
check that the wrapper is numerically correct:

- capture+replay matches running the bare optimizer eagerly, step for step, for the
  matrix (Muon/NorMuon/Dion2/NorDion2) path and the AdamW scalar path together; and
- a per-step LR schedule takes effect under replay -- the LR is carried as a device
  tensor the optimizer reads and the wrapper refreshes before each replay -- rather than
  freezing at the capture-time value.
"""
import io
import os

import pytest
import torch

from dion import Dion2, Muon, NorDion2, NorMuon
from dion.cuda_graph import CudaGraphOptimizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu", reason="requires CUDA")

STEPS, WARMUP, SEED = 12, 3, 0
DIST_STEPS = 6
OPTIMIZERS = [Muon, NorMuon, Dion2, NorDion2]


def _build(optimizer_cls):
    torch.manual_seed(SEED)
    weights = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE)),
               torch.nn.Parameter(torch.randn(128, 64, device=DEVICE))]
    biases = [torch.nn.Parameter(torch.randn(64, device=DEVICE)),
              torch.nn.Parameter(torch.randn(128, device=DEVICE))]
    opt = optimizer_cls([
        {"params": weights},
        {"params": biases, "algorithm": "adamw"},
    ], distributed_mesh=None, lr=0.02)
    return weights + biases, opt


def _grad_seq(params):
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    return [[torch.randn(p.shape, device=DEVICE, generator=gen) for p in params]
            for _ in range(STEPS)]


def _run(params, opt, grad_seq, step_fn, lrs=None):
    # Stable .grad buffers (the graph pins them); refill in place, never reallocate.
    for p in params:
        p.grad = torch.zeros_like(p)
    for t, gs in enumerate(grad_seq):
        if lrs is not None:
            # Update the LR *in place*, the way a torch LR scheduler does. group["lr"] is
            # the device tensor the kernels read; reassigning a python float would swap the
            # object out and leave the captured graph reading the stale one under replay.
            for g in opt.param_groups:
                g["lr"].fill_(lrs[t])
        for p, g in zip(params, gs):
            p.grad.copy_(g)
        step_fn()
    return [p.detach().clone() for p in params]


def _roundtrip(opt):
    # The ordinary resume idiom: checkpoint to disk, reload via CPU, load_state_dict.
    buf = io.BytesIO()
    torch.save(opt.state_dict(), buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=False)


def _resumed(optimizer_cls, presteps=3):
    params, opt = _build(optimizer_cls)
    _run(params, opt, _grad_seq(params)[:presteps], opt.step)
    sd = _roundtrip(opt)
    params, opt = _build(optimizer_cls)
    opt.load_state_dict(sd)
    return params, opt


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_cudagraph_matches_eager(optimizer_cls):
    p0, _ = _build(optimizer_cls)
    grad_seq = _grad_seq(p0)

    pe, oe = _build(optimizer_cls)
    final_eager = _run(pe, oe, grad_seq, oe.step)

    pg, og = _build(optimizer_cls)
    wrap = CudaGraphOptimizer(og, warmup_steps=WARMUP)
    # A framework (Lightning) both isinstance-checks the optimizer and reads its groups
    # through the wrapper; the wrapper is-a Optimizer and shares the wrapped groups/state.
    assert isinstance(wrap, torch.optim.Optimizer)
    assert wrap.param_groups is og.param_groups and wrap.state is og.state
    final_graph = _run(pg, og, grad_seq, wrap.step)

    diff = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    assert diff <= 1e-5, f"{optimizer_cls.__name__}: capture-vs-eager diff {diff:.3e}"


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_cudagraph_tracks_scheduled_lr(optimizer_cls):
    p0, o0 = _build(optimizer_cls)
    base = o0.param_groups[0]["lr"]
    grad_seq = _grad_seq(p0)
    # A schedule that varies ~8x across the run (warmup then decay).
    sched = [base * (0.25 + 1.5 * (t + 1) / STEPS) for t in range(STEPS)]
    const = [base] * STEPS

    pe, oe = _build(optimizer_cls)
    final_eager = _run(pe, oe, grad_seq, oe.step, lrs=sched)

    pg, og = _build(optimizer_cls)
    wrap = CudaGraphOptimizer(og, warmup_steps=WARMUP)
    final_graph = _run(pg, og, grad_seq, wrap.step, lrs=sched)

    pc, oc = _build(optimizer_cls)
    wrapc = CudaGraphOptimizer(oc, warmup_steps=WARMUP)
    final_const = _run(pc, oc, grad_seq, wrapc.step, lrs=const)

    tracks = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    nontrivial = max((a - b).abs().max().item() for a, b in zip(final_graph, final_const))
    assert tracks <= 1e-5, (
        f"{optimizer_cls.__name__}: graph did not track the schedule (diff {tracks:.3e})"
    )
    assert nontrivial > 1e-3, (
        f"{optimizer_cls.__name__}: schedule looks frozen -- scheduled and constant runs "
        f"agree to {nontrivial:.3e}, so the test would pass even if the LR were baked"
    )


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_scheduled_lr_tracks_after_resume(optimizer_cls):
    # Same contract as test_cudagraph_tracks_scheduled_lr, but on an optimizer restored
    # from a checkpoint -- the state every real run is in after the first resume.
    # Regression: the device LR tensors were serialized into param_groups, so
    # load_state_dict() overwrote the live ones with the checkpoint's copies, which land
    # on CPU under the usual map_location="cpu". A CPU LR still trains correctly eagerly,
    # so nothing raises; it is read once at capture and baked into the graph, silently
    # freezing the schedule under replay.
    p0, o0 = _build(optimizer_cls)
    base = o0.param_groups[0]["lr"]
    grad_seq = _grad_seq(p0)
    sched = [base * (0.25 + 1.5 * (t + 1) / STEPS) for t in range(STEPS)]
    const = [base] * STEPS

    pe, oe = _resumed(optimizer_cls)
    final_eager = _run(pe, oe, grad_seq, oe.step, lrs=sched)

    pg, og = _resumed(optimizer_cls)
    wrap = CudaGraphOptimizer(og, warmup_steps=WARMUP)
    final_graph = _run(pg, og, grad_seq, wrap.step, lrs=sched)

    pc, oc = _resumed(optimizer_cls)
    wrapc = CudaGraphOptimizer(oc, warmup_steps=WARMUP)
    final_const = _run(pc, oc, grad_seq, wrapc.step, lrs=const)

    tracks = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    nontrivial = max((a - b).abs().max().item() for a, b in zip(final_graph, final_const))
    assert tracks <= 1e-5, (
        f"{optimizer_cls.__name__}: resumed graph did not track the schedule "
        f"(diff {tracks:.3e})"
    )
    assert nontrivial > 1e-3, (
        f"{optimizer_cls.__name__}: schedule looks frozen -- scheduled and constant runs "
        f"agree to {nontrivial:.3e}, so the test would pass even if the LR were baked"
    )


def test_lr_is_device_tensor_but_checkpoints_as_float():
    # Runtime: group["lr"] is the 0-d device fp32 tensor the kernels read (and a scheduler
    # fills in place). Checkpoint: serialized as a plain float, so it stays portable and
    # does not ride a CUDA tensor back onto the wrong device on resume.
    params, opt = _build(Dion2)
    _run(params, opt, _grad_seq(params)[:1], opt.step)

    for g in opt.param_groups:
        lr = g["lr"]
        assert torch.is_tensor(lr) and lr.is_cuda and lr.dtype == torch.float32 and lr.ndim == 0

    for g in opt.state_dict()["param_groups"]:
        assert type(g["lr"]) is float


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_real_lr_scheduler_drives_captured_step(optimizer_cls):
    # The elegance payoff: a stock torch LR scheduler on the inner optimizer drives the
    # captured step with no wrapper LR plumbing, because it fills group["lr"] -- the exact
    # device tensor the graph reads -- in place. Eager and captured runs must agree, and the
    # captured run must actually move with the schedule (not freeze at the capture value).
    grad_seq = _grad_seq(_build(optimizer_cls)[0])

    def run(step_wrapped):
        params, opt = _build(optimizer_cls)
        # T_max small vs STEPS so the cosine sweeps a wide LR range across the run.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=1e-4)
        step = CudaGraphOptimizer(opt, warmup_steps=WARMUP).step if step_wrapped else opt.step
        for p in params:
            p.grad = torch.zeros_like(p)
        for gs in grad_seq:
            for p, g in zip(params, gs):
                p.grad.copy_(g)
            step()
            sched.step()
        return [p.detach().clone() for p in params]

    final_eager = run(step_wrapped=False)
    final_graph = run(step_wrapped=True)
    diff = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    assert diff <= 1e-5, f"{optimizer_cls.__name__}: scheduled capture-vs-eager diff {diff:.3e}"

    # Guard against a frozen LR: rerun the graph at a constant LR and require it to differ.
    params, opt = _build(optimizer_cls)
    for p in params:
        p.grad = torch.zeros_like(p)
    wrap = CudaGraphOptimizer(opt, warmup_steps=WARMUP)
    for gs in grad_seq:
        for p, g in zip(params, gs):
            p.grad.copy_(g)
        wrap.step()
    frozen = max((a - b).abs().max().item() for a, b in zip(final_graph, params))
    assert frozen > 1e-3, (
        f"{optimizer_cls.__name__}: scheduled and constant-LR captured runs agree to "
        f"{frozen:.3e} -- the schedule looks frozen under replay"
    )


def test_resume_from_checkpoint_without_step_tensor():
    # A checkpoint written before the AdamW step moved on-device has no "step_dev" entry.
    # The host-side group["step"] is what to resume from: starting over at 0 replays
    # AdamW's bias-correction warmup, spiking the effective LR after every resume.
    params, opt = _build(Dion2)
    _run(params, opt, _grad_seq(params), opt.step)
    sd = _roundtrip(opt)
    for entry in sd["state"].values():
        entry.pop("step_dev", None)  # what an older build wrote

    _, opt2 = _build(Dion2)
    opt2.load_state_dict(sd)
    steps = [s["step_dev"].item() for s in opt2.state.values() if "step_dev" in s]
    assert steps, "adamw params should carry a device step counter after load"
    assert all(s == STEPS for s in steps), f"expected step {STEPS}, got {steps}"


def test_step_tensor_survives_bf16_param_resume():
    # load_state_dict() casts state tensors to the owning param's dtype (its "step" special
    # case keys off the literal name, which this is not). A bf16 counter silently stops
    # incrementing at 256, freezing bias correction for the rest of training.
    torch.manual_seed(SEED)
    biases = [torch.nn.Parameter(torch.randn(64, device=DEVICE, dtype=torch.bfloat16))]
    opt = Dion2([{"params": biases, "algorithm": "adamw"}], distributed_mesh=None, lr=0.02)
    for p in biases:
        p.grad = torch.zeros_like(p)
    opt.step()

    sd = _roundtrip(opt)
    biases2 = [torch.nn.Parameter(torch.randn(64, device=DEVICE, dtype=torch.bfloat16))]
    opt2 = Dion2(
        [{"params": biases2, "algorithm": "adamw"}], distributed_mesh=None, lr=0.02
    )
    opt2.load_state_dict(sd)

    step = next(s["step_dev"] for s in opt2.state.values() if "step_dev" in s)
    assert step.dtype == torch.float32, f"step counter cast to {step.dtype}"

    # It must still count where bf16 cannot: 256 + 1 == 256 in bf16.
    step.fill_(300)
    for p in biases2:
        p.grad = torch.zeros_like(p)
    opt2.step()
    assert step.item() == 301, f"step counter stuck at {step.item()}"


def test_host_step_counter_advances_under_replay():
    # step()'s python group["step"] += 1 only runs when step() is traced, so replay must
    # advance it host-side or every later checkpoint records a stale step.
    params, opt = _build(Dion2)
    wrap = CudaGraphOptimizer(opt, warmup_steps=WARMUP)
    _run(params, opt, _grad_seq(params), wrap.step)
    assert all(g["step"] == STEPS for g in opt.param_groups), (
        f"host step counters froze at {[g['step'] for g in opt.param_groups]}, expected {STEPS}"
    )


def test_warmup_steps_must_allow_an_eager_step():
    _, opt = _build(Dion2)
    with pytest.raises(ValueError, match="warmup_steps"):
        CudaGraphOptimizer(opt, warmup_steps=0)


# ---- distributed: capture must hold through the megabatch all-to-all (NCCL) ----
# The single-GPU tests above run with distributed_mesh=None, which never reaches the
# all-to-all -- yet collapsing that dispatch on a sharded step is the whole point of
# capturing. This runs the real 2-rank sharded path, captured, against an eager run.


def _dist_worker(rank, world_size, port, mode, out_path):
    import torch.distributed as dist
    from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    wrap = step_fn = None
    try:
        dev = torch.device(f"cuda:{rank}")
        mesh = DeviceMesh("cuda", list(range(world_size)))
        gen = torch.Generator(device=dev).manual_seed(1234)
        w0 = torch.randn(64, 128, generator=gen, device=dev)
        param = torch.nn.Parameter(distribute_tensor(w0, mesh, [Shard(0)]))
        param.grad = distribute_tensor(torch.zeros(64, 128, device=dev), mesh, [Shard(0)])

        opt = NorDion2(
            [dict(params=[param])], distributed_mesh=mesh, lr=0.1, newton_schulz_func=None
        )
        wrap = None if mode == "eager" else CudaGraphOptimizer(opt, warmup_steps=2)
        step_fn = opt.step if wrap is None else wrap.step

        # Shard the grads up front: distribute_tensor issues its own collectives, and
        # interleaving those with replay measures the harness, not the optimizer.
        grads = [
            distribute_tensor(torch.randn(64, 128, generator=gen, device=dev), mesh, [Shard(0)])
            for _ in range(DIST_STEPS)
        ]
        for t, g in enumerate(grads):
            param.grad.copy_(g)
            for group in opt.param_groups:
                # In place (as a scheduler does): group["lr"] is the device tensor the
                # captured graph reads; reassigning a float would leave replay on the stale
                # tensor.
                group["lr"].fill_(0.1 * (0.25 + t / DIST_STEPS))  # scheduled, must track
            step_fn()
        torch.cuda.synchronize()

        # full_tensor() is itself a collective: every rank must call it, or the ranks that
        # skip it run ahead to teardown and the ones that called it block forever.
        full = param.detach().full_tensor()
        if rank == 0:
            torch.save(full.cpu(), out_path)
    finally:
        # Release the captured graph before tearing the process group down: it holds the
        # captured NCCL ops, and destroy_process_group() blocks while they are alive.
        step_fn = wrap = None
        torch.cuda.synchronize()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_cudagraph_matches_eager_on_sharded_megabatch(tmp_path):
    mp = torch.multiprocessing
    out = {}
    # Unique port per spawn to avoid bind collisions with the previous group.
    for port, mode in ((29655, "eager"), (29656, "graph")):
        path = tmp_path / f"{mode}.pt"
        mp.spawn(_dist_worker, args=(2, port, mode, str(path)), nprocs=2, join=True)
        out[mode] = torch.load(path, weights_only=False)

    diff = (out["eager"] - out["graph"]).abs().max().item()
    assert diff <= 1e-5, f"sharded capture-vs-eager diff {diff:.3e}"


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_cudagraph_step_with_closure(optimizer_cls):
    # Lightning's automatic optimization calls step(closure); the closure runs
    # forward+backward (here it fills .grad in place) and returns the loss. The wrapped
    # step must run the closure, apply the captured update, and hand back the loss.
    p0, _ = _build(optimizer_cls)
    grad_seq = _grad_seq(p0)

    pe, oe = _build(optimizer_cls)
    final_eager = _run(pe, oe, grad_seq, oe.step)

    pg, og = _build(optimizer_cls)
    wrap = CudaGraphOptimizer(og, warmup_steps=WARMUP)
    for p in pg:
        p.grad = torch.zeros_like(p)
    losses = []
    for gs in grad_seq:
        def closure(gs=gs):
            for p, g in zip(pg, gs):
                p.grad.copy_(g)
            return torch.full((), 0.5, device=DEVICE)
        losses.append(wrap.step(closure))
    final_graph = [p.detach().clone() for p in pg]

    diff = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    assert diff <= 1e-5, f"{optimizer_cls.__name__}: closure step-vs-eager diff {diff:.3e}"
    assert all(l is not None for l in losses)
