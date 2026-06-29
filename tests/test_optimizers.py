"""Functional tests for all optimizers in the dion package.

Tests cover:
- Basic operation: each optimizer can run multiple steps without error
- Determinism: same seed produces same results
- Parameter update: parameters actually change after a step
- Mixed param groups: matrix params (ortho) + vector params (scalar opt)
- All optimizer-specific options (nesterov, cautious_wd, base_opt, etc.)
- Step timing: optimizer step takes measurable wall-clock time
"""

import os
import pytest
import time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_DEVICE_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0

# Bump torch.compile cache size to avoid recompilation failures
# when the same compiled function sees different input shapes across tests.
torch._dynamo.config.cache_size_limit = 64


def _make_params(shapes, device=DEVICE):
    """Create parameters with deterministic values."""
    torch.manual_seed(42)
    return [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]


def _run_steps(optimizer_cls, params, opt_kwargs, n_steps=3):
    """Run n optimizer steps with deterministic gradients."""
    opt = optimizer_cls(params, **opt_kwargs)
    for step in range(n_steps):
        torch.manual_seed(100 + step)
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    return [p.data.clone() for p in params]


@pytest.mark.parametrize(
    ("scalar_opt", "expected_scale"),
    [
        ("adamw", 1.0),
        ("lion", 1 / 768**0.5),
    ],
)
def test_lm_head_lr_scale_by_scalar_opt(scalar_opt, expected_scale):
    from dion.opt_utils import lm_head_lr_scale

    assert lm_head_lr_scale(scalar_opt, model_dim=768) == pytest.approx(expected_scale)


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestMuon:
    def test_basic(self):
        from dion import Muon
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(Muon, params, dict(lr=0.01))

    def test_determinism(self):
        from dion import Muon
        p1 = _make_params([(64, 128)])
        r1 = _run_steps(Muon, p1, dict(lr=0.01))
        p2 = _make_params([(64, 128)])
        r2 = _run_steps(Muon, p2, dict(lr=0.01))
        assert torch.equal(r1[0], r2[0])

    def test_params_change(self):
        from dion import Muon
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        _run_steps(Muon, params, dict(lr=0.01), n_steps=1)
        assert not torch.equal(params[0].data, before)

    def test_nesterov(self):
        from dion import Muon
        params = _make_params([(64, 128)])
        _run_steps(Muon, params, dict(lr=0.01, nesterov=True))

    def test_cautious_wd(self):
        from dion import Muon
        params = _make_params([(64, 128)])
        _run_steps(Muon, params, dict(lr=0.01, cautious_wd=True))

    def test_adjust_lr_options(self):
        from dion import Muon
        for adjust_lr in ["spectral_norm", "rms_norm", None]:
            params = _make_params([(64, 128)])
            _run_steps(Muon, params, dict(lr=0.01, adjust_lr=adjust_lr))

    def test_megabatch_same_shape(self):
        """Multiple same-shape params should be megabatched."""
        from dion import Muon
        params = _make_params([(64, 128)] * 5)
        _run_steps(Muon, params, dict(lr=0.01))

    def test_mixed_shapes(self):
        """Different shapes go to different shape groups."""
        from dion import Muon
        params = _make_params([(64, 128), (128, 64), (32, 32)])
        _run_steps(Muon, params, dict(lr=0.01))


# ---------------------------------------------------------------------------
# NorMuon
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNorMuon:
    def test_basic(self):
        from dion import NorMuon
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(NorMuon, params, dict(lr=0.01))

    def test_determinism(self):
        from dion import NorMuon
        p1 = _make_params([(64, 128)])
        r1 = _run_steps(NorMuon, p1, dict(lr=0.01))
        p2 = _make_params([(64, 128)])
        r2 = _run_steps(NorMuon, p2, dict(lr=0.01))
        assert torch.equal(r1[0], r2[0])

    def test_params_change(self):
        from dion import NorMuon
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        _run_steps(NorMuon, params, dict(lr=0.01), n_steps=1)
        assert not torch.equal(params[0].data, before)

    def test_variance_neuron_state(self):
        """NorMuon should create and update variance_neuron state."""
        from dion import NorMuon
        params = _make_params([(64, 128)])
        opt = NorMuon(params, lr=0.01)
        params[0].grad = torch.randn_like(params[0])
        opt.step()
        state = opt.state[params[0]]
        assert "variance_neuron" in state
        assert state["variance_neuron"].shape == (64, 1)

    def test_megabatch_same_shape(self):
        from dion import NorMuon
        params = _make_params([(64, 128)] * 5)
        _run_steps(NorMuon, params, dict(lr=0.01))


# ---------------------------------------------------------------------------
# NorDion2
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNorDion2:
    def test_basic(self):
        from dion import NorDion2
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(NorDion2, params, dict(lr=0.01))

    def test_determinism(self):
        from dion import NorDion2
        p1 = _make_params([(64, 128)])
        r1 = _run_steps(NorDion2, p1, dict(lr=0.01))
        p2 = _make_params([(64, 128)])
        r2 = _run_steps(NorDion2, p2, dict(lr=0.01))
        assert torch.equal(r1[0], r2[0])

    def test_params_change(self):
        from dion import NorDion2
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        _run_steps(NorDion2, params, dict(lr=0.01), n_steps=1)
        assert not torch.equal(params[0].data, before)

    def test_variance_neuron_shape(self):
        """variance_neuron should have shape [rows, 1]."""
        from dion import NorDion2
        params = _make_params([(64, 128)])
        opt = NorDion2(params, lr=0.01)
        params[0].grad = torch.randn_like(params[0])
        opt.step()
        state = opt.state[params[0]]
        assert "variance_neuron" in state
        assert state["variance_neuron"].shape == (64, 1)

    def test_fraction(self):
        from dion import NorDion2
        for fraction in [0.1, 0.25, 0.5, 1.0, 1]:
            params = _make_params([(64, 128)])
            _run_steps(NorDion2, params, dict(lr=0.01, fraction=fraction))
            if fraction == 1.0:
                #ensure all variance values are updated after 1 step as all rows are selected
                opt = NorDion2(params, lr=0.01, fraction=fraction)
                params[0].grad = torch.randn_like(params[0])
                opt.step()
                v_before = opt.state[params[0]]["variance_neuron"].clone()
                params[0].grad = torch.randn_like(params[0])
                opt.step()
                v_after = opt.state[params[0]]["variance_neuron"]
                assert (v_after != v_before).all()

    def test_megabatch_same_shape(self):
        from dion import NorDion2
        params = _make_params([(64, 128)] * 5)
        _run_steps(NorDion2, params, dict(lr=0.01))

    def test_output_dtype_matches_param(self):
        """After optimizer step, param dtype should be unchanged."""
        from dion import NorDion2
        for param_dtype in [torch.float32, torch.bfloat16]:
            params = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE, dtype=param_dtype))]
            opt = NorDion2(params, lr=0.01)
            params[0].grad = torch.randn_like(params[0])
            opt.step()
            assert params[0].dtype == param_dtype, (
                f"param dtype changed from {param_dtype} to {params[0].dtype}"
            )

    def test_triton_post_ortho(self):
        """Test that the post-ortho Triton kernel runs without error."""
        from dion import NorDion2
        params = _make_params([(64, 128)])
        opt = NorDion2(params, lr=0.01, triton_post_ortho=True)
        params[0].grad = torch.randn_like(params[0])
        opt.step()

    def test_triton_post_ortho_parity(self):
        """Triton post-ortho should produce same/similar result as eager path."""
        from dion import NorDion2
        p1 = _make_params([(64, 128)])
        r1 = _run_steps(NorDion2, p1, dict(lr=0.01, triton_post_ortho=False))
        p2 = _make_params([(64, 128)])
        r2 = _run_steps(NorDion2, p2, dict(lr=0.01, triton_post_ortho=True))
        torch.testing.assert_close(r1[0], r2[0], atol=1e-5, rtol=1e-5)

    def test_normalize_selected_stacked_matches_unfused(self):
        from dion.nordion2 import nordion2_normalize_selected_stacked
        from dion.normuon import normuon_normalization_stacked
        torch.manual_seed(3)
        n, rows, cols, k = 4, 16, 32, 8
        u = torch.randn(n, k, cols, device=DEVICE, dtype=torch.bfloat16)
        v_full = torch.rand(n, rows, 1, device=DEVICE, dtype=torch.bfloat16)
        indices = torch.stack(
            [torch.randperm(rows, device=DEVICE)[:k] for _ in range(n)], dim=0
        )
        beta2 = torch.tensor(0.9)

        u_fused, v_fused = nordion2_normalize_selected_stacked(
            u.clone(), v_full.clone(), indices, beta2
        )

        idx = indices.unsqueeze(-1)
        v_sel = torch.gather(v_full, dim=-2, index=idx).float()
        u_ref, v_sel_new = normuon_normalization_stacked(u.clone(), v_sel, beta2)
        v_ref = v_full.clone()
        for i in range(n):
            v_ref[i].scatter_(dim=-2, index=idx[i], src=v_sel_new[i].to(v_ref.dtype))

        # The fused helper and the reference run through different inductor
        # fusion groupings (gather/normalize/scatter fused into one graph vs
        # eager gather/scatter around a separately compiled normalization), so
        # they are not guaranteed bit-identical: inductor may reorder the fp32
        # reductions/divisions, leaving last-ULP differences. Compare at the
        # same fp tolerance as the sibling parity tests.
        torch.testing.assert_close(u_fused, u_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_fused, v_ref, atol=1e-5, rtol=1e-5)

# ---------------------------------------------------------------------------
# Dion2
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestDion2:
    def test_basic(self):
        from dion import Dion2
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(Dion2, params, dict(lr=0.01))

    def test_determinism(self):
        from dion import Dion2
        p1 = _make_params([(64, 128)])
        r1 = _run_steps(Dion2, p1, dict(lr=0.01))
        p2 = _make_params([(64, 128)])
        r2 = _run_steps(Dion2, p2, dict(lr=0.01))
        assert torch.equal(r1[0], r2[0])

    def test_params_change(self):
        from dion import Dion2
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        _run_steps(Dion2, params, dict(lr=0.01), n_steps=1)
        assert not torch.equal(params[0].data, before)

    def test_fraction(self):
        """Different fraction values should work, including int (issue #39)."""
        from dion import Dion2
        for fraction in [0.1, 0.25, 0.5, 1.0, 1]:
            params = _make_params([(64, 128)])
            _run_steps(Dion2, params, dict(lr=0.01, fraction=fraction))

    def test_ef_decay(self):
        from dion import Dion2
        for ef_decay in [0.0, 0.5, 0.95, 1.0]:
            params = _make_params([(64, 128)])
            _run_steps(Dion2, params, dict(lr=0.01, ef_decay=ef_decay))

    def test_megabatch_same_shape(self):
        from dion import Dion2
        params = _make_params([(64, 128)] * 5)
        _run_steps(Dion2, params, dict(lr=0.01))

    def test_select_dim_rows_vs_cols(self):
        """Tall matrices select columns, wide select rows."""
        from dion import Dion2
        # Wide: rows < cols → select rows
        params = _make_params([(32, 128)])
        _run_steps(Dion2, params, dict(lr=0.01, verbose=True))
        # Tall: rows > cols → select cols
        params = _make_params([(128, 32)])
        _run_steps(Dion2, params, dict(lr=0.01, verbose=True))

    def test_3d_params_wide(self):
        """3D params (batch of wide matrices) should work with select_dim=-2."""
        from dion import Dion2
        params = _make_params([(4, 32, 128)])
        before = params[0].data.clone()
        _run_steps(Dion2, params, dict(lr=0.01, fraction=0.5), n_steps=3)
        assert not torch.equal(params[0].data, before)

    def test_3d_params_tall(self):
        """3D params (batch of tall matrices) should work with select_dim=-1."""
        from dion import Dion2
        params = _make_params([(4, 128, 32)])
        before = params[0].data.clone()
        _run_steps(Dion2, params, dict(lr=0.01, fraction=0.25), n_steps=3)
        assert not torch.equal(params[0].data, before)

    def test_3d_params_flatten(self):
        """3D params with flatten=True should flatten to 2D for ortho."""
        from dion import Dion2
        params = _make_params([(4, 32, 128)])
        _run_steps(Dion2, params, dict(lr=0.01, flatten=True), n_steps=3)

    def test_3d_megabatch(self):
        """Multiple 3D params with same shape should be megabatched."""
        from dion import Dion2
        params = _make_params([(4, 32, 64)] * 3)
        _run_steps(Dion2, params, dict(lr=0.01), n_steps=3)


# ---------------------------------------------------------------------------
# num_heads per-group option (per-head Newton-Schulz on 2D weights)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNumHeads:
    """The ``num_heads`` param-group option lets a 2D weight be treated as a
    batch of ``num_heads`` matrices for Newton-Schulz, matching the behavior of
    an equivalent 3D-stored weight without changing the model layout."""

    def _run_parity(self, optimizer_cls, opt_kwargs, num_heads=4, head_dim=8, in_features=16, n_steps=3):
        torch.manual_seed(0)
        init = torch.randn(num_heads * head_dim, in_features, device=DEVICE)

        w2d = torch.nn.Parameter(init.clone())
        opt2d = optimizer_cls(
            [{"params": [w2d], "num_heads": num_heads}], **opt_kwargs
        )

        w3d = torch.nn.Parameter(init.clone().view(num_heads, head_dim, in_features))
        opt3d = optimizer_cls([w3d], **opt_kwargs)

        for step in range(n_steps):
            torch.manual_seed(100 + step)
            g = torch.randn(num_heads * head_dim, in_features, device=DEVICE)
            w2d.grad = g.clone()
            w3d.grad = g.view(num_heads, head_dim, in_features).clone()
            opt2d.step()
            opt3d.step()

        torch.testing.assert_close(
            w2d.data.view(num_heads, head_dim, in_features), w3d.data
        )

    def test_muon_matches_3d(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01))

    def test_muon_matches_3d_nesterov(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01, nesterov=True))

    def test_dion2_matches_3d(self):
        from dion import Dion2
        self._run_parity(Dion2, dict(lr=0.01, fraction=0.5))

    def test_dion2_matches_3d_full_fraction(self):
        from dion import Dion2
        self._run_parity(Dion2, dict(lr=0.01, fraction=1.0))

    def test_normuon_matches_3d(self):
        from dion import NorMuon
        self._run_parity(NorMuon, dict(lr=0.01))

    def test_nordion2_matches_3d(self):
        from dion import NorDion2
        self._run_parity(NorDion2, dict(lr=0.01, fraction=0.5))
    
    def test_nordion2_matches_3d_full_fraction(self):
        from dion import NorDion2
        self._run_parity(NorDion2, dict(lr=0.01, fraction=1.0))

    def test_muon_invalid_num_heads(self):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(30, 16, device=DEVICE))
        w.grad = torch.randn_like(w)
        # 30 not divisible by 4
        opt = Muon([{"params": [w], "num_heads": 4}], lr=0.01)
        with pytest.raises(ValueError, match="num_heads"):
            opt.step()

    def test_muon_num_heads_rejects_1d(self):
        from dion import Muon
        # 1D params never reach _prepare_head_split (Muon asserts ndim >= 2),
        # but a 3D param with num_heads>1 should raise since the reshape assumes 2D.
        w = torch.nn.Parameter(torch.randn(4, 8, 16, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon([{"params": [w], "num_heads": 2}], lr=0.01)
        with pytest.raises(ValueError, match="2D"):
            opt.step()

    def test_megabatch(self):
        """Multiple 2D params with same shape + num_heads should be megabatched."""
        from dion import Muon
        num_heads, head_dim, in_features = 4, 8, 16
        params = _make_params([(num_heads * head_dim, in_features)] * 3)
        opt = Muon([{"params": params, "num_heads": num_heads}], lr=0.01)
        for step in range(3):
            torch.manual_seed(100 + step)
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()

    @pytest.mark.parametrize("bad", [0, -1, 2.0, "4", True])
    def test_invalid_num_heads_raises(self, bad):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(32, 16, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon([{"params": [w], "num_heads": bad}], lr=0.01)
        with pytest.raises(ValueError, match="num_heads"):
            opt.step()

    def test_num_heads_one_is_noop(self):
        # num_heads=1 is semantically equivalent to the default path; verify
        # it runs without error and doesn't accidentally hit the head-split code.
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(32, 16, device=DEVICE))
        w_ref = torch.nn.Parameter(w.data.clone())
        g = torch.randn_like(w)
        for step in range(2):
            w.grad = g.clone()
            w_ref.grad = g.clone()
        opt = Muon([{"params": [w], "num_heads": 1}], lr=0.01)
        opt_ref = Muon([w_ref], lr=0.01)
        opt.step()
        opt_ref.step()
        torch.testing.assert_close(w.data, w_ref.data)

    def test_flatten_true_incompatible(self):
        # flatten=True would collapse the per-head 3D view back to 2D, giving
        # the wrong update silently. It must raise.
        from dion import Muon
        num_heads, head_dim, in_features = 4, 8, 16
        w = torch.nn.Parameter(
            torch.randn(num_heads * head_dim, in_features, device=DEVICE)
        )
        w.grad = torch.randn_like(w)
        opt = Muon(
            [{"params": [w], "num_heads": num_heads}], lr=0.01, flatten=True
        )
        with pytest.raises(ValueError, match="flatten"):
            opt.step()


# ---------------------------------------------------------------------------
# split_sizes per-group option (per-row-block Newton-Schulz on fused weights)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestSplitSizes:
    """The ``split_sizes`` param-group option orthogonalizes row blocks of a
    fused 2D weight independently (e.g. a fused QKV projection with unequal
    Q/K/V blocks under GQA), matching the update that separate per-block
    parameters would receive."""

    def _run_parity(
        self,
        optimizer_cls,
        opt_kwargs,
        split_sizes=(32, 16, 16),
        in_features=32,
        n_steps=3,
    ):
        torch.manual_seed(0)
        rows = sum(split_sizes)
        init = torch.randn(rows, in_features, device=DEVICE)

        fused = torch.nn.Parameter(init.clone())
        opt_fused = optimizer_cls(
            [{"params": [fused], "split_sizes": split_sizes}], **opt_kwargs
        )

        separate = [
            torch.nn.Parameter(block.clone())
            for block in init.split(list(split_sizes), dim=0)
        ]
        opt_separate = optimizer_cls(separate, **opt_kwargs)

        for step in range(n_steps):
            torch.manual_seed(100 + step)
            g = torch.randn(rows, in_features, device=DEVICE)
            fused.grad = g.clone()
            for p, gb in zip(separate, g.split(list(split_sizes), dim=0)):
                p.grad = gb.clone()
            opt_fused.step()
            opt_separate.step()

        # Newton-Schulz runs in bf16 and the per-block lr rescale rounds
        # differently than the separate-param adjusted lr, so allow bf16-level
        # tolerance rather than exact equality.
        torch.testing.assert_close(
            fused.data,
            torch.cat([p.data for p in separate], dim=0),
            rtol=1e-2,
            atol=1e-3,
        )

    def test_muon_matches_separate(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01))

    def test_muon_matches_separate_rms_norm(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01, adjust_lr="rms_norm"))

    def test_muon_matches_separate_no_adjust(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01, adjust_lr=None))

    def test_muon_matches_separate_nesterov(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01, nesterov=True))

    def test_muon_equal_blocks(self):
        from dion import Muon
        self._run_parity(Muon, dict(lr=0.01), split_sizes=(16, 16, 16, 16))

    def test_normuon_matches_separate(self):
        from dion import NorMuon
        self._run_parity(NorMuon, dict(lr=0.01))

    def test_normuon_matches_separate_rms_norm(self):
        from dion import NorMuon
        self._run_parity(NorMuon, dict(lr=0.01, adjust_lr="rms_norm"))

    def test_momentum_state_stays_fused(self):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon([{"params": [w], "split_sizes": (32, 16, 16)}], lr=0.01)
        opt.step()
        assert opt.state[w]["momentum"].shape == w.shape

    def test_megabatch(self):
        """Multiple same-shape params in one split_sizes group are megabatched."""
        from dion import Muon
        params = _make_params([(64, 32)] * 3)
        opt = Muon([{"params": params, "split_sizes": (32, 16, 16)}], lr=0.01)
        for step in range(3):
            torch.manual_seed(100 + step)
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()

    def test_sum_mismatch_raises(self):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(60, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon([{"params": [w], "split_sizes": (32, 16, 16)}], lr=0.01)
        with pytest.raises(ValueError, match="split_sizes"):
            opt.step()

    def test_rejects_3d_param(self):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(4, 16, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon([{"params": [w], "split_sizes": (2, 2)}], lr=0.01)
        with pytest.raises(ValueError, match="2D"):
            opt.step()

    def test_incompatible_with_num_heads(self):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon(
            [{"params": [w], "split_sizes": (32, 16, 16), "num_heads": 4}],
            lr=0.01,
        )
        with pytest.raises(ValueError, match="num_heads"):
            opt.step()

    def test_incompatible_with_flatten(self):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon(
            [{"params": [w], "split_sizes": (32, 16, 16)}], lr=0.01, flatten=True
        )
        with pytest.raises(ValueError, match="flatten"):
            opt.step()

    @pytest.mark.parametrize("bad", [3, (32,), (32, 0), (32, -1), (32, 2.0), (32, "16"), (32, True)])
    def test_invalid_split_sizes_raises(self, bad):
        from dion import Muon
        w = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Muon([{"params": [w], "split_sizes": bad}], lr=0.01)
        with pytest.raises(ValueError, match="split_sizes"):
            opt.step()

    def test_dion2_not_supported(self):
        from dion import Dion2
        w = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = Dion2([{"params": [w], "split_sizes": (32, 16, 16)}], lr=0.01)
        with pytest.raises(NotImplementedError, match="split_sizes"):
            opt.step()


# ---------------------------------------------------------------------------
# Mixed param groups (matrix + scalar)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestMixedParamGroups:
    def _make_model_params(self):
        """Simulate a model with matrix weights + vector biases."""
        torch.manual_seed(42)
        weights = [
            torch.nn.Parameter(torch.randn(64, 128, device=DEVICE)),
            torch.nn.Parameter(torch.randn(128, 64, device=DEVICE)),
        ]
        biases = [
            torch.nn.Parameter(torch.randn(64, device=DEVICE)),
            torch.nn.Parameter(torch.randn(128, device=DEVICE)),
        ]
        return weights, biases

    def test_muon_with_adamw_scalars(self):
        from dion import Muon
        weights, biases = self._make_model_params()
        opt = Muon([
            {"params": weights},
            {"params": biases, "algorithm": "adamw"},
        ], lr=0.01)
        for step in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()

    def test_normuon_with_lion_scalars(self):
        from dion import NorMuon
        weights, biases = self._make_model_params()
        opt = NorMuon([
            {"params": weights},
            {"params": biases, "algorithm": "lion"},
        ], lr=0.01)
        for step in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()

    def test_dion2_with_adamw_scalars(self):
        from dion import Dion2
        weights, biases = self._make_model_params()
        opt = Dion2([
            {"params": weights},
            {"params": biases, "algorithm": "adamw"},
        ], lr=0.01)
        for step in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()


# ---------------------------------------------------------------------------
# Hyperparameter validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestValidation:
    def test_muon_invalid_lr(self):
        from dion import Muon
        with pytest.raises(ValueError):
            Muon(_make_params([(32, 64)]), lr=-0.01)

    def test_normuon_invalid_mu(self):
        from dion import NorMuon
        with pytest.raises(ValueError):
            NorMuon(_make_params([(32, 64)]), mu=-1.0)

    def test_dion2_invalid_fraction(self):
        from dion import Dion2
        with pytest.raises(ValueError):
            Dion2(_make_params([(32, 64)]), fraction=0.0)
        with pytest.raises(ValueError):
            Dion2(_make_params([(32, 64)]), fraction=1.5)

    def test_invalid_adjust_lr(self):
        from dion import Muon
        with pytest.raises(ValueError):
            Muon(_make_params([(32, 64)]), adjust_lr="invalid")

    def test_1d_param_rejected(self):
        """Ortho optimizers should reject 1D parameters."""
        from dion import Muon
        params = [torch.nn.Parameter(torch.randn(64, device=DEVICE))]
        opt = Muon([{"params": params}], lr=0.01)
        params[0].grad = torch.randn_like(params[0])
        with pytest.raises(AssertionError):
            opt.step()


# ---------------------------------------------------------------------------
# No-grad params (some params have no gradient)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestSparseGradients:
    def test_some_params_no_grad(self):
        """Optimizer should handle params with no gradient gracefully."""
        from dion import Muon
        params = _make_params([(64, 128), (64, 128), (64, 128)])
        opt = Muon(params, lr=0.01)
        # Only give gradients to first and third params
        params[0].grad = torch.randn_like(params[0])
        params[2].grad = torch.randn_like(params[2])
        opt.step()  # Should not crash

    def test_no_grads_at_all(self):
        """Step with zero gradients should be a no-op."""
        from dion import Muon
        params = _make_params([(64, 128)])
        before = params[0].data.clone()
        opt = Muon(params, lr=0.01)
        opt.step()
        assert torch.equal(params[0].data, before)


# ---------------------------------------------------------------------------
# Step timing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestTiming:
    def test_optimizer_step_takes_time(self):
        """Optimizer step should take measurable wall-clock time."""
        from dion import Muon
        params = _make_params([(256, 512)] * 10)
        opt = Muon(params, lr=0.01)
        for p in params:
            p.grad = torch.randn_like(p)

        # Warmup (torch.compile)
        opt.step()
        for p in params:
            p.grad = torch.randn_like(p)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        elapsed_ms = 1000 * (time.perf_counter() - t0)

        # Step should complete in reasonable time (> 0, < 10s)
        assert elapsed_ms > 0, "Step took zero time"
        assert elapsed_ms < 10_000, f"Step took too long: {elapsed_ms:.0f}ms"

    def test_timer_accumulates_across_steps(self):
        """Simulated training timer should accumulate monotonically."""
        from dion import NorMuon
        params = _make_params([(64, 128)] * 3)
        opt = NorMuon(params, lr=0.01)

        training_time_ms = 0.0
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        prev_time = 0.0
        for step in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()

            torch.cuda.synchronize()
            training_time_ms = 1000 * (time.perf_counter() - t0)
            assert training_time_ms > prev_time, (
                f"Timer did not advance: step {step}, "
                f"prev={prev_time:.2f}ms, now={training_time_ms:.2f}ms"
            )
            prev_time = training_time_ms

    def test_perf_counter_resolution(self):
        """time.perf_counter should have sub-millisecond resolution."""
        t0 = time.perf_counter()
        # Busy wait briefly
        x = 0
        for _ in range(1000):
            x += 1
        t1 = time.perf_counter()
        # Should register nonzero time
        assert t1 > t0


# ---------------------------------------------------------------------------
# Empty FSDP2 local shards (Dion2 / NorDion2)
# ---------------------------------------------------------------------------
# FSDP2 contiguous chunking leaves some ranks with an empty (size-0) local
# shard along the sharded dim when the param's sharded dim is smaller than
# world_size (or doesn't divide evenly to fill all ranks). dion2_pre_orthogonalize
# computes topk(k>=1) over that dim, which raised "k not in range for dimension"
# under torch.compile's fake-tensor pass and deadlocked the remaining ranks at
# the NCCL all-to-all. These tests pin the empty-shard short-circuit.

def test_dion2_pre_orthogonalize_empty_row_shard():
    from dion.dion2 import dion2_pre_orthogonalize
    cols, n = 16, 4
    M = [torch.zeros(0, cols) for _ in range(n)]
    G = [torch.randn(0, cols) for _ in range(n)]
    U, indices = dion2_pre_orthogonalize(
        G=G, M=M, fraction=0.5, ef_decay=torch.tensor(0.95), select_dim=-2
    )
    assert len(U) == n and len(indices) == n
    for u, idx in zip(U, indices):
        assert tuple(u.shape) == (0, cols)
        assert u.dtype == torch.bfloat16
        assert tuple(idx.shape) == (0,)
        assert idx.dtype == torch.long


def test_dion2_pre_orthogonalize_empty_col_shard():
    from dion.dion2 import dion2_pre_orthogonalize
    rows, n = 16, 4
    M = [torch.zeros(rows, 0) for _ in range(n)]
    G = [torch.randn(rows, 0) for _ in range(n)]
    U, indices = dion2_pre_orthogonalize(
        G=G, M=M, fraction=0.5, ef_decay=torch.tensor(0.95), select_dim=-1
    )
    assert len(U) == n and len(indices) == n
    for u, idx in zip(U, indices):
        assert tuple(u.shape) == (rows, 0)
        assert u.dtype == torch.bfloat16
        assert tuple(idx.shape) == (0,)
        assert idx.dtype == torch.long


def _dion2_empty_shard_step_worker(rank, world_size, global_rows, cols, port, triton_post_ortho):
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor, Shard, init_device_mesh
    from dion import Dion2

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    mesh = init_device_mesh("cuda", (world_size,))
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(0)
    full = torch.randn(global_rows, cols, device=device)
    # Row-sharded over world_size: ranks beyond ceil(global_rows / world_size)
    # contiguous chunks hold an empty (0, cols) local shard.
    param = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
    before = param.to_local().clone()

    opt = Dion2([param], distributed_mesh=mesh, lr=0.01, triton_post_ortho=triton_post_ortho)
    for step in range(3):
        torch.manual_seed(step + 1)
        g = torch.randn(global_rows, cols, device=device)
        param.grad = distribute_tensor(g, mesh, [Shard(0)])
        opt.step()

    local_after = param.to_local()
    if local_after.shape[0] > 0:
        assert not torch.equal(local_after, before), f"rank {rank}: weights did not update"
    else:
        assert torch.equal(local_after, before), f"rank {rank}: empty shard mutated"
    dist.destroy_process_group()


# triton_post_ortho=True exercises dion2_post_orthogonalize_triton, whose
# B = x.numel() // (M * N) divides by zero on the empty (0, cols) shard unless
# the empty rank is skipped; the default (False) path no-ops via scatter_add_.
@pytest.mark.parametrize("triton_post_ortho", [False, True])
@pytest.mark.parametrize(
    "world_size, global_rows",
    [
        (2, 1),  # rank 1 holds an empty (0, cols) shard
        (4, 2),  # ranks 2 and 3 empty
        (4, 5),  # chunk sizes (2, 2, 1, 0): rank 3 empty (mirrors sparse-3b (18, D)/8)
    ],
)
def test_dion2_optimizer_step_with_empty_shard(world_size, global_rows, triton_post_ortho):
    import torch.multiprocessing as mp

    if CUDA_DEVICE_COUNT < world_size:
        pytest.skip(f"needs >= {world_size} CUDA devices for NCCL alltoall")
    if triton_post_ortho:
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not installed")
    port = 29800 + world_size * 10 + global_rows + (1000 if triton_post_ortho else 0)
    mp.spawn(
        _dion2_empty_shard_step_worker,
        args=(world_size, global_rows, 16, port, triton_post_ortho),
        nprocs=world_size,
        join=True,
    )
