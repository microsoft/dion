"""Functional tests for all optimizers in the dion package.

Tests cover:
- Basic operation: each optimizer can run multiple steps without error
- Determinism: same seed produces same results
- Parameter update: parameters actually change after a step
- Mixed param groups: matrix params (ortho) + vector params (scalar opt)
- All optimizer-specific options (nesterov, cautious_wd, base_opt, etc.)
- Step timing: optimizer step takes measurable wall-clock time
"""

import pytest
import time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

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
# MuonH
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestMuonH:
    def test_basic(self):
        from dion import MuonH
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(MuonH, params, dict(lr=0.01))

    @pytest.mark.parametrize("normalization", [None, "None", "neuron", "short_axis"])
    def test_normalization_options(self, normalization):
        from dion import MuonH
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(MuonH, params, dict(lr=0.01, normalization=normalization))

    @pytest.mark.parametrize("normalization", [None, "neuron", "short_axis"])
    def test_preserves_frobenius_norm(self, normalization):
        from dion import MuonH
        params = _make_params([(64, 128), (128, 64)])
        before = [p.data.float().norm() for p in params]
        _run_steps(
            MuonH,
            params,
            dict(lr=0.01, normalization=normalization),
            n_steps=2,
        )
        after = [p.data.float().norm() for p in params]
        for a, b in zip(after, before):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_variance_shapes(self):
        from dion import MuonH
        wide = torch.nn.Parameter(torch.randn(64, 128, device=DEVICE))
        tall = torch.nn.Parameter(torch.randn(128, 64, device=DEVICE))

        opt = MuonH([wide], lr=0.01, normalization="neuron")
        wide.grad = torch.randn_like(wide)
        opt.step()
        assert opt.state[wide]["variance_normalization"].shape == (64, 1)

        opt = MuonH([wide], lr=0.01, normalization="short_axis")
        wide.grad = torch.randn_like(wide)
        opt.step()
        assert opt.state[wide]["variance_normalization"].shape == (1, 128)

        opt = MuonH([tall], lr=0.01, normalization="short_axis")
        tall.grad = torch.randn_like(tall)
        opt.step()
        assert opt.state[tall]["variance_normalization"].shape == (128, 1)

    def test_zero_radius_rejected(self):
        from dion import MuonH
        param = torch.nn.Parameter(torch.zeros(64, 128, device=DEVICE))
        param.grad = torch.randn_like(param)
        opt = MuonH([param], lr=0.01)
        with pytest.raises(ValueError, match="non-zero"):
            opt.step()

    def test_invalid_normalization(self):
        from dion import MuonH
        with pytest.raises(ValueError):
            MuonH(_make_params([(32, 64)]), normalization="invalid")

    @pytest.mark.parametrize("normalization", [None, "neuron", "short_axis"])
    def test_3d_experts_match_separate_params(self, normalization):
        from dion import MuonH

        torch.manual_seed(11)
        init = torch.randn(4, 32, 16, device=DEVICE)

        packed = torch.nn.Parameter(init.clone())
        packed_opt = MuonH([packed], lr=0.01, normalization=normalization)

        separate = [
            torch.nn.Parameter(init[i].clone()) for i in range(init.shape[0])
        ]
        separate_opt = MuonH(separate, lr=0.01, normalization=normalization)

        for step in range(3):
            torch.manual_seed(200 + step)
            grad = torch.randn_like(init)
            packed.grad = grad.clone()
            for i, p in enumerate(separate):
                p.grad = grad[i].clone()
            packed_opt.step()
            separate_opt.step()

        torch.testing.assert_close(
            packed.data,
            torch.stack([p.data for p in separate]),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("normalization", [None, "neuron", "short_axis"])
    def test_flatten_3d_matches_explicit_2d(self, normalization):
        from dion import MuonH

        torch.manual_seed(13)
        init = torch.randn(4, 32, 16, device=DEVICE)

        packed = torch.nn.Parameter(init.clone())
        packed_opt = MuonH(
            [packed], lr=0.01, normalization=normalization, flatten=True
        )

        flat = torch.nn.Parameter(init.clone().flatten(start_dim=1))
        flat_opt = MuonH(
            [flat], lr=0.01, normalization=normalization, flatten=False
        )

        for step in range(3):
            torch.manual_seed(300 + step)
            grad = torch.randn_like(init)
            packed.grad = grad.clone()
            flat.grad = grad.flatten(start_dim=1).clone()
            packed_opt.step()
            flat_opt.step()

        torch.testing.assert_close(
            packed.data.flatten(start_dim=1),
            flat.data,
            rtol=1e-5,
            atol=1e-5,
        )


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

    @pytest.mark.parametrize("normalization", ["neuron", "short_axis"])
    def test_normalization_options(self, normalization):
        from dion import NorMuon
        params = _make_params([(64, 128), (128, 64)])
        _run_steps(NorMuon, params, dict(lr=0.01, normalization=normalization))

    def test_variance_shapes(self):
        from dion import NorMuon
        wide = torch.nn.Parameter(torch.randn(64, 128, device=DEVICE))
        tall = torch.nn.Parameter(torch.randn(128, 64, device=DEVICE))

        opt = NorMuon([wide], lr=0.01, normalization="neuron")
        wide.grad = torch.randn_like(wide)
        opt.step()
        assert opt.state[wide]["variance_neuron"].shape == (64, 1)

        opt = NorMuon([wide], lr=0.01, normalization="short_axis")
        wide.grad = torch.randn_like(wide)
        opt.step()
        assert opt.state[wide]["variance_neuron"].shape == (1, 128)

        opt = NorMuon([tall], lr=0.01, normalization="short_axis")
        tall.grad = torch.randn_like(tall)
        opt.step()
        assert opt.state[tall]["variance_neuron"].shape == (128, 1)

    def test_short_axis_matches_neuron_for_tall(self):
        """For tall matrices the reduction axis is the same, so short_axis
        and neuron should produce numerically equivalent updates.

        The two paths are not bitwise identical: ``neuron`` does the variance
        reduction in the param dtype via ``(U*U).mean(-1)``, while
        ``short_axis`` reduces in fp32 via ``square().sum(-1)/N`` and casts
        back. Tolerances are sized to allow that difference plus its
        accumulation across 3 steps, and to avoid flaking on hardware whose
        matmul/reduction kernels differ from Hopper's.
        """
        from dion import NorMuon
        p1 = _make_params([(128, 64)])
        r1 = _run_steps(NorMuon, p1, dict(lr=0.01, normalization="neuron"))
        p2 = _make_params([(128, 64)])
        r2 = _run_steps(NorMuon, p2, dict(lr=0.01, normalization="short_axis"))
        torch.testing.assert_close(r1[0], r2[0], rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("normalization", ["neuron", "short_axis"])
    def test_flatten_3d_matches_explicit_2d(self, normalization):
        from dion import NorMuon

        torch.manual_seed(17)
        init = torch.randn(4, 32, 16, device=DEVICE)

        packed = torch.nn.Parameter(init.clone())
        packed_opt = NorMuon(
            [packed], lr=0.01, normalization=normalization, flatten=True
        )

        flat = torch.nn.Parameter(init.clone().flatten(start_dim=1))
        flat_opt = NorMuon(
            [flat], lr=0.01, normalization=normalization, flatten=False
        )

        for step in range(3):
            torch.manual_seed(400 + step)
            grad = torch.randn_like(init)
            packed.grad = grad.clone()
            flat.grad = grad.flatten(start_dim=1).clone()
            packed_opt.step()
            flat_opt.step()

        torch.testing.assert_close(
            packed.data.flatten(start_dim=1),
            flat.data,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_invalid_normalization(self):
        from dion import NorMuon
        with pytest.raises(ValueError):
            NorMuon(_make_params([(32, 64)]), normalization="invalid")

    def test_megabatch_same_shape(self):
        from dion import NorMuon
        params = _make_params([(64, 128)] * 5)
        _run_steps(NorMuon, params, dict(lr=0.01))


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

    def test_normuon_matches_3d_short_axis(self):
        from dion import NorMuon
        self._run_parity(NorMuon, dict(lr=0.01, normalization="short_axis"))

    def test_muonh_rejects_num_heads(self):
        from dion import MuonH
        w = torch.nn.Parameter(torch.randn(32, 16, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = MuonH([{"params": [w], "num_heads": 4}], lr=0.01)
        with pytest.raises(ValueError, match="does not support num_heads"):
            opt.step()

    def test_muonh_rejects_num_heads_short_axis(self):
        from dion import MuonH
        w = torch.nn.Parameter(torch.randn(32, 16, device=DEVICE))
        w.grad = torch.randn_like(w)
        opt = MuonH(
            [{"params": [w], "num_heads": 4}],
            lr=0.01,
            normalization="short_axis",
        )
        with pytest.raises(ValueError, match="does not support num_heads"):
            opt.step()

    def test_muonh_num_heads_one_is_noop(self):
        from dion import MuonH
        w = torch.nn.Parameter(torch.randn(32, 16, device=DEVICE))
        w_ref = torch.nn.Parameter(w.data.clone())
        g = torch.randn_like(w)
        for _ in range(2):
            w.grad = g.clone()
            w_ref.grad = g.clone()
        opt = MuonH([{"params": [w], "num_heads": 1}], lr=0.01)
        opt_ref = MuonH([w_ref], lr=0.01)
        opt.step()
        opt_ref.step()
        torch.testing.assert_close(w.data, w_ref.data)

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
# DTensor-sharded MuonH smoke test (multi-process, NCCL)
# ---------------------------------------------------------------------------
#
# Most of MuonH's complexity is the FSDP2 sharding/all-reduce paths
# (radius init, normalization reduction, fused post-step reduction). The
# single-GPU tests above don't exercise any of that. This launches two
# real ranks via mp.spawn, runs a few steps on a 2-way Shard(dim) DTensor,
# and verifies that (a) it completes without error, (b) the parameter
# Frobenius norm stays at the initial radius, and (c) the parameter is
# updated identically across ranks.


def _muonh_dtensor_worker(rank, world_size, port, shard_dim, normalization, queue):
    import os, traceback
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_tensor, Shard

        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fs",))

        from dion import MuonH

        device = f"cuda:{rank}"
        torch.manual_seed(42)
        full = torch.randn(64, 128, device=device)
        initial_norm = float(full.float().norm())

        param = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(shard_dim)]))
        opt = MuonH(
            [param],
            distributed_mesh=mesh,
            lr=0.01,
            normalization=normalization,
        )

        for step in range(3):
            torch.manual_seed(100 + step)
            grad_full = torch.randn(64, 128, device=device)
            param.grad = distribute_tensor(grad_full, mesh, [Shard(shard_dim)])
            opt.step()

        # Reconstruct the full tensor and report its Frobenius norm.
        full_param = param.full_tensor()
        final_norm = float(full_param.float().norm())
        # Hash a fingerprint of the full tensor for cross-rank consistency.
        fingerprint = float(full_param.float().sum())
        queue.put((rank, "ok", initial_norm, final_norm, fingerprint))
    except Exception:
        queue.put((rank, "err", traceback.format_exc(), None, None))
    finally:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires >=2 CUDA devices"
)
class TestMuonHDTensor:
    @pytest.mark.parametrize(
        "shard_dim,normalization",
        [
            (0, None),          # tall, batch-sharded (matrix dims unsharded)
            (-2, "neuron"),     # matrix-sharded along rows, neuron normalization
            (-1, "neuron"),     # matrix-sharded along last dim (MuonH-only path)
            (-2, "short_axis"), # the case the reviewer specifically called out
            (-1, "short_axis"),
        ],
    )
    def test_sharded_step(self, shard_dim, normalization):
        import torch.multiprocessing as mp

        world_size = 2
        # Deterministic-ish port per param combo to avoid collisions across runs.
        port = 29500 + (abs(hash((shard_dim, str(normalization)))) % 500)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_muonh_dtensor_worker,
                args=(rank, world_size, port, shard_dim, normalization, queue),
            )
            for rank in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
        results = []
        while not queue.empty():
            results.append(queue.get())
        for p in procs:
            if p.is_alive():
                p.terminate()
            assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"

        assert len(results) == world_size, f"Expected {world_size} results, got {results}"
        for rank, status, *info in results:
            assert status == "ok", f"Rank {rank} failed:\n{info[0]}"

        # Frobenius radius preserved (core MuonH invariant) across ranks.
        norms = [(r[2], r[3]) for r in results]
        for initial, final in norms:
            assert abs(final - initial) / initial < 1e-3, (
                f"Frobenius radius drifted: initial={initial}, final={final}"
            )
        # Both ranks should see the same reconstructed parameter.
        fps = [r[4] for r in results]
        assert abs(fps[0] - fps[1]) < 1e-3 * max(1.0, abs(fps[0])), (
            f"Reconstructed parameter differs across ranks: {fps}"
        )
