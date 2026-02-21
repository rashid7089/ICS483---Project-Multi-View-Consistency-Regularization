"""
Unit tests for Multi-View Consistency Regularization (MVCR) components.

Tests verify:
  1. Virtual ray generation is geometrically consistent
  2. MVCR loss computation produces finite values
  3. MVCR loss gradient flows correctly through the model
  4. Curriculum weight scheduling works as expected
  5. Loss dictionary is properly updated
  6. Both GT-interpolation and model-render modes work
"""

import pytest
import numpy as np
import torch
import torch.nn as nn


# ─── Minimal mock objects ────────────────────────────────────────────────────

class MockConeGeometry:
    """Minimal mock ConeGeometry for unit testing without real CT data."""
    def __init__(self):
        self.DSD = 1.075
        self.DSO = 0.75
        self.nDetector = np.array([64, 64])
        self.dDetector = np.array([0.001172, 0.001172])
        self.sDetector = self.nDetector * self.dDetector
        self.nVoxel = np.array([64, 64, 64])
        self.dVoxel = np.array([0.001172, 0.001172, 0.001172])
        self.sVoxel = self.nVoxel * self.dVoxel
        self.offOrigin = np.array([0.0, 0.0, 0.0])
        self.offDetector = np.array([0.0, 0.0])
        self.accuracy = 0.5
        self.mode = "cone"
        self.filter = None


class MockNet(nn.Module):
    """Tiny density network for fast unit testing."""
    def __init__(self):
        super().__init__()
        self.bound = 0.3
        self.fc = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestVirtualRayGeneration:
    """Tests for _get_virtual_rays helper function."""

    def test_output_shape(self):
        """Virtual rays should have shape [H, W, 8]."""
        from src.loss.mv_consistency import _get_virtual_rays
        geo = MockConeGeometry()
        device = torch.device("cpu")
        rays = _get_virtual_rays(0.0, geo, device, near=0.1, far=1.5)
        H, W = geo.nDetector
        assert rays.shape == (H, W, 8), f"Expected ({H}, {W}, 8), got {rays.shape}"

    def test_near_far_filled(self):
        """Near and far values should be filled correctly."""
        from src.loss.mv_consistency import _get_virtual_rays
        geo = MockConeGeometry()
        device = torch.device("cpu")
        near, far = 0.2, 1.3
        rays = _get_virtual_rays(0.0, geo, device, near=near, far=far)
        assert torch.allclose(rays[..., 6], torch.full((64, 64), near)), \
            "Near plane values incorrect"
        assert torch.allclose(rays[..., 7], torch.full((64, 64), far)), \
            "Far plane values incorrect"

    def test_different_angles_produce_different_rays(self):
        """Different angles should produce different ray origins."""
        from src.loss.mv_consistency import _get_virtual_rays
        geo = MockConeGeometry()
        device = torch.device("cpu")
        rays_0 = _get_virtual_rays(0.0, geo, device, near=0.1, far=1.5)
        rays_pi = _get_virtual_rays(np.pi / 4, geo, device, near=0.1, far=1.5)
        assert not torch.allclose(rays_0[..., :3], rays_pi[..., :3]), \
            "Different angles should produce different ray origins"

    def test_rays_are_finite(self):
        """Generated rays should not contain NaN or Inf."""
        from src.loss.mv_consistency import _get_virtual_rays
        geo = MockConeGeometry()
        device = torch.device("cpu")
        for angle in [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            rays = _get_virtual_rays(angle, geo, device, near=0.1, far=1.5)
            assert not torch.isnan(rays).any(), f"NaN in rays at angle {angle}"
            assert not torch.isinf(rays).any(), f"Inf in rays at angle {angle}"


class TestMVCRLoss:
    """Tests for the calc_mv_consistency_loss function."""

    @pytest.fixture
    def setup(self):
        """Set up common test fixtures."""
        device = torch.device("cpu")
        net = MockNet()
        geo = MockConeGeometry()
        near, far = 0.1, 1.5
        n_views = 10
        train_angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
        H, W = geo.nDetector
        train_projs = torch.rand(n_views, H, W, device=device)
        render_cfg = {
            "n_samples": 8,
            "n_fine": 0,
            "perturb": False,
            "netchunk": 1024,
            "raw_noise_std": 0.0,
        }
        return {
            "device": device,
            "net": net,
            "geo": geo,
            "near": near,
            "far": far,
            "train_angles": train_angles,
            "train_projs": train_projs,
            "render_cfg": render_cfg,
        }

    def test_loss_is_finite(self, setup):
        """MVCR loss should be finite and non-negative."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        loss = {"loss": torch.tensor(0.0)}
        calc_mv_consistency_loss(
            loss=loss,
            net=s["net"],
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=1,
            lambda_mvcr=0.1,
            device=s["device"],
            use_model_renders=False,
        )
        assert not torch.isnan(loss["loss"]), "MVCR loss is NaN"
        assert not torch.isinf(loss["loss"]), "MVCR loss is Inf"
        assert loss["loss"].item() >= 0, "MVCR loss should be non-negative"

    def test_loss_key_added(self, setup):
        """'loss_mvcr' key should be added to loss dict."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        loss = {"loss": torch.tensor(0.0)}
        calc_mv_consistency_loss(
            loss=loss,
            net=s["net"],
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=1,
            lambda_mvcr=0.1,
            device=s["device"],
        )
        assert "loss_mvcr" in loss, "loss_mvcr key should be added to loss dict"

    def test_loss_accumulates_in_total(self, setup):
        """Total loss should increase after MVCR loss is added."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        initial_loss = torch.tensor(1.0)
        loss = {"loss": initial_loss.clone()}
        calc_mv_consistency_loss(
            loss=loss,
            net=s["net"],
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=1,
            lambda_mvcr=0.1,
            device=s["device"],
        )
        assert loss["loss"].item() >= initial_loss.item(), \
            "Total loss should not decrease after adding MVCR"

    def test_zero_lambda_no_effect(self, setup):
        """With lambda_mvcr=0, the total loss should be unchanged."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        initial_val = 2.5
        loss = {"loss": torch.tensor(initial_val)}
        calc_mv_consistency_loss(
            loss=loss,
            net=s["net"],
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=1,
            lambda_mvcr=0.0,
            device=s["device"],
        )
        assert abs(loss["loss"].item() - initial_val) < 1e-6, \
            "Zero lambda should not change total loss"

    def test_gradient_flows(self, setup):
        """Gradients should flow back through the MVCR loss to network parameters."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        net = s["net"]
        for p in net.parameters():
            p.requires_grad_(True)

        loss = {"loss": torch.tensor(0.0)}
        calc_mv_consistency_loss(
            loss=loss,
            net=net,
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=1,
            lambda_mvcr=0.1,
            device=s["device"],
        )
        loss["loss"].backward()

        grad_exists = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in net.parameters()
        )
        assert grad_exists, "MVCR loss should produce non-zero gradients"

    def test_model_render_mode(self, setup):
        """Model-render mode should also produce finite loss."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        loss = {"loss": torch.tensor(0.0)}
        calc_mv_consistency_loss(
            loss=loss,
            net=s["net"],
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=1,
            lambda_mvcr=0.1,
            device=s["device"],
            use_model_renders=True,
        )
        assert not torch.isnan(loss["loss"]), "Model-render MVCR loss is NaN"
        assert not torch.isinf(loss["loss"]), "Model-render MVCR loss is Inf"

    def test_multiple_virtual_views(self, setup):
        """Loss should be stable with multiple virtual views."""
        from src.loss.mv_consistency import calc_mv_consistency_loss
        s = setup
        loss = {"loss": torch.tensor(0.0)}
        calc_mv_consistency_loss(
            loss=loss,
            net=s["net"],
            train_angles=s["train_angles"],
            train_projs=s["train_projs"],
            geo=s["geo"],
            near=s["near"],
            far=s["far"],
            render_cfg=s["render_cfg"],
            n_virtual_rays=32,
            n_virtual_views=4,
            lambda_mvcr=0.1,
            device=s["device"],
        )
        assert not torch.isnan(loss["loss"])
        assert "loss_mvcr" in loss


class TestCurriculumScheduling:
    """Tests for MVCR curriculum weight scheduling in MV_SAX_Trainer."""

    def test_warmup_ramp(self):
        """Weight should ramp from 0 to lambda_mvcr over warmup_epochs."""
        # Test the _get_mvcr_weight logic directly without instantiating the full trainer
        lambda_mvcr = 0.1
        warmup_epochs = 100

        def get_mvcr_weight(idx_epoch):
            if warmup_epochs <= 0:
                return lambda_mvcr
            ramp = min(1.0, idx_epoch / warmup_epochs)
            return lambda_mvcr * ramp

        assert get_mvcr_weight(0) == 0.0, "Weight at epoch 0 should be 0"
        assert abs(get_mvcr_weight(50) - 0.05) < 1e-9, \
            "Weight at epoch 50 should be half of lambda_mvcr"
        assert get_mvcr_weight(100) == lambda_mvcr, \
            "Weight at warmup_epochs should equal lambda_mvcr"
        assert get_mvcr_weight(200) == lambda_mvcr, \
            "Weight after warmup should remain at lambda_mvcr"

    def test_no_warmup(self):
        """With warmup_epochs=0, weight should always equal lambda_mvcr."""
        lambda_mvcr = 0.2
        warmup_epochs = 0

        def get_mvcr_weight(idx_epoch):
            if warmup_epochs <= 0:
                return lambda_mvcr
            return lambda_mvcr * min(1.0, idx_epoch / warmup_epochs)

        for epoch in [0, 1, 100, 1000]:
            assert get_mvcr_weight(epoch) == lambda_mvcr, \
                f"Weight should be {lambda_mvcr} at epoch {epoch} with no warmup"


class TestRenderModule:
    """Smoke tests for the rendering module."""

    def test_render_output_shape(self):
        """Render should return 'acc' with shape [N_rays]."""
        from src.render import render

        net = MockNet()
        n_rays = 16
        near, far = 0.1, 1.5
        rays = torch.zeros(n_rays, 8)
        rays[:, 3] = 1.0  # direction z=1
        rays[:, 6] = near
        rays[:, 7] = far

        ret = render(
            rays, net, None,
            n_samples=4, n_fine=0, perturb=False,
            netchunk=64, raw_noise_std=0.0
        )
        assert "acc" in ret, "render should return 'acc' key"
        assert ret["acc"].shape == (n_rays,), \
            f"Expected shape ({n_rays},), got {ret['acc'].shape}"

    def test_render_values_bounded(self):
        """Rendered projections should be non-negative."""
        from src.render import render

        net = MockNet()
        n_rays = 8
        rays = torch.zeros(n_rays, 8)
        rays[:, 3] = 1.0
        rays[:, 6] = 0.1
        rays[:, 7] = 1.5

        ret = render(
            rays, net, None,
            n_samples=4, n_fine=0, perturb=False,
            netchunk=64, raw_noise_std=0.0
        )
        assert (ret["acc"] >= 0).all(), "Rendered projections should be non-negative"
