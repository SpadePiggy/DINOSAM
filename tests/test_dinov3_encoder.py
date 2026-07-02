"""Tests for DINOv3MonaEncoder with DPTHead (dynamic layer support).

These tests require a DINOv3 checkpoint which is not available in CI/test envs.
They are skipped automatically when no checkpoint is found.
"""

import os
import pytest
import torch

# Common checkpoint search paths
_CKPT_CANDIDATES = [
    os.path.expanduser("~/lxt/dinov3/checkpoints/dinov3_vitb16.pth"),
    os.path.expanduser("~/checkpoints/dinov3_vitb16.pth"),
    "/data/checkpoints/dinov3_vitb16.pth",
]

CKPT_PATH = next((p for p in _CKPT_CANDIDATES if os.path.isfile(p)), None)

skip_no_ckpt = pytest.mark.skipif(
    CKPT_PATH is None,
    reason="DINOv3 checkpoint not available in test environment",
)


@skip_no_ckpt
class TestDINOv3MonaEncoderDPTHead:
    """Verify DINOv3MonaEncoder works with DPTHead for various layer counts."""

    def _make_encoder(self, dpt_layers):
        from dinosam.model.dinov3_encoder import DINOv3MonaEncoder

        return DINOv3MonaEncoder(
            checkpoint_path=CKPT_PATH,
            adapter_type="dpt_simple",
            dpt_layers=dpt_layers,
        )

    @torch.no_grad()
    def test_encoder_3_layers(self):
        """Encoder works with 3 intermediate layers."""
        enc = self._make_encoder(dpt_layers=[2, 5, 8])
        assert enc.dpt_head.n_layers == 3
        images = torch.randn(1, 3, 1024, 1024)
        out, input_size = enc(images)
        assert out.shape == (1, 256, 64, 64)

    @torch.no_grad()
    def test_encoder_12_layers(self):
        """Encoder works with all 12 intermediate layers."""
        enc = self._make_encoder(dpt_layers=list(range(12)))
        assert enc.dpt_head.n_layers == 12
        images = torch.randn(1, 3, 1024, 1024)
        out, input_size = enc(images)
        assert out.shape == (1, 256, 64, 64)

    @torch.no_grad()
    def test_encoder_default_layers(self):
        """Default dpt_layers (None) gives 4 layers, backward compat."""
        enc = self._make_encoder(dpt_layers=None)
        assert enc.dpt_head.n_layers == 4
        images = torch.randn(1, 3, 1024, 1024)
        out, input_size = enc(images)
        assert out.shape == (1, 256, 64, 64)
