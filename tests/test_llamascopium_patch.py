"""The DTensor-safe encoder init keeps llamascopium's single-device behavior."""

import einops
import torch

from readout.crosscoder.wu_adapter import build_crosscoder


def test_encoder_init_is_scaled_decoder_transpose():
    torch.manual_seed(0)
    cc = build_crosscoder(n_snapshots=3, d_model=8, expansion_factor=2, init_encoder_with_decoder_transpose_factor=0.0)
    cc.init_encoder_with_decoder_transpose(factor=0.5)
    expected = einops.rearrange(cc.W_D, "n_heads d_sae d_model -> n_heads d_model d_sae") * 0.5
    torch.testing.assert_close(cc.W_E.detach(), expected.detach())
