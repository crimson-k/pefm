"""Small CPU check for the encoder-to-RSSM evaluator path."""

import torch
import pytest
from omegaconf import OmegaConf

from pefm.models import (
    build_evaluator,
)
from pefm.models.dit_hidden import SpatiallyAlignedAdapter
from pefm.models.vjepa_adapter import align_row_major_tokens, row_major_token_grid


def make_rssm_config():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    cfg = OmegaConf.create({"model": OmegaConf.merge(base, size)})
    cfg.model.rssm.device = "cpu"
    cfg.model.rssm.initial = "zeros"
    return cfg


def test_spatial_grid_alignment():
    coordinates = torch.arange(1200, dtype=torch.float32).reshape(1, 1, 1200, 1)
    grid = row_major_token_grid(coordinates)
    aligned = align_row_major_tokens(coordinates)
    assert grid.shape == (1, 1, 1, 30, 40, 1)
    assert aligned.shape == (1, 1, 1, 15, 20, 1)
    assert grid[0, 0, 0, 1, 0, 0] == 40
    assert aligned[0, 0, 0, 0, 0, 0] == 20.5


@pytest.mark.parametrize("batch_size", [1, 2])
def test_clean_latent_rssm_flow(batch_size):
    cfg = make_rssm_config()
    cfg.device = "cpu"
    evaluator = build_evaluator(cfg)
    latents = torch.randn(batch_size, 48, 21, 12, 16, requires_grad=True)
    reset = torch.zeros(batch_size, 21, dtype=torch.bool)
    reset[:, 0] = True
    output = evaluator({
        "clean_latents": latents,
        "action_delta": torch.randn(batch_size, 21, 14),
        "reset": reset,
    })
    assert not hasattr(evaluator, "vjepa_adapter")
    assert evaluator.token_aggregator.attn.kdim == 48
    assert output["latent_tokens"].shape == (batch_size, 21, 192, 48)
    assert output["embed"].shape == (batch_size, 21, 512)
    assert output["posterior_reconstruction"].shape == (batch_size, 21, 1408)
    assert output["posterior_logits"].shape == (batch_size, 21, 32, 16)
    losses = evaluator.compute_loss(
        output, reset, free_nats=0.0, target_embed=torch.randn(batch_size, 21, 1408),
    )
    losses["total"].backward()
    assert torch.isfinite(latents.grad).all() and latents.grad.norm() > 0
    assert evaluator.token_aggregator.attn.k_proj_weight.grad.norm() > 0
    assert evaluator.token_aggregator.queries.grad.norm() > 0
    assert next(evaluator.rssm._deter_net.parameters()).grad.norm() > 0


def test_spatial_adapter_requires_explicit_view_layout():
    adapter = SpatiallyAlignedAdapter(
        dit_dim=32, vjepa_dim=8, target_grid=(15, 20), num_views=2, hidden_dim=32,
    )
    hidden = torch.randn(1, 21 * 60 * 40, 32)
    tokens = adapter(hidden, dit_grid=(21, 60, 40))
    assert tokens.shape == (1, 21, 2, 15, 20, 8)
    assert not hasattr(adapter, "aggregator")
    with pytest.raises(ValueError, match="not divisible"):
        adapter(hidden, dit_grid=(21, 60, 40), num_views=7)


if __name__ == "__main__":
    test_clean_latent_rssm_flow(batch_size=1)
    test_spatial_grid_alignment()
    print("test passed")
