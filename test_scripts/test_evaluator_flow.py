"""Small CPU check for the encoder-to-RSSM evaluator path."""

from pathlib import Path

import torch
import pytest
from omegaconf import OmegaConf
from src.vjepa2.src.models.vision_transformer import vit_tiny

from pefm.models import (
    DiTHiddenObservation,
    RSSM,
    TokenAggregator,
    VJEPARSSMEvaluator,
    VJEPAObservationAdapter,
)
from pefm.models.dit_hidden import SpatiallyAlignedAdapter
from pefm.models.vjepa_adapter import align_row_major_tokens, row_major_token_grid


DIT_HIDDEN_PATH = Path(
    "/data1/fangxuebin/boundless-world-model/outputs/infer/sft_iter2000_pefm_base_6tasks/adjust_bottle/episode40_dit_block20_hidden.pt"
)


def make_rssm_config():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    cfg = OmegaConf.create({"model": OmegaConf.merge(base, size)})
    cfg.model.rssm.device = "cpu"
    cfg.model.rssm.initial = "zeros"
    return cfg


def make_test_vjepa():
    """Build a small real V-JEPA encoder for a fast end-to-end CPU test."""
    return vit_tiny(
        img_size=(256, 256),
        num_frames=2,
        use_rope=True,
    )


def print_prior_results(name, output):
    """Print the categorical prior and posterior state for every RSSM step."""
    prior = output["prior_logits"].detach()
    posterior = output["posterior_logits"].detach()
    for step in range(prior.shape[1]):
        prior_state = prior[0, step].argmax(-1).tolist()
        posterior_state = posterior[0, step].argmax(-1).tolist()
        prediction_norm = output["prior_prediction"][0, step].detach().norm().item()
        print(
            f"[{name}] step={step:02d} "
            f"prior_mean={prior[0, step].mean().item():+.6f} "
            f"prior_prediction_norm={prediction_norm:.6f} "
            f"prior_argmax={prior_state} "
            f"posterior_argmax={posterior_state}",
            flush=True,
        )


def assert_rssm_output(output, steps):
    assert output["embed"].shape[:2] == (1, steps)
    assert output["prior_logits"].shape == (1, steps, 32, 16)
    assert output["posterior_logits"].shape == (1, steps, 32, 16)
    assert output["prior_stoch"].shape == (1, steps, 32, 16)
    assert output["posterior_stoch"].shape == (1, steps, 32, 16)
    torch.testing.assert_close(
        output["prior_stoch"].sum(-1), torch.ones(1, steps, 32), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        output["posterior_stoch"].sum(-1), torch.ones(1, steps, 32), atol=1e-5, rtol=1e-5
    )
    assert output["prior_prediction"].shape == (1, steps, 256)
    assert output["posterior_reconstruction"].shape == (1, steps, 256)
    for value in output.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), f"non-finite output: {value.shape}"


def test_bwm_vjepa_rssm_flow():
    torch.manual_seed(0)
    cfg = make_rssm_config()

    evaluator = VJEPARSSMEvaluator(
        VJEPAObservationAdapter(make_test_vjepa()),
        TokenAggregator(token_dim=192, embed_dim=256, num_heads=4),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    )
    # BWM uses 81 frames: 3 history groups (the first 9 frames) plus 18 future groups.
    group_ids = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.arange(1, 21, dtype=torch.long).repeat_interleave(4),
    ]).view(1, -1)
    batch = {
        "rgb": torch.rand(1, 1, 3, 81, 480, 640) * 2 - 1,
        "group_ids": group_ids,
        "action_delta": torch.randn(1, 21, 14),
        "reset": torch.tensor([[True] + [False] * 20]),
    }
    raw_tokens = evaluator.vjepa_adapter(batch["rgb"])
    assert raw_tokens.shape == (1, 81, 1200, 192)
    grouped_tokens = raw_tokens.new_zeros((1, 21, 1200, 192))
    grouped_tokens.index_add_(1, group_ids[0], raw_tokens)
    grouped_tokens /= torch.bincount(group_ids[0]).to(raw_tokens).view(1, -1, 1, 1)
    assert grouped_tokens.shape == (1, 21, 1200, 192)
    aligned_tokens = align_row_major_tokens(grouped_tokens)
    assert aligned_tokens.shape == (1, 21, 1, 15, 20, 192)
    coordinates = torch.arange(1200, dtype=torch.float32).reshape(1, 1, 1200, 1)
    coordinate_grid = row_major_token_grid(coordinates)
    coordinate_aligned = align_row_major_tokens(coordinates)
    assert coordinate_grid[0, 0, 0, 1, 0, 0] == 40
    assert coordinate_aligned[0, 0, 0, 0, 0, 0] == 20.5

    output = evaluator(batch)
    assert output["visual_grid"].shape == (1, 21, 1, 15, 20, 192)
    assert output["visual_tokens"].shape == (1, 21, 300, 192)
    assert torch.allclose(output["visual_grid"], aligned_tokens)
    assert_rssm_output(output, steps=21)
    print_prior_results("rgb", output)
    losses = evaluator.compute_loss(output, batch["reset"], cfg.model.kl_free,
                                    cfg.model.loss_scales.dyn, cfg.model.loss_scales.rep)
    assert set(losses) == {"posterior_reconstruction", "prior_prediction",
                           "kl_dynamics", "kl_representation", "kl", "total"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total"].backward()
    assert next(evaluator.rssm._img_net.parameters()).grad is not None
    assert next(evaluator.rssm._obs_net.parameters()).grad is not None
    assert evaluator.token_aggregator.queries.grad is not None
    assert all(parameter.grad is None for parameter in evaluator.vjepa_adapter.encoder.parameters())


def test_dit_hidden_rssm_flow():
    torch.manual_seed(0)
    assert DIT_HIDDEN_PATH.is_file(), f"Missing DiT hidden fixture: {DIT_HIDDEN_PATH}"
    hidden = torch.load(DIT_HIDDEN_PATH, map_location="cpu", weights_only=True)
    assert hidden.shape == (1, 6300, 3072)

    observation = DiTHiddenObservation(dit_dim=3072, token_dim=192, embed_dim=256)
    grid = observation.project_grid(hidden, dit_grid=(21, 15, 20), valid_len=6300)
    spatial = observation(hidden, dit_grid=(21, 15, 20), valid_len=6300)
    assert grid.shape == (1, 21, 15, 20, 192)
    assert spatial.shape == (1, 21, 1, 15, 20, 192)
    dit_tokens = spatial.flatten(2, 4).contiguous()
    token_aggregator = TokenAggregator(token_dim=192, embed_dim=256, num_heads=4)
    embed = token_aggregator(dit_tokens)

    cfg = make_rssm_config()
    rssm = RSSM(cfg.model.rssm, embed_size=256, act_dim=14)
    action = torch.randn(1, 21, 14)
    reset = torch.zeros(1, 21, dtype=torch.bool)
    reset[:, 0] = True
    evaluator = VJEPARSSMEvaluator(
        vjepa_adapter=torch.nn.Identity(),
        token_aggregator=token_aggregator,
        rssm=rssm,
    )
    output = evaluator.forward_embeddings(embed, action, reset)
    assert_rssm_output(output, steps=21)
    print_prior_results("dit", output)
    losses = evaluator.compute_loss(output, reset, free_nats=0.0)
    assert all(torch.isfinite(loss) for loss in losses.values())

    # The DiT observation projection participates in posterior inference.
    output["posterior_logits"].square().mean().backward()
    assert observation.proj[1].weight.grad is not None
    assert token_aggregator.queries.grad is not None


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
    test_bwm_vjepa_rssm_flow()
    test_dit_hidden_rssm_flow()
    print("test passed")
