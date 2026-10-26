"""Check RGB/VAE training and reuse of the clean-latent evaluator."""

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch import nn

from pefm.export import export_frozen_evaluator, load_frozen_evaluator, teacher_contract
from pefm.models import VJEPASemanticTarget, build_evaluator
from pefm.training.train import encode_rgb_latents, mask_teacher_losses, run_epoch, save_state


class FakeWanVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)

    def encode(self, videos, device):
        assert not torch.is_grad_enabled()
        assert videos.device == device
        return (videos[:, :1, ::4, ::16, ::16] * self.weight).expand(-1, 48, -1, -1, -1)


class TinySemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 1408)
        self.dropout = nn.Dropout(0.75)

    def forward(self, clips):
        tokens = torch.nn.functional.adaptive_avg_pool2d(clips.mean(2), (2, 2))
        return self.dropout(self.proj(tokens.flatten(2).transpose(1, 2)))


def training_config():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    for unused in ("encoder", "decoder"):
        del base[unused]
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    cfg = OmegaConf.create({
        "model": OmegaConf.merge(base, size), "device": "cpu", "seed": 0,
        "observation": "clean_latents", "height": 192, "width": 256,
        "resize_mode": "crop", "max_grad_norm": 0.5, "max_batches": 1,
        "reward_label_reduction": "masked_or",
    })
    cfg.model.rssm.device = "cpu"
    cfg.model.rssm.initial = "zeros"
    return cfg


def test_vae_view_layout_and_no_grad():
    rgb = torch.zeros(2, 2, 3, 5, 32, 32, requires_grad=True)
    with torch.no_grad():
        rgb[0, 0].fill_(0.1)
        rgb[0, 1].fill_(0.2)
        rgb[1, 0].fill_(0.3)
        rgb[1, 1].fill_(0.4)
    latents = encode_rgb_latents(FakeWanVAE(), rgb)
    assert latents.shape == (2, 48, 2, 4, 2)
    assert not latents.requires_grad
    for batch_index in range(2):
        for view in range(2):
            torch.testing.assert_close(
                latents[batch_index, :, :, view * 2:(view + 1) * 2],
                torch.full((48, 2, 2, 2), 0.1 * (batch_index * 2 + view + 1)),
            )


@pytest.mark.parametrize("dual_arm", [False, True])
@pytest.mark.parametrize("multi_object", [False, True])
def test_rgb_vae_training_and_frozen_export(tmp_path, dual_arm, multi_object):
    cfg = training_config()
    cfg.model.loss_scales.update(relation=2.0, event=3.0, motion=5.0, interaction=0.2)
    model = build_evaluator(cfg)
    vae = FakeWanVAE()
    semantic_target = VJEPASemanticTarget(TinySemanticEncoder())
    teacher_before = {name: value.clone() for name, value in semantic_target.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    group_ids = torch.cat((torch.zeros(1, dtype=torch.long),
                           torch.arange(1, 21).repeat_interleave(4))).view(1, -1)
    batch = {
        "rgb": torch.rand(1, 1, 3, 81, 192, 256) * 2 - 1,
        "group_ids": group_ids,
        "action_delta": torch.randn(1, 21, 14),
        "reset": torch.tensor([[True] + [False] * 20]),
        "interaction_target": torch.cat((
            torch.nn.functional.one_hot(torch.arange(21).remainder(3), 3).float(),
            torch.nn.functional.one_hot(torch.arange(21).remainder(8), 8).float(),
            torch.randint(0, 2, (21, 5)).float(),
        ), dim=-1).unsqueeze(0),
        "interaction_mask": torch.ones(1, 21, 16, dtype=torch.bool),
        "event_is_dual_arm": torch.full((1, 21), dual_arm, dtype=torch.bool),
    }
    if multi_object:
        target = batch["interaction_target"]
        target[..., :3] = torch.maximum(target[..., :3], target[..., :3].roll(1, dims=-1))
        target[..., 3:11] = torch.maximum(target[..., 3:11], target[..., 3:11].roll(1, dims=-1))
        batch["interaction_is_multi_object"] = torch.ones(1, 21, dtype=torch.bool)
    elif dual_arm:
        events = batch["interaction_target"][..., 3:11]
        events.add_(events.roll(1, dims=-1))
        batch["interaction_target"] = batch["interaction_target"].unsqueeze(2).repeat(1, 1, 2, 1)
        batch["interaction_mask"] = batch["interaction_mask"].unsqueeze(2).repeat(1, 1, 2, 1)
    observed = []
    hook = model.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0]["clean_latents"].detach())
    )
    queries_before = model.token_aggregator.queries.detach().clone()
    metrics = run_epoch(
        model, [batch], cfg, optimizer, 0, "train", Accelerator(cpu=True),
        vae=vae, semantic_target=semantic_target,
    )
    hook.remove()
    assert observed[0].shape == (1, 48, 21, 12, 16)
    assert not torch.equal(queries_before, model.token_aggregator.queries)
    assert vae.weight.grad is None
    assert all(name in metrics for name in ("relation", "event", "motion", "interaction"))
    torch.testing.assert_close(
        torch.tensor(metrics["interaction"]),
        torch.tensor(2 * metrics["relation"] + 3 * metrics["event"] + 5 * metrics["motion"]),
    )
    torch.testing.assert_close(
        torch.tensor(metrics["total"]),
        torch.tensor(metrics["posterior_reconstruction"] + metrics["prior_prediction"]
                     + metrics["kl"] + 0.2 * metrics["interaction"]),
    )
    for head in (model.relation_head, model.event_head, model.motion_head):
        assert head[-1].weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in semantic_target.parameters())
    for name, value in semantic_target.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name])
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

    checkpoint, bundle = tmp_path / "latest.pt", tmp_path / "frozen_evaluator.pt"
    save_state(checkpoint, model, optimizer, cfg, epoch=1)
    export_frozen_evaluator(checkpoint, bundle)
    frozen = load_frozen_evaluator(bundle, device="cpu", dtype=torch.float32)
    assert frozen.token_aggregator.attn.kdim == 48
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    contract = teacher_contract(cfg)
    assert contract["embed_size"] == 512
    assert contract["token_aggregator"]["embed_dim"] == 512
    assert contract["observation"] == "wan_vae_clean_latents"
    assert contract["vae"]["rgb_size"] == [192, 256]
    assert contract["semantic_target"]["dim"] == 1408
    assert contract["interaction_heads"]["shared_prior_posterior"]
    assert contract["interaction_heads"]["event"] == 8
    assert contract["interaction_heads"]["motion"] == 5
    assert contract["interaction_heads"]["object_label_reduction"] == "masked_or"
    assert contract["interaction_heads"]["object_loss_reduction"] == "none"
    assert contract["interaction_heads"]["relation_loss"] == "bce_with_logits"
    torch.testing.assert_close(frozen.token_aggregator.queries, model.token_aggregator.queries)

    latents = observed[0].clone().requires_grad_()
    output = frozen.forward_latents(latents, batch["action_delta"], batch["reset"])
    assert output["embed"].shape == (1, 21, 512)
    assert output["posterior_reconstruction"].shape == (1, 21, 1408)
    for branch in ("posterior", "prior"):
        assert output[f"{branch}_relation_logits"].shape == (1, 21, 3)
        assert output[f"{branch}_event_logits"].shape == (1, 21, 8)
        assert output[f"{branch}_motion_logits"].shape == (1, 21, 5)
    output["posterior_logits"].square().mean().backward()
    assert torch.isfinite(latents.grad).all() and latents.grad.norm() > 0


def test_semantic_target_stays_frozen_and_deterministic():
    target = VJEPASemanticTarget(TinySemanticEncoder())
    target.train(True)
    assert all(not module.training for module in target.modules())
    assert all(not parameter.requires_grad for parameter in target.parameters())
    rgb = torch.rand(1, 1, 3, 5, 32, 32, requires_grad=True)
    groups = torch.tensor([[0, 1, 1, 1, 1]])
    features = target(rgb, groups)
    assert features.shape == (1, 2, 1408)
    assert not features.requires_grad
    torch.testing.assert_close(features, target(rgb, groups), rtol=0, atol=0)


def test_semantic_loss_masks_history_and_detaches_gt():
    prediction = torch.zeros(1, 4, 1408, requires_grad=True)
    gt = torch.full((1, 4, 1408), 2.0)
    gt[:, :2] = 100
    gt.requires_grad_()
    output = {"embed": torch.zeros(1, 4, 512), "posterior_reconstruction": prediction}
    original_reconstruction = (prediction - gt.detach()).square().mean()
    losses = {
        "posterior_reconstruction": original_reconstruction,
        "prior_prediction": torch.tensor(3.0), "kl": torch.tensor(1.0),
        "interaction": torch.tensor(5.0), "total": original_reconstruction + 3 + 1 + 5,
    }
    losses = mask_teacher_losses(
        output, losses, torch.tensor([[True, True, False, False]]), target_embed=gt,
    )
    torch.testing.assert_close(losses["posterior_reconstruction"], torch.tensor(4.0))
    torch.testing.assert_close(losses["total"], torch.tensor(13.0))
    losses["total"].backward()
    assert gt.grad is None
    assert prediction.grad[:, :2].abs().sum() == 0
    assert prediction.grad[:, 2:].abs().sum() > 0


def test_real_tiny_vjepa_target_to_latent_rssm():
    from src.vjepa2.src.models.vision_transformer import vit_tiny

    target = VJEPASemanticTarget(vit_tiny(img_size=(32, 32), num_frames=2, use_rope=True))
    rgb = torch.rand(1, 1, 3, 5, 32, 32) * 2 - 1
    gt = target(rgb, torch.tensor([[0, 1, 1, 1, 1]]))
    assert gt.shape == (1, 2, 192)
    cfg = training_config()
    cfg.semantic_dim = 192
    model = build_evaluator(cfg)
    reset = torch.tensor([[True, False]])
    output = model.forward_latents(
        encode_rgb_latents(FakeWanVAE(), rgb), torch.randn(1, 2, 14), reset,
    )
    losses = model.compute_loss(output, reset, free_nats=0, target_embed=gt)
    losses["total"].backward()
    assert model.token_aggregator.queries.grad.norm() > 0
    assert model.posterior_reconstruction_head.weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
