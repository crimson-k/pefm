"""Small CPU checks for the PEFM-side Stage 2 contracts."""

import copy
import math

import torch
import pytest
from omegaconf import OmegaConf
from torch import nn

from pefm.export import (
    check_teacher_contract,
    export_stage2_adapter,
    load_stage2_adapter,
    teacher_contract,
)
from pefm.models import (
    SpatiallyAlignedAdapter,
    TemporalSemanticDistillation,
    TokenAggregator,
    categorical_kl,
    token_alignment_metrics,
)


class FakeTeacher(nn.Module):
    """Tiny frozen Teacher with the same spatial/pooled/logit interfaces."""

    def __init__(self):
        super().__init__()
        self.token_aggregator = TokenAggregator(token_dim=4, embed_dim=3, num_heads=1)
        self.state_head = nn.Linear(3, 6)
        self.action_head = nn.Linear(14, 6)

    def _posterior(self, embed):
        return self.state_head(embed).reshape(embed.shape[0], embed.shape[1], 2, 3)

    def forward(self, batch):
        visual_grid = batch["visual_grid"]
        embed = self.token_aggregator(visual_grid.flatten(2, 4))
        return {
            "visual_grid": visual_grid,
            "embed": embed,
            "posterior_logits": self._posterior(embed),
        }

    def forward_embeddings(self, embed, actions, reset):
        posterior = self._posterior(embed)
        # p_t depends on e_{t-1} and the preceding action, mirroring the
        # one-step RSSM offset and making action-shuffle checks meaningful.
        action_logits = self.action_head(actions).reshape(actions.shape[0], actions.shape[1], 2, 3)
        prior = torch.cat((posterior[:, :1], posterior[:, :-1] + action_logits[:, :-1]), dim=1)
        return {"embed": embed, "posterior_logits": posterior, "prior_logits": prior}


def teacher_batch():
    return {
        "rgb": torch.zeros(1, 1, 3, 81, 2, 2),
        "group_ids": torch.cat((torch.zeros(1, dtype=torch.long),
                                 torch.arange(1, 21).repeat_interleave(4))).view(1, -1),
        "action_delta": torch.randn(1, 21, 14),
        "reset": torch.tensor([[True] + [False] * 20]),
        "visual_grid": torch.randn(1, 21, 1, 15, 20, 4),
    }


def test_coordinate_round_trip_and_strict_length():
    adapter = SpatiallyAlignedAdapter(4, vjepa_dim=4, hidden_dim=4)
    coordinates = torch.arange(21 * 15 * 20 * 4, dtype=torch.float32).reshape(1, -1, 4)
    restored = adapter.restore_grid(coordinates, (21, 15, 20), valid_len=6300)
    torch.testing.assert_close(
        restored.reshape_as(coordinates), coordinates,
    )
    assert restored[0, 3, 7, 11, 2] == coordinates[0, (3 * 15 + 7) * 20 + 11, 2]
    with pytest.raises(ValueError, match="strict grid"):
        adapter.restore_grid(torch.cat((coordinates, coordinates[:, :1]), dim=1),
                             (21, 15, 20), valid_len=6300)


def test_phase_b_and_c_gradient_boundaries():
    torch.manual_seed(0)
    teacher = FakeTeacher()
    adapter = SpatiallyAlignedAdapter(4, vjepa_dim=4, hidden_dim=4)
    distiller = TemporalSemanticDistillation(teacher, adapter)
    batch = teacher_batch()
    hidden = torch.randn(1, 6300, 4, requires_grad=True)

    adapter_result = distiller(
        hidden.detach(), (21, 15, 20), valid_len=6300,
        teacher_batch=batch, phase="adapter",
    )
    adapter_result["adapter"]["total"].backward()
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert torch.isfinite(adapter_result["adapter"]["total"])
    teacher_state = {
        name: value.detach().clone() for name, value in teacher.state_dict().items()
    }
    adapter_state = {
        name: value.detach().clone() for name, value in adapter.state_dict().items()
    }

    for parameter in adapter.parameters():
        parameter.grad = None
    hidden = hidden.detach().requires_grad_(True)
    distiller.freeze_adapter()
    prior_result = distiller(
        hidden, (21, 15, 20), valid_len=6300,
        teacher_batch=batch, phase="bwm",
    )
    prior_outputs = prior_result["prior"]
    assert prior_outputs["gt_posterior"].shape == (1, 17, 2, 3)
    assert prior_outputs["generated_prior"].shape == (1, 17, 2, 3)
    assert not prior_outputs["gt_posterior"].requires_grad
    assert prior_outputs["generated_prior"].requires_grad
    assert "prior_kl" not in prior_outputs
    # BWM owns this KL reduction; the test only simulates its consumer.
    categorical_kl(
        prior_outputs["gt_posterior"], prior_outputs["generated_prior"],
    ).mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    step_grad = hidden.grad.reshape(1, 21, -1, 4).norm(dim=(-1, -2))[0]
    assert step_grad[19] > 0
    torch.testing.assert_close(step_grad[20], torch.zeros_like(step_grad[20]))
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_state[name])
    for name, value in adapter.state_dict().items():
        torch.testing.assert_close(value, adapter_state[name])


def test_token_diagnostics_and_adapter_bundle(tmp_path):
    target = torch.randn(1, 4, 1, 2, 2, 3)
    metrics = token_alignment_metrics(target, target)
    assert metrics["correct_cosine"] > metrics["shuffled_spatial_cosine"]
    assert metrics["correct_cosine"] > metrics["shuffled_time_view_cosine"]
    paired = torch.cat((target, target + 0.1), dim=0)
    mismatched = paired.flip(0)
    assert token_alignment_metrics(paired, paired)["correct_cosine"] > (
        token_alignment_metrics(paired, mismatched)["correct_cosine"]
    )

    adapter = SpatiallyAlignedAdapter(4, vjepa_dim=4, hidden_dim=4)
    path = export_stage2_adapter(
        tmp_path / "adapter.pt", adapter, selected_block=20,
        teacher_contract={"state_space": "shared_rssm_v1", "time_groups": 21},
    )
    loaded = load_stage2_adapter(
        path, device="cpu", dit_dim=4, selected_block=20,
        teacher_contract={"state_space": "shared_rssm_v1", "time_groups": 21},
    )
    assert loaded.training
    hidden = torch.randn(1, 6300, 4)
    torch.testing.assert_close(
        adapter(hidden, (21, 15, 20), valid_len=6300),
        loaded(hidden, (21, 15, 20), valid_len=6300),
    )
    loaded = load_stage2_adapter(
        path, device="cpu", dit_dim=4, selected_block=20,
        teacher_contract={"state_space": "shared_rssm_v1", "time_groups": 21},
        freeze=True,
    )
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for name, value in adapter.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    with pytest.raises(ValueError, match="DiT hidden dim mismatch"):
        load_stage2_adapter(
            path, device="cpu", dit_dim=8, selected_block=20,
            teacher_contract={"state_space": "shared_rssm_v1", "time_groups": 21},
        )


def test_adapter_pairing_acceptance():
    """A content-matched hidden/GT pair must beat a cross-sample pairing."""
    class IdentityAdapter(nn.Module):
        dit_dim = 4
        vjepa_dim = 4
        target_grid = (15, 20)
        num_views = 1

        def forward(self, hidden, dit_grid, valid_len=None, num_views=None):
            del valid_len, num_views
            time, height, width = (int(value) for value in dit_grid)
            return hidden.reshape(hidden.shape[0], time, 1, height, width, -1)

    torch.manual_seed(7)
    teacher = FakeTeacher().eval()
    distiller = TemporalSemanticDistillation(teacher, IdentityAdapter())
    batch = teacher_batch()
    other = teacher_batch()
    paired_batch = {
        key: torch.cat((batch[key], other[key]), dim=0)
        for key in ("rgb", "group_ids", "action_delta", "reset", "visual_grid")
    }
    hidden = paired_batch["visual_grid"].flatten(1, 4)
    paired = distiller(
        hidden, (21, 15, 20), valid_len=6300,
        teacher_batch=paired_batch, phase="adapter",
    )["adapter"]
    mismatched_batch = {
        key: value.flip(0) for key, value in paired_batch.items()
    }
    mismatched = distiller(
        hidden, (21, 15, 20), valid_len=6300,
        teacher_batch=mismatched_batch, phase="adapter",
    )["adapter"]
    assert paired["embedding_mse"] < mismatched["embedding_mse"]
    assert paired["posterior_kl"] < mismatched["posterior_kl"]


def test_teacher_state_and_action_acceptance():
    torch.manual_seed(1)
    teacher = FakeTeacher().eval()
    batch = teacher_batch()
    output = teacher(batch)
    probabilities = output["posterior_logits"].softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1).mean()
    temporal_variance = probabilities.var(dim=1).mean()
    assert temporal_variance > 1e-8
    assert entropy < math.log(3.0)

    normal = teacher.forward_embeddings(
        output["embed"], batch["action_delta"], batch["reset"],
    )
    shuffled_actions = batch["action_delta"][:, torch.arange(20, -1, -1)]
    shuffled = teacher.forward_embeddings(
        output["embed"], shuffled_actions, batch["reset"],
    )
    assert (
        normal["prior_logits"][:, 4:] - shuffled["prior_logits"][:, 4:]
    ).abs().mean() > 1e-6


def teacher_config():
    return OmegaConf.create({
        "model": {
            "rssm": {
                "stoch": 32, "deter": 2048, "hidden": 256, "discrete": 16,
                "img_layers": 2, "obs_layers": 1, "dyn_layers": 1,
                "blocks": 8, "act": "SiLU", "unimix_ratio": 0.01,
                "initial": "zeros",
            },
        },
    })


def test_teacher_contract_rejects_old_or_incompatible_checkpoint():
    cfg = teacher_config()
    contract = teacher_contract(cfg)
    check_teacher_contract({"semantic_teacher": contract}, cfg)
    with pytest.raises(ValueError, match="missing the semantic_teacher"):
        check_teacher_contract({}, cfg)
    incompatible = copy.deepcopy(contract)
    incompatible["rssm"]["deter"] += 1
    with pytest.raises(ValueError, match="Teacher contract mismatch"):
        check_teacher_contract({"semantic_teacher": incompatible}, cfg)
