"""Temporal semantic losses shared by Adapter and BWM training."""

import torch
from torch import nn
from torch.nn import functional as F

from src.r2dreamer import distributions as dists


TIME_GROUPS = 21
HISTORY_GROUPS = 3
VALID_PRIOR_STEPS = 17
TARGET_GRID = (15, 20)


def categorical_kl(target_logits, student_logits):
    """Return KL(target || student), summed over the RSSM stochastic variables."""
    return dists.kl(target_logits.detach().float(), student_logits.float()).sum(-1)


def token_alignment_metrics(predicted, target):
    """Measure correct-position cosine similarity and deterministic shuffles.

    Both tensors use ``[B,T,V,H,W,D]``.  The shuffled values are diagnostics
    for the Adapter gate; they are not part of the training loss.
    """
    if predicted.ndim != 6 or predicted.shape != target.shape:
        raise ValueError(
            f"Expected matching [B,T,V,H,W,D] tensors, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    predicted = F.normalize(predicted.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)
    correct = (predicted * target).sum(-1).mean()
    spatial_target = target.flatten(2, 4).roll(1, dims=2).reshape_as(target)
    time_view_target = target.roll(1, dims=1).roll(1, dims=2)
    return {
        "correct_cosine": correct,
        "shuffled_spatial_cosine": (predicted * spatial_target).sum(-1).mean(),
        "shuffled_time_view_cosine": (predicted * time_view_target).sum(-1).mean(),
    }


class TemporalSemanticDistillation(nn.Module):
    """Map DiT tokens into a frozen Teacher and form the Stage-2 losses."""

    def __init__(self, teacher, adapter, history_groups=3):
        super().__init__()
        if tuple(adapter.target_grid) != TARGET_GRID or int(adapter.num_views) != 1:
            raise ValueError(
                "PEFM Stage 2 currently supports only a single view mapped to 15x20"
            )
        aggregator = getattr(teacher, "token_aggregator", None)
        if aggregator is None:
            raise ValueError("Stage 2 Teacher must expose a shared token_aggregator")
        attention = getattr(aggregator, "attn", None)
        teacher_token_dim = getattr(attention, "kdim", None)
        if teacher_token_dim is None and attention is not None:
            teacher_token_dim = getattr(attention, "embed_dim", None)
        if teacher_token_dim is not None and int(adapter.vjepa_dim) != int(teacher_token_dim):
            raise ValueError(
                f"Adapter vjepa_dim={adapter.vjepa_dim} does not match "
                f"Teacher TokenAggregator token_dim={teacher_token_dim}"
            )
        self.teacher = teacher.requires_grad_(False).eval()
        self.adapter = adapter
        self.history_groups = int(history_groups)
        if self.history_groups != HISTORY_GROUPS:
            raise ValueError(f"Stage 2 requires history_groups={HISTORY_GROUPS}")

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def teacher_targets(self, batch):
        self._validate_teacher_batch(batch)
        output = self.teacher(batch)
        grid = output.get("visual_grid")
        if grid is None or grid.ndim != 6 or tuple(grid.shape[:5]) != (
            batch["rgb"].shape[0], TIME_GROUPS, 1, TARGET_GRID[0], TARGET_GRID[1]
        ):
            raise ValueError(
                "Teacher visual_grid must have shape [B,21,1,15,20,D], got "
                f"{None if grid is None else tuple(grid.shape)}"
            )
        embed = output.get("embed")
        posterior = output.get("posterior_logits")
        if embed is None or embed.ndim != 3 or tuple(embed.shape[:2]) != (
            batch["rgb"].shape[0], TIME_GROUPS
        ):
            raise ValueError(
                f"Teacher embed must have shape [B,21,E], got "
                f"{None if embed is None else tuple(embed.shape)}"
            )
        if posterior is None or posterior.ndim != 4 or tuple(posterior.shape[:2]) != (
            batch["rgb"].shape[0], TIME_GROUPS
        ):
            raise ValueError(
                f"Teacher posterior_logits must have shape [B,21,S,K], got "
                f"{None if posterior is None else tuple(posterior.shape)}"
            )
        return output

    @staticmethod
    def _validate_teacher_batch(batch):
        required = ("rgb", "group_ids", "action_delta", "reset")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing Stage 2 Teacher batch fields: {missing}")
        rgb, groups, action, reset = (batch[key] for key in required)
        if rgb.ndim != 6 or rgb.shape[1:3] != (1, 3) or rgb.shape[3] != 81:
            raise ValueError(
                f"Expected single-view RGB [B,1,3,81,H,W], got {tuple(rgb.shape)}"
            )
        if groups.ndim != 2 or groups.shape[1] != 81:
            raise ValueError(f"Expected group_ids [B,81], got {tuple(groups.shape)}")
        batch_size = rgb.shape[0]
        if groups.shape[0] != batch_size:
            raise ValueError("rgb and group_ids must have the same batch size")
        if action.ndim != 3 or action.shape[1:] != (TIME_GROUPS, 14):
            raise ValueError(f"Expected action_delta [B,21,14], got {tuple(action.shape)}")
        if action.shape[0] != batch_size:
            raise ValueError("rgb and action_delta must have the same batch size")
        if reset.ndim != 2 or reset.shape[1] != TIME_GROUPS:
            raise ValueError(f"Expected reset [B,21], got {tuple(reset.shape)}")
        if reset.shape[0] != batch_size:
            raise ValueError("rgb and reset must have the same batch size")

    def _validate_hidden(self, hidden, dit_grid, valid_len, num_views):
        if hidden.ndim != 3:
            raise ValueError(f"Expected hidden [B,N,D], got {tuple(hidden.shape)}")
        if hidden.shape[-1] != int(self.adapter.dit_dim):
            raise ValueError(
                f"DiT hidden dim {hidden.shape[-1]} does not match Adapter dim "
                f"{self.adapter.dit_dim}"
            )
        if num_views not in (None, 1):
            raise ValueError("PEFM Stage 2 currently supports num_views=1 only")
        if valid_len is None:
            raise ValueError("Stage 2 requires valid_len; it must not infer the DiT grid")
        if len(tuple(dit_grid)) != 3:
            raise ValueError(f"Expected dit_grid=(T,H,W), got {dit_grid!r}")
        time, height, width = (int(value) for value in dit_grid)
        expected = time * height * width
        if time != TIME_GROUPS:
            raise ValueError(f"Stage 2 requires T=21, got dit_grid={dit_grid!r}")
        if int(valid_len) != expected or hidden.shape[1] != expected:
            raise ValueError(
                f"Hidden/grid mismatch: hidden={hidden.shape[1]}, valid_len={valid_len}, "
                f"expected={expected}"
            )
        return (time, height, width)

    def generated_rollout(self, hidden, dit_grid, teacher_output, actions, reset,
                          num_views=None, valid_len=None):
        generated_spatial = self.adapter(
            hidden, dit_grid, valid_len=valid_len, num_views=num_views,
        )
        if generated_spatial.shape != teacher_output["visual_grid"].shape:
            raise ValueError(
                "DiT and V-JEPA spatial grids must align, got "
                f"{tuple(generated_spatial.shape)} and "
                f"{tuple(teacher_output['visual_grid'].shape)}"
            )
        generated_tokens = generated_spatial.flatten(2, 4).contiguous()
        aggregator_dtype = next(self.teacher.token_aggregator.parameters()).dtype
        generated_tokens = generated_tokens.to(aggregator_dtype)
        # The only pooling operation in the DiT branch is the frozen Teacher
        # aggregator; there is deliberately no second learned aggregator.
        generated_embed = self.teacher.token_aggregator(generated_tokens)
        target_embed = teacher_output["embed"].detach()
        if generated_embed.shape != target_embed.shape:
            raise ValueError(
                f"DiT and Teacher embeddings must align, got "
                f"{tuple(generated_embed.shape)} and {tuple(target_embed.shape)}"
            )
        # The common GT history initializes the same state. Future observations
        # come from DiT, so e_t^G first affects the prior at logical step t+1.
        embed = torch.cat(
            [target_embed[:, :self.history_groups], generated_embed[:, self.history_groups:]],
            dim=1,
        )
        output = self.teacher.forward_embeddings(embed, actions, reset)
        output["generated_embed"] = generated_embed
        output["generated_spatial"] = generated_spatial
        return output

    def freeze_adapter(self):
        """Freeze the calibrated Adapter while retaining input autograd."""
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.adapter.eval()
        return self

    def forward(self, hidden, dit_grid, teacher_batch, actions=None, reset=None,
                num_views=None, valid_len=None, phase="bwm", token_scale=0.0,
                embedding_scale=1.0, posterior_scale=1.0):
        """Run the complete RGB-target/DiT-hidden processing path.

        ``teacher_batch`` is the normal evaluator batch containing GT RGB,
        grouped actions and reset flags.  ``hidden`` is the selected DiT block
        output; it remains attached to autograd in both phases.
        """
        if phase not in ("adapter", "bwm", "both"):
            raise ValueError(f"Unknown semantic distillation phase: {phase!r}")
        self._validate_hidden(hidden, dit_grid, valid_len, num_views)
        if actions is None:
            actions = teacher_batch["action_delta"]
        if reset is None:
            reset = teacher_batch["reset"]
        target_batch = teacher_batch
        if actions is not teacher_batch["action_delta"] or reset is not teacher_batch["reset"]:
            target_batch = dict(teacher_batch)
            target_batch["action_delta"] = actions
            target_batch["reset"] = reset
        teacher_output = self.teacher_targets(target_batch)
        generated_output = self.generated_rollout(
            hidden, dit_grid, teacher_output, actions, reset,
            num_views=num_views, valid_len=valid_len,
        )
        result = {"teacher": teacher_output, "generated": generated_output}
        if phase in ("adapter", "both"):
            result["adapter"] = self.adapter_loss(
                generated_output, teacher_output, token_scale,
                embedding_scale, posterior_scale,
            )
        if phase in ("bwm", "both"):
            result["prior"] = self.prior_outputs(generated_output, teacher_output)
        return result

    def adapter_loss(self, generated_output, teacher_output, token_scale=0.0,
                     embedding_scale=1.0, posterior_scale=1.0):
        """Calibrate the Adapter on all 18 future groups (logical steps 3..20)."""
        start = self.history_groups
        target_tokens = teacher_output["visual_grid"][:, start:].detach().float()
        predicted_tokens = generated_output["generated_spatial"][:, start:].float()
        token_metrics = token_alignment_metrics(predicted_tokens, target_tokens)
        # Kept as ``token_mse`` for log compatibility; this is normalized
        # cosine distance, not an unnormalized feature-space MSE.
        token_mse = 2.0 - 2.0 * token_metrics["correct_cosine"]
        embedding_mse = (
            generated_output["generated_embed"][:, start:].float()
            - teacher_output["embed"][:, start:].detach().float()
        ).square().mean()
        posterior_kl = categorical_kl(
            teacher_output["posterior_logits"][:, start:],
            generated_output["posterior_logits"][:, start:],
        ).mean()
        # This is a validation-only next-step prior diagnostic.  The actual
        # prior KL is deliberately left to BWM in Phase C.
        prior_target = teacher_output["posterior_logits"][:, start + 1:].detach().float()
        prior_prediction = generated_output["prior_logits"][:, start + 1:].float()
        next_prior_cosine = F.cosine_similarity(
            prior_prediction.flatten(1), prior_target.flatten(1), dim=-1,
        ).mean().detach()
        total = (
            token_scale * token_mse
            + embedding_scale * embedding_mse
            + posterior_scale * posterior_kl
        )
        return {
            "token_mse": token_mse,
            **token_metrics,
            "embedding_mse": embedding_mse,
            "posterior_kl": posterior_kl,
            "next_prior_cosine": next_prior_cosine,
            "total": total,
        }

    def prior_outputs(self, generated_output, teacher_output):
        """Return p^G_4..20 and q^GT_4..20 for BWM-side KL computation."""
        start = self.history_groups + 1
        generated_prior = generated_output["prior_logits"][:, start:]
        gt_posterior = teacher_output["posterior_logits"][:, start:].detach()
        if generated_prior.shape[1] != VALID_PRIOR_STEPS:
            raise ValueError(
                f"Stage 2 expects 21 groups and 17 valid transitions, got "
                f"{generated_prior.shape[1]}"
            )
        if gt_posterior.shape != generated_prior.shape:
            raise ValueError(
                "GT posterior and Generated prior must have the same [B,17,S,K] shape, "
                f"got {tuple(gt_posterior.shape)} and {tuple(generated_prior.shape)}"
            )
        return {
            "gt_posterior": gt_posterior,
            "generated_prior": generated_prior,
        }
