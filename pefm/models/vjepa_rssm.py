"""Minimal evaluator call stack from BWM RGB to RSSM prior/posterior."""

import torch
from torch import nn

from .vjepa_adapter import align_row_major_tokens


class VJEPARSSMEvaluator(nn.Module):
    def __init__(self, vjepa_adapter, token_aggregator, rssm,
                 rgb_grid=(30, 40), aligned_grid=(15, 20)):
        super().__init__()
        self.vjepa_adapter = vjepa_adapter
        self.token_aggregator = token_aggregator
        self.rssm = rssm
        self.rgb_grid = tuple(rgb_grid)
        self.aligned_grid = tuple(aligned_grid)
        embed_size = token_aggregator.queries.shape[-1]
        self.posterior_reconstruction_head = nn.Linear(rssm.feat_size, embed_size)
        self.prior_prediction_head = nn.Linear(rssm.feat_size, embed_size)

    def _run_rssm(self, embed, actions, reset):
        parameter = next(self.rssm.parameters())
        embed = embed.to(device=parameter.device, dtype=parameter.dtype)
        actions = actions.to(device=parameter.device, dtype=parameter.dtype)
        reset = reset.to(device=parameter.device, dtype=torch.bool)
        initial = self.rssm.initial(embed.shape[0])
        post_stoch, deter, post_logits, prior_stoch, prior_logits = self.rssm.observe(
            embed, actions, initial, reset
        )
        posterior_reconstruction = self.posterior_reconstruction_head(
            self.rssm.get_feat(post_stoch, deter)
        )
        prior_prediction = self.prior_prediction_head(self.rssm.get_feat(prior_stoch, deter))
        return {
            "embed": embed,
            "deter": deter,
            "posterior_stoch": post_stoch,
            "posterior_logits": post_logits,
            "prior_stoch": prior_stoch,
            "prior_logits": prior_logits,
            "posterior_reconstruction": posterior_reconstruction,
            "prior_prediction": prior_prediction,
        }

    def forward_embeddings(self, embed, actions, reset):
        """Run RSSM inference from precomputed per-step embeddings.

        This is the entry point for DiT hidden observations. ``embed`` is
        ``[B,T,E]`` and uses the same action/reset alignment as the RGB path.
        """
        if embed.ndim != 3 or actions.ndim != 3 or reset.ndim != 2:
            raise ValueError(
                f"Expected embed [B,T,E], actions [B,T,A], reset [B,T]; "
                f"got {tuple(embed.shape)}, {tuple(actions.shape)}, {tuple(reset.shape)}"
            )
        if embed.shape[:2] != actions.shape[:2] or embed.shape[:2] != reset.shape:
            raise ValueError("embed, actions, and reset must have the same B,T dimensions")
        return self._run_rssm(embed, actions, reset)

    def forward(self, batch):
        visual_tokens = self.vjepa_adapter(batch["rgb"], batch["group_ids"])
        visual_grid = align_row_major_tokens(
            visual_tokens, self.rgb_grid, self.aligned_grid
        )
        visual_tokens = visual_grid.flatten(2, 4).contiguous()
        embed = self.token_aggregator(visual_tokens)
        output = self._run_rssm(embed, batch["action_delta"], batch["reset"])
        output.update({"visual_tokens": visual_tokens, "visual_grid": visual_grid, "embed": embed})
        return output

    @staticmethod
    def _masked_mean(value, valid):
        valid = valid.to(value)
        return (value * valid).sum() / valid.sum().clamp_min(1)

    def compute_loss(self, output, reset, free_nats=1.0, dyn_scale=1.0, rep_scale=0.1):
        """Positive-pair losses; reset positions have no prediction target."""
        valid = ~reset
        target_embed = output["embed"].detach()

        reconstruction_error = (output["posterior_reconstruction"] - target_embed).square().mean(-1)
        prediction_error = (output["prior_prediction"] - target_embed).square().mean(-1)
        dyn_loss, rep_loss = self.rssm.kl_loss(
            output["posterior_logits"], output["prior_logits"], free_nats
        )

        losses = {
            "posterior_reconstruction": reconstruction_error.mean(),
            "prior_prediction": self._masked_mean(prediction_error, valid),
            "kl_dynamics": self._masked_mean(dyn_loss, valid),
            "kl_representation": self._masked_mean(rep_loss, valid),
        }
        losses["kl"] = dyn_scale * losses["kl_dynamics"] + rep_scale * losses["kl_representation"]
        losses["total"] = (
            losses["posterior_reconstruction"]
            + losses["prior_prediction"]
            + losses["kl"]
        )
        return losses
