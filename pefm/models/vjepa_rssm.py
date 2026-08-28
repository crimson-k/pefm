"""Minimal evaluator call stack from BWM RGB to RSSM prior/posterior."""

import torch
from torch import nn

class VJEPARSSMEvaluator(nn.Module):
    def __init__(self, vjepa_adapter, context_predictor, token_aggregator, rssm):
        super().__init__()
        self.vjepa_adapter = vjepa_adapter
        self.context_predictor = context_predictor
        self.token_aggregator = token_aggregator
        self.rssm = rssm
        embed_size = token_aggregator.queries.shape[-1]
        self.posterior_reconstruction_head = nn.Linear(rssm.feat_size, embed_size)
        self.prior_prediction_head = nn.Linear(rssm.feat_size, embed_size)

    def forward(self, batch):
        visual_tokens = self.vjepa_adapter(batch["rgb"], batch["group_ids"])
        actions = batch["eef"][:, 1:] - batch["eef"][:, :-1]
        states = batch["eef"][:, :-1]
        b, t, hw, n = visual_tokens.shape
        source_tokens = visual_tokens[:, :-1].flatten(1, 2)
        predicted = self.context_predictor(source_tokens, actions=actions, states=states)
        predicted = predicted.reshape(b, t-1, hw, n)
        predicted_visual_tokens = torch.cat([torch.zeros_like(predicted[:, :1]), predicted], 1)
        embed = self.token_aggregator(visual_tokens)
        predicted_context = self.token_aggregator(predicted_visual_tokens)
        initial = self.rssm.initial(embed.shape[0])
        actions = torch.cat([torch.zeros_like(actions[:, :1]), actions], 1)
        post_stoch, deter, post_logits, prior_stoch, prior_logits = self.rssm.observe(
            embed, actions, predicted_context, initial, batch["reset"]
        )
        posterior_reconstruction = self.posterior_reconstruction_head(
            self.rssm.get_feat(post_stoch, deter)
        )
        prior_prediction = self.prior_prediction_head(self.rssm.get_feat(prior_stoch, deter))
        return {
            "visual_tokens": visual_tokens,
            "predicted_visual_tokens": predicted_visual_tokens,
            "embed": embed,
            "predicted_context": predicted_context,
            "deter": deter,
            "posterior_stoch": post_stoch,
            "posterior_logits": post_logits,
            "prior_stoch": prior_stoch,
            "prior_logits": prior_logits,
            "posterior_reconstruction": posterior_reconstruction,
            "prior_prediction": prior_prediction,
        }

    @staticmethod
    def _masked_mean(value, valid):
        valid = valid.to(value)
        return (value * valid).sum() / valid.sum().clamp_min(1)

    def compute_loss(self, output, reset, free_nats=1.0, dyn_scale=1.0, rep_scale=0.1):
        """Positive-pair losses; reset positions have no prediction target."""
        valid = ~reset
        target_tokens = output["visual_tokens"].detach()
        target_embed = output["embed"].detach()

        token_error = (output["predicted_visual_tokens"] - target_tokens).abs().mean((-1, -2))
        reconstruction_error = (output["posterior_reconstruction"] - target_embed).square().mean(-1)
        prediction_error = (output["prior_prediction"] - target_embed).square().mean(-1)
        dyn_loss, rep_loss = self.rssm.kl_loss(
            output["posterior_logits"], output["prior_logits"], free_nats
        )

        losses = {
            "token_prediction": self._masked_mean(token_error, valid),
            "posterior_reconstruction": reconstruction_error.mean(),
            "prior_prediction": self._masked_mean(prediction_error, valid),
            "kl_dynamics": self._masked_mean(dyn_loss, valid),
            "kl_representation": self._masked_mean(rep_loss, valid),
        }
        losses["kl"] = dyn_scale * losses["kl_dynamics"] + rep_scale * losses["kl_representation"]
        losses["total"] = (
            losses["token_prediction"]
            + losses["posterior_reconstruction"]
            + losses["prior_prediction"]
            + losses["kl"]
        )
        return losses
