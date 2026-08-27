"""Minimal evaluator call stack from BWM RGB to RSSM prior/posterior."""

import torch
from torch import nn


class DummyVisualPredictor(nn.Module):
    """Causal shape stand-in for VJEPA-AC: previous tokens + transition action."""

    def __init__(self, token_dim, action_dim):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, token_dim)

    def forward(self, tokens, action):
        previous = torch.cat([torch.zeros_like(tokens[:, :1]), tokens[:, :-1]], dim=1)
        predicted = previous + self.action_proj(action).unsqueeze(2)
        return torch.cat([torch.zeros_like(predicted[:, :1]), predicted[:, 1:]], dim=1)


class VJEPARSSMEvaluator(nn.Module):
    def __init__(self, vjepa_adapter, context_predictor, token_aggregator, rssm):
        super().__init__()
        self.vjepa_adapter = vjepa_adapter
        self.context_predictor = context_predictor
        self.token_aggregator = token_aggregator
        self.rssm = rssm

    def forward(self, batch):
        visual_tokens = self.vjepa_adapter(batch["rgb"], batch["group_ids"])
        predicted_visual_tokens = self.context_predictor(visual_tokens, batch["eef"])
        embed = self.token_aggregator(visual_tokens)
        predicted_context = self.token_aggregator(predicted_visual_tokens)
        initial = self.rssm.initial(embed.shape[0])
        post_stoch, deter, post_logits, prior_stoch, prior_logits = self.rssm.observe(
            embed, batch["eef"], predicted_context, initial, batch["reset"]
        )
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
        }
