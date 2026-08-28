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

    def forward(self, batch):
        visual_tokens = self.vjepa_adapter(batch["rgb"], batch["group_ids"])
        #TODO: replace actions with delta actions
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
