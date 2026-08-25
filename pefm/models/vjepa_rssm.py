"""Minimal evaluator call stack from BWM RGB to RSSM prior/posterior."""

from torch import nn


class VJEPARSSMEvaluator(nn.Module):
    def __init__(self, vjepa_adapter, token_aggregator, rssm):
        super().__init__()
        self.vjepa_adapter = vjepa_adapter
        self.token_aggregator = token_aggregator
        self.rssm = rssm

    def forward(self, batch):
        visual_tokens = self.vjepa_adapter(batch["rgb"], batch["group_ids"])
        real_embed = self.token_aggregator(visual_tokens)
        initial = self.rssm.initial(real_embed.shape[0])
        post_stoch, deter, post_logits = self.rssm.observe(
            real_embed, batch["eef"], initial, batch["reset"]
        )
        prior_stoch, prior_logits = self.rssm.prior(deter)
        return {
            "visual_tokens": visual_tokens,
            "real_embed": real_embed,
            "deter": deter,
            "posterior_stoch": post_stoch,
            "posterior_logits": post_logits,
            "prior_stoch": prior_stoch,
            "prior_logits": prior_logits,
        }
