"""Build the clean-latent -> TokenAggregator -> RSSM evaluator."""

from .rssm_core import RSSM
from .vjepa_adapter import TokenAggregator
from .vjepa_rssm import Evaluator


def build_evaluator(cfg):
    return Evaluator(
        token_aggregator=TokenAggregator(token_dim=48, embed_dim=512, num_heads=4),
        rssm=RSSM(cfg.model.rssm, embed_size=512, act_dim=14),
        target_dim=int(cfg.get("semantic_dim", 1408)),
    ).to(cfg.device)
