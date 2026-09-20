"""Build the fixed Stage-1 V-JEPA--RSSM evaluator architecture."""

import torch

from src.vjepa2.src.models.vision_transformer import VisionTransformer

from .rssm_core import RSSM
from .vjepa_adapter import TokenAggregator, VJEPAObservationAdapter
from .vjepa_rssm import VJEPARSSMEvaluator


def build_evaluator(cfg, pretrained_checkpoint=None):
    encoder = VisionTransformer(
        img_size=(256, 256), num_frames=81, embed_dim=1408, depth=40,
        num_heads=22, mlp_ratio=48 / 11, use_silu=False, use_rope=True,
        use_activation_checkpointing=True,
    )
    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
        clean = lambda state: {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
        }
        encoder.load_state_dict(clean(checkpoint["encoder"]), strict=True)
        del checkpoint

    return VJEPARSSMEvaluator(
        VJEPAObservationAdapter(encoder),
        TokenAggregator(token_dim=1408, embed_dim=256, num_heads=4),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    ).to(cfg.device)
