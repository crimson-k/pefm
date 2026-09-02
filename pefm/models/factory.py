"""Build the fixed Stage-1 V-JEPA--RSSM evaluator architecture."""

import torch

from src.vjepa2.src.models.ac_predictor import VisionTransformerPredictorAC
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
    predictor = VisionTransformerPredictorAC(
        img_size=256, num_frames=40, patch_size=16, embed_dim=1408,
        action_embed_dim=14,
    )

    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
        clean = lambda state: {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()
        }
        encoder.load_state_dict(clean(checkpoint["encoder"]), strict=True)
        predictor_state = clean(checkpoint["predictor"])
        current = predictor.state_dict()
        compatible = {
            key: value for key, value in predictor_state.items()
            if key in current and value.shape == current[key].shape
        }
        loaded = predictor.load_state_dict(compatible, strict=False)
        assert set(loaded.missing_keys) == {
            "action_encoder.weight", "state_encoder.weight", "extrinsics_encoder.weight",
        }
        del checkpoint, predictor_state

    return VJEPARSSMEvaluator(
        VJEPAObservationAdapter(encoder, input_size=256), predictor,
        TokenAggregator(token_dim=1408, embed_dim=256, num_heads=4),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    ).to(cfg.device)
