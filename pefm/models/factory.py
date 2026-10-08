"""Build the visual target branch; Stage 2 has no stochastic state model."""

import torch

from .vjepa_adapter import TokenAggregator, VJEPAObservationAdapter
from .visual_teacher import VisualTeacher


def build_visual_teacher(cfg, pretrained_checkpoint=None):
    # Cached Phase A training only needs the GRU; import the encoder on demand.
    from src.vjepa2.src.models.vision_transformer import VisionTransformer

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

    rgb_grid = tuple(int(value) for value in cfg.get("rgb_grid", [15, 20]))
    aligned_grid = tuple(int(value) for value in cfg.get("aligned_grid", rgb_grid))
    aggregator = cfg.get("token_aggregator", {})
    return VisualTeacher(
        VJEPAObservationAdapter(encoder),
        TokenAggregator(token_dim=1408, embed_dim=int(aggregator.get("embed_dim", 256)),
                        num_heads=int(aggregator.get("num_heads", 4)),
                        num_queries=int(aggregator.get("num_queries", 4))),
        rgb_grid=rgb_grid,
        aligned_grid=aligned_grid,
    ).to(cfg.device)


# Compatibility for callers loading the visual part of an old Teacher bundle.
build_evaluator = build_visual_teacher
