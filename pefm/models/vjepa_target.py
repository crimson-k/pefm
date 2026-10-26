"""Frozen GT RGB -> V-JEPA -> parameter-free spatial mean targets."""

import torch
from torch import nn

from src.vjepa2.src.models.vision_transformer import VisionTransformer

from .vjepa_adapter import VJEPAObservationAdapter


def build_vjepa_encoder(pretrained_checkpoint=None):
    encoder = VisionTransformer(
        img_size=(256, 256), num_frames=81, embed_dim=1408, depth=40,
        num_heads=22, mlp_ratio=48 / 11, use_silu=False, use_rope=True,
        use_activation_checkpointing=True,
    )
    if pretrained_checkpoint:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False, mmap=True)
        state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint["encoder"].items()
        }
        encoder.load_state_dict(state, strict=True)
    return encoder


class VJEPASemanticTarget(nn.Module):
    """Keep the encoder frozen and return grouped [B,T,1408] GT targets."""

    def __init__(self, encoder):
        super().__init__()
        self.adapter = VJEPAObservationAdapter(encoder, freeze=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb, group_ids):
        return self.adapter(rgb, group_ids).mean(dim=2)


def build_vjepa_target(checkpoint, device, dtype=torch.bfloat16):
    return VJEPASemanticTarget(build_vjepa_encoder(checkpoint)).to(device=device, dtype=dtype)
