"""BWM RGB to grouped V-JEPA observation embeddings."""

import torch
import torch.nn.functional as F
from torch import nn


class VJEPAObservationAdapter(nn.Module):
    """Apply a supplied V-JEPA encoder frame-wise and keep spatial tokens."""

    def __init__(self, encoder: nn.Module, input_size=224, freeze=True):
        super().__init__()
        self.encoder = encoder
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self.freeze = freeze
        # RGB normalization constants expected by V-JEPA calculated on ImageNet dataset
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)) 
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1))
        if freeze:
            self.encoder.requires_grad_(False)

    def _prepare_frames(self, rgb):
        # BWM: [B,V,C,F,H,W] in [-1,1]. V-JEPA AC: one frame repeated for a 2-frame tubelet.
        b, v, c, frames, height, width = rgb.shape
        x = rgb.permute(0, 1, 3, 2, 4, 5).reshape(b * v * frames, c, height, width)
        scale = max(self.input_size[0] / height, self.input_size[1] / width)
        resized = (round(height * scale), round(width * scale))
        x = F.interpolate(x, resized, mode="bilinear", align_corners=False)
        top = (resized[0] - self.input_size[0]) // 2
        left = (resized[1] - self.input_size[1]) // 2
        x = x[:, :, top : top + self.input_size[0], left : left + self.input_size[1]] # centre-crop
        x = ((x + 1.0) / 2.0).unsqueeze(2).repeat(1, 1, 2, 1, 1) # consistent with tubelet_size=2 in jepa encoder
        return (x - self.mean) / self.std

    def forward(self, rgb, group_ids):
        if rgb.ndim != 6:
            raise ValueError(f"Expected RGB [B,V,C,F,H,W], got {tuple(rgb.shape)}")
        b, views, _, frames, _, _ = rgb.shape
        clips = self._prepare_frames(rgb)
        if self.freeze:
            self.encoder.eval()
        tokens = self.encoder(clips)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]

        tokens = tokens.reshape(b, views, frames, tokens.shape[-2], tokens.shape[-1])
        tokens = tokens.permute(0, 2, 1, 3, 4).flatten(2, 3)  # [B,F,V*S,D]
        ids = group_ids[0] if group_ids.ndim == 2 else group_ids
        grouped = tokens.new_zeros((b, int(ids[-1]) + 1, tokens.shape[2], tokens.shape[3]))
        grouped.index_add_(1, ids, tokens)
        counts = torch.bincount(ids).to(tokens).view(1, -1, 1, 1)
        return grouped / counts


class TokenAggregator(nn.Module):
    """Cross-attend over view/spatial tokens and return [B,T,E]."""

    def __init__(self, token_dim, embed_dim, num_heads=4, num_queries=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, kdim=token_dim, vdim=token_dim, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        b, time, spatial, dim = tokens.shape
        tokens = tokens.reshape(b * time, spatial, dim)
        queries = self.queries.expand(b * time, -1, -1)
        pooled, _ = self.attn(queries, tokens, tokens, need_weights=False)
        return self.norm(pooled.mean(1)).reshape(b, time, -1)
