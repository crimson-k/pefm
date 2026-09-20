"""BWM RGB to grouped V-JEPA observation embeddings."""

import torch
from torch import nn

class VJEPAObservationAdapter(nn.Module):
    """Apply a supplied V-JEPA encoder frame-wise and keep spatial tokens."""

    def __init__(self, encoder: nn.Module, freeze=True, normalize=True):
        super().__init__()
        self.encoder = encoder
        self.freeze = freeze
        self.normalize = normalize
        # RGB normalization constants expected by V-JEPA calculated on ImageNet dataset
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)) 
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1))
        if freeze:
            self.encoder.requires_grad_(False)

    def _prepare_frames(self, rgb):
        # BWM: [B,V,C,F,H,W] in [-1,1]. V-JEPA AC: one frame repeated for a 2-frame tubelet.
        b, v, c, frames, height, width = rgb.shape
        x = rgb.permute(0, 1, 3, 2, 4, 5).reshape(b * v * frames, c, height, width)
        x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1) # consistent with tubelet_size=2 in jepa encoder
        if self.normalize:
            x = (x + 1.0) / 2.0
            x = (x - self.mean) / self.std
        return x

    def forward(self, rgb, group_ids=None):
        if rgb.ndim != 6:
            raise ValueError(f"Expected RGB [B,V,C,F,H,W], got {tuple(rgb.shape)}")
        b, views, _, frames, _, _ = rgb.shape
        # (B*V*F, C, T, H, W) -> (B*V*F, C, 2, H, W)
        clips = self._prepare_frames(rgb)
        if self.freeze:
            self.encoder.eval()
        encoder_parameter = next(self.encoder.parameters(), None)
        if encoder_parameter is not None:
            clips = clips.to(device=encoder_parameter.device, dtype=encoder_parameter.dtype)
        tokens = self.encoder(clips)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]

        tokens = tokens.reshape(b, views, frames, tokens.shape[-2], tokens.shape[-1])
        tokens = tokens.permute(0, 2, 1, 3, 4).flatten(2, 3)  # [B,F,V*S,D]
        if group_ids is None:
            return tokens
        ids = group_ids[0] if group_ids.ndim == 2 else group_ids
        ids = ids.to(tokens.device)
        grouped = tokens.new_zeros((b, int(ids[-1]) + 1, tokens.shape[2], tokens.shape[3]))
        grouped.index_add_(1, ids, tokens)
        counts = torch.bincount(ids).to(tokens).view(1, -1, 1, 1)
        return grouped / counts


def row_major_token_grid(tokens, spatial_grid=(30, 40)):
    """Restore ``[B,F,V*H*W,D]`` tokens to ``[B,F,V,H,W,D]``."""
    if tokens.ndim != 4:
        raise ValueError(f"Expected [B,F,V*S,D] tokens, got {tuple(tokens.shape)}")
    height, width = spatial_grid
    spatial_tokens = height * width
    if tokens.shape[2] % spatial_tokens:
        raise ValueError(
            f"Token count {tokens.shape[2]} is not divisible by spatial grid {spatial_grid}"
        )
    views = tokens.shape[2] // spatial_tokens
    return tokens.reshape(tokens.shape[0], tokens.shape[1], views, height, width, tokens.shape[3])


def align_row_major_tokens(tokens, source_grid=(30, 40), target_grid=(15, 20)):
    """Average source-grid blocks while preserving row-major spatial alignment."""
    source_height, source_width = source_grid
    target_height, target_width = target_grid
    if source_height % target_height or source_width % target_width:
        raise ValueError(f"Non-integer grid mapping: {source_grid} -> {target_grid}")
    grid = row_major_token_grid(tokens, source_grid)
    scale_height = source_height // target_height
    scale_width = source_width // target_width
    grid = grid.reshape(
        tokens.shape[0], tokens.shape[1], grid.shape[2], target_height, scale_height,
        target_width, scale_width, tokens.shape[3],
    )
    return grid.mean(dim=(4, 6))


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
