"""Fixed V-JEPA + Aggregator targets for the simplified Stage 2."""

from torch import nn

from .vjepa_adapter import align_row_major_tokens


class VisualTeacher(nn.Module):
    def __init__(self, vjepa_adapter, token_aggregator,
                 rgb_grid=(15, 20), aligned_grid=(15, 20)):
        super().__init__()
        self.vjepa_adapter = vjepa_adapter
        self.token_aggregator = token_aggregator
        self.rgb_grid = tuple(rgb_grid)
        self.aligned_grid = tuple(aligned_grid)

    def forward(self, batch):
        tokens = self.vjepa_adapter(batch["rgb"], batch["group_ids"])
        grid = align_row_major_tokens(tokens, self.rgb_grid, self.aligned_grid)
        tokens = grid.flatten(2, 4).contiguous()
        return {"embed": self.token_aggregator(tokens),
                "visual_tokens": tokens, "visual_grid": grid}
