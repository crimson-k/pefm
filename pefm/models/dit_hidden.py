"""Position-preserving projection from Wan DiT tokens to V-JEPA tokens."""

from torch import nn


def _pool_grid(grid, target_grid):
    """Area-pool an integer DiT grid without interpolating across coordinates."""
    if grid.ndim != 4:
        raise ValueError(f"Expected [N,H,W,D] grid, got {tuple(grid.shape)}")
    height, width = grid.shape[1:3]
    target_height, target_width = target_grid
    if height % target_height or width % target_width:
        raise ValueError(
            f"DiT view grid {(height, width)} cannot map exactly to {target_grid}"
        )
    scale_height, scale_width = height // target_height, width // target_width
    grid = grid.reshape(
        grid.shape[0], target_height, scale_height,
        target_width, scale_width, grid.shape[3],
    )
    return grid.mean(dim=(2, 4))


class SpatiallyAlignedAdapter(nn.Module):
    """Map DiT hidden tokens to the Teacher's ``[V, 15, 20, 1408]`` grid.

    Wan stores multiple camera views by stacking their spatial height.  The
    ``num_views`` argument is therefore part of the layout contract: this
    class refuses ambiguous reshapes and only performs integer area pooling.
    """

    def __init__(
        self,
        dit_dim,
        vjepa_dim=1408,
        target_grid=(15, 20),
        num_views=1,
        hidden_dim=None,
    ):
        super().__init__()
        self.dit_dim = int(dit_dim)
        self.vjepa_dim = int(vjepa_dim)
        self.target_grid = tuple(target_grid)
        self.num_views = int(num_views)
        hidden_dim = int(hidden_dim or vjepa_dim)
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(dit_dim),
            nn.Linear(dit_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, self.vjepa_dim),
            nn.LayerNorm(self.vjepa_dim),
        )
        nn.init.normal_(self.proj[1].weight, std=0.02)
        nn.init.zeros_(self.proj[1].bias)
        nn.init.normal_(self.proj[3].weight, std=0.02)
        nn.init.zeros_(self.proj[3].bias)

    def restore_grid(self, hidden, dit_grid, valid_len=None):
        """Restore Wan's row-major token order to ``[B,T,H,W,D]``."""
        time, height, width = (int(value) for value in dit_grid)
        expected = time * height * width
        if valid_len is not None and int(valid_len) != expected:
            raise ValueError(f"valid_len={valid_len}, but dit_grid requires {expected} tokens")
        if hidden.ndim != 3 or hidden.shape[1] < expected:
            raise ValueError(f"DiT hidden has {tuple(hidden.shape)}; expected at least {expected} tokens")
        if valid_len is not None and hidden.shape[1] != expected:
            raise ValueError(
                f"valid_len was provided for a strict grid input, but hidden has "
                f"{hidden.shape[1]} tokens and dit_grid requires {expected}"
            )
        return hidden[:, :expected].reshape(hidden.shape[0], time, height, width, hidden.shape[-1])

    def project_grid(self, hidden, dit_grid, valid_len=None):
        """Project to the channel-last DiT grid ``[B,T,H,W,vjepa_dim]``."""
        grid = self.restore_grid(hidden, dit_grid, valid_len)
        projection_dtype = self.proj[1].weight.dtype
        projected = self.proj(grid.to(projection_dtype))
        return projected.reshape(
            hidden.shape[0], int(dit_grid[0]), int(dit_grid[1]), int(dit_grid[2]),
            self.vjepa_dim,
        )

    def forward(self, hidden, dit_grid, valid_len=None, num_views=None):
        """Return V-JEPA-style tokens ``[B,T,V,15,20,1408]``."""
        grid = self.project_grid(hidden, dit_grid, valid_len)
        views = self.num_views if num_views is None else int(num_views)
        if views < 1 or grid.shape[2] % views:
            raise ValueError(
                f"DiT height {grid.shape[2]} is not divisible by num_views={views}"
            )
        view_height = grid.shape[2] // views
        grid = grid.reshape(
            grid.shape[0], grid.shape[1], views, view_height, grid.shape[3], grid.shape[4]
        )
        # Pool each view independently; no token is allowed to cross a view boundary.
        aligned = _pool_grid(grid.flatten(0, 2), self.target_grid)
        return aligned.reshape(
            grid.shape[0], grid.shape[1], views,
            self.target_grid[0], self.target_grid[1], self.vjepa_dim,
        )


class DiTHiddenObservation(SpatiallyAlignedAdapter):
    """Backward-compatible name for the position-preserving DiT Adapter."""

    def __init__(self, dit_dim, token_dim=1408, embed_dim=None, hidden_dim=None,
                 target_grid=(15, 20), num_views=1):
        # ``embed_dim`` belonged to the removed private aggregator.  It is
        # accepted for config compatibility but does not create a module.
        del embed_dim
        super().__init__(
            dit_dim=dit_dim,
            vjepa_dim=token_dim,
            target_grid=target_grid,
            num_views=num_views,
            hidden_dim=hidden_dim,
        )
