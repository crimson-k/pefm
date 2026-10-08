"""固定 Aggregator embedding 上的动作预测与 DiT 特征对齐。"""

import torch
from torch import nn

from .vjepa_adapter import align_row_major_tokens

TIME_GROUPS = 21
HISTORY_GROUPS = 3
VALID_PREDICTION_STEPS = 17
TARGET_GRID = (15, 20)


class ActionPredictor(nn.Module):
    """每个起点独立初始化；多步预测只递推隐状态和有序动作。"""

    def __init__(self, embed_dim, hidden_dim=256, use_actions=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_actions = use_actions
        self.initial = nn.Linear(embed_dim, hidden_dim)
        self.action = nn.Sequential(nn.Linear(56, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, embed_dim)
        self.register_buffer("mean", torch.zeros(embed_dim))
        self.register_buffer("std", torch.ones(embed_dim))

    def distance(self, prediction, target):
        return ((prediction.float() - target.detach().float()) / self.std.float()).square().mean(-1)

    def forward(self, embedding, action_segments):
        # action_segments: [..., horizon, 4, 14]，四帧按原顺序送入 MLP。
        dtype = self.initial.weight.dtype
        state = self.initial(((embedding - self.mean) / self.std).to(dtype)).flatten(0, -2)
        actions = action_segments.to(dtype).reshape(state.shape[0], -1, 56)
        if not self.use_actions:
            actions = torch.zeros_like(actions)
        outputs = []
        for action in actions.unbind(1):
            state = self.gru(self.action(action), state)
            outputs.append(self.output(state))
        result = torch.stack(outputs, dim=1).reshape(*embedding.shape[:-1], -1, self.embed_dim)
        return result * self.std + self.mean

    def pairs(self, embedding, actions, horizon=1):
        if not 1 <= horizon <= 17:
            raise ValueError("Prediction horizon must be between 1 and 17")
        stop = TIME_GROUPS - horizon
        segments = torch.stack([actions[:, t + 1:t + 1 + horizon]
                                for t in range(HISTORY_GROUPS, stop)], dim=1)
        return self(embedding[:, HISTORY_GROUPS:stop], segments)


class TemporalSemanticDistillation(nn.Module):
    def __init__(self, teacher, adapter, predictor, sigma_max=1.0):
        super().__init__()
        # 只保留已训练的视觉模块，RSSM 不参与此结构，也不驻留在辅助分支。
        self.visual = teacher.vjepa_adapter.requires_grad_(False).eval()
        self.aggregator = teacher.token_aggregator.requires_grad_(False).eval()
        self.rgb_grid = teacher.rgb_grid
        self.target_grid = teacher.aligned_grid
        self.adapter = adapter
        self.predictor = predictor.requires_grad_(False).eval()
        self.sigma_max = float(sigma_max)

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        self.aggregator.eval()
        self.predictor.eval()
        return self

    @torch.no_grad()
    def teacher_targets(self, batch):
        tokens = self.visual(batch["rgb"], batch["group_ids"])
        grid = align_row_major_tokens(tokens, self.rgb_grid, self.target_grid)
        embedding = self.aggregator(grid.flatten(2, 4).contiguous())
        if embedding.shape[1] != TIME_GROUPS:
            raise ValueError("Stage 2 requires 21 target groups")
        return embedding

    def freeze_adapter(self):
        self.adapter.requires_grad_(False).eval()
        return self

    def forward(self, hidden, dit_grid, teacher_batch, sigma, valid_len, num_views=1):
        if int(dit_grid[0]) != TIME_GROUPS or num_views != 1:
            raise ValueError("Stage 2 requires 21 groups and one view")
        target = self.teacher_targets(teacher_batch)
        tokens = self.adapter(hidden, dit_grid, sigma, valid_len, num_views)
        dtype = next(self.aggregator.parameters()).dtype
        estimated = self.aggregator(tokens.flatten(2, 4).to(dtype))
        predicted = self.predictor.pairs(estimated, teacher_batch["action_segments"]).squeeze(2)
        align = self.predictor.distance(estimated[:, 3:20], target[:, 3:20])
        pred = self.predictor.distance(predicted, target[:, 4:21])
        active = (torch.as_tensor(sigma, device=hidden.device).reshape(-1, 1) <= self.sigma_max).float()
        return {"align": (align * active).mean(), "pred": (pred * active).mean(),
                "align_per_step": align, "pred_per_step": pred,
                "estimated": estimated, "target": target,
                "active_fraction": active.mean()}
