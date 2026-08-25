from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from pefm.data import build_evaluator_dataloader
from pefm.models import RSSM, TokenAggregator, VJEPAObservationAdapter, VJEPARSSMEvaluator


class DummyBWMDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "video": torch.zeros(1, 3, 81, 16, 16),
            "action": torch.arange(81 * 14).reshape(1, 81, 14).float(),
            "frame_indices": [0] + list(range(8)) + list(range(8, 80)),
            "episode_index": index,
            "task": "dummy",
            "split": "train",
        }


class DummyVJEPA(nn.Module):
    def forward(self, clips):
        return F.adaptive_avg_pool3d(clips, (1, 2, 2)).flatten(2).transpose(1, 2)


def test_bwm_vjepa_rssm_flow():
    batch = next(iter(build_evaluator_dataloader(DummyBWMDataset(), 2, False, 0)))
    batch["rgb"].requires_grad_()
    config = SimpleNamespace(
        stoch=4,
        deter=32,
        hidden=16,
        discrete=4,
        act="SiLU",
        unimix_ratio=0.01,
        initial="zeros",
        device="cpu",
        obs_layers=1,
        img_layers=1,
        dyn_layers=1,
        blocks=4,
    )
    model = VJEPARSSMEvaluator(
        VJEPAObservationAdapter(DummyVJEPA(), input_size=16),
        TokenAggregator(token_dim=3, embed_dim=8, num_heads=2),
        RSSM(config, embed_size=8, act_dim=14),
    )
    output = model(batch)

    assert batch["rgb"].shape == (2, 1, 3, 81, 16, 16)
    assert batch["eef"].shape == (2, 21, 14)
    assert output["visual_tokens"].shape == (2, 21, 4, 3)
    assert output["real_embed"].shape == (2, 21, 8)
    assert output["posterior_logits"].shape == (2, 21, 4, 4)
    assert output["prior_logits"].shape == (2, 21, 4, 4)
    output["real_embed"].square().mean().backward()
    assert batch["rgb"].grad is not None
