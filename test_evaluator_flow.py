from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch import nn

from pefm.data import build_evaluator_dataloader
from pefm.models import DummyVisualPredictor, RSSM, TokenAggregator, VJEPAObservationAdapter, VJEPARSSMEvaluator
from src.bwm.wan_video_action.data.wan_dataset import RoboTwinUnifiedDataset
from src.bwm.wan_video_action.utils import load_action_stats
from src.bwm.wan_video_action.data.operators import LoadCobotAction, create_video_operator


class DummyBWMDataset:
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "video": torch.zeros(1, 3, 81, 480, 640),
            "action": torch.arange(81 * 14).reshape(1, 81, 14).float(),
            "frame_indices": [0] + list(range(8)) + list(range(8, 80)),
            "episode_index": index,
            "task": "dummy",
            "split": "train",
        }


# (b*V*F, 3, 2, 256, 256) -> (B*V*F, 256, 1408)
class DummyVJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 1408)
    def forward(self, clips):
        x = F.adaptive_avg_pool3d(clips, (1, 16, 16)).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

def test_bwm_vjepa_rssm_flow():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    model = OmegaConf.merge(base, size)
    cfg = OmegaConf.create({"device": "cuda:0", "model": model, "batch_size": 1, "num_workers": 0,})
    cfg.model.rssm.initial = 'zeros'

    dataset_base_path = "/data1/fangxuebin/boundless-world-model/converted_dataset_task1"
    device = torch.device(cfg.device)
    bwmdataset = RoboTwinUnifiedDataset(
        base_path=dataset_base_path,
        metadata_path=f"{dataset_base_path}/metadata.jsonl",
        main_data_operator=create_video_operator(
            base_path=dataset_base_path,
        ),
        special_operator_map={
            "action": LoadCobotAction(
                base_path=dataset_base_path,
                stat=load_action_stats(f"{dataset_base_path}/stat.json"),
            ),},)
    batch = next(iter(build_evaluator_dataloader(bwmdataset, cfg.batch_size, False, cfg.num_workers)))
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    batch["rgb"].requires_grad_()
    
    evaluator = VJEPARSSMEvaluator(
        VJEPAObservationAdapter(DummyVJEPA(), input_size=256),
        DummyVisualPredictor(token_dim=1408, action_dim=14),
        TokenAggregator(token_dim=1408, embed_dim=256, num_heads=2),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    ).to(device)
    output = evaluator(batch)

    assert batch["rgb"].shape == (cfg.batch_size, 1, 3, 81, 480, 640)
    assert batch["eef"].shape == (cfg.batch_size, 21, 14)
    assert output["visual_tokens"].shape == (cfg.batch_size, 21, 256, 1408)
    assert output["predicted_visual_tokens"].shape == (cfg.batch_size, 21, 256, 1408)
    assert output["embed"].shape == (cfg.batch_size, 21, 256)
    assert output["predicted_context"].shape == (cfg.batch_size, 21, 256)
    assert output["posterior_logits"].shape == (cfg.batch_size, 21, 32, 16)
    assert output["prior_logits"].shape == (cfg.batch_size, 21, 32, 16)
    (output["posterior_logits"].square().mean() + output["prior_logits"].square().mean()).backward()
    assert batch["rgb"].grad is not None
    assert evaluator.context_predictor.action_proj.weight.grad is not None
    assert next(evaluator.rssm._img_net.parameters()).grad is not None
    assert next(evaluator.rssm._obs_net.parameters()).grad is not None
    print("test passed")

if __name__ == "__main__":
    test_bwm_vjepa_rssm_flow()
