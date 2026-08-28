from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch import nn

from pefm.data import build_evaluator_dataloader
from pefm.models import RSSM, TokenAggregator, VJEPAObservationAdapter, VJEPARSSMEvaluator
from src.bwm.wan_video_action.data.wan_dataset import RoboTwinUnifiedDataset
from src.bwm.wan_video_action.utils import load_action_stats
from src.bwm.wan_video_action.data.operators import LoadCobotAction, create_video_operator
from src.vjepa2.src.models.utils.patch_embed import PatchEmbed3D
from src.vjepa2.src.models.ac_predictor import VisionTransformerPredictorAC

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

def test_bwm_vjepa_rssm_flow():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    model = OmegaConf.merge(base, size)
    cfg = OmegaConf.create({"device": "cuda:0", "model": model, "batch_size": 1, "num_workers": 0, "time_division_factor": 1, "token_dim": 1408})
    cfg.model.rssm.initial = 'zeros'

    dataset_base_path = "/data1/fangxuebin/pefm/data/bwm_task1_episode0_8f"
    device = torch.device(cfg.device)
    bwmdataset = RoboTwinUnifiedDataset(
        base_path=dataset_base_path,
        metadata_path=f"{dataset_base_path}/metadata.jsonl",
        main_data_operator=create_video_operator(
            base_path=dataset_base_path,
            num_frames=8
        ),
        special_operator_map={
            "action": LoadCobotAction(
                base_path=dataset_base_path,
                num_frames=8,
                time_division_factor=1,
                time_division_remainder=0,
                stat=load_action_stats(f"{dataset_base_path}/stat.json"),
            ),},)
    batch = next(iter(build_evaluator_dataloader(bwmdataset, cfg.time_division_factor, cfg.batch_size, False, cfg.num_workers)))
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    batch["rgb"].requires_grad_()

    videoembedder = PatchEmbed3D(embed_dim=cfg.token_dim)
    predictor = VisionTransformerPredictorAC(img_size = 256, num_frames=14, patch_size = 16, embed_dim = 1408, action_embed_dim = 14)

    evaluator = VJEPARSSMEvaluator(
        VJEPAObservationAdapter(videoembedder, input_size=256),
        predictor,
        TokenAggregator(token_dim=cfg.token_dim, embed_dim=256, num_heads=2),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    ).to(device)
    output = evaluator(batch)

    assert batch["rgb"].shape == (cfg.batch_size, 1, 3, 8, 480, 640)
    assert batch["eef"].shape == (cfg.batch_size, 8, 14)
    assert output["visual_tokens"].shape == (cfg.batch_size, 8, 256, 1408)
    assert output["predicted_visual_tokens"].shape == (cfg.batch_size, 8, 256, 1408)
    assert output["embed"].shape == (cfg.batch_size, 8, 256)
    assert output["predicted_context"].shape == (cfg.batch_size, 8, 256)
    assert output["posterior_logits"].shape == (cfg.batch_size, 8, 32, 16)
    assert output["prior_logits"].shape == (cfg.batch_size, 8, 32, 16)
    (output["posterior_logits"].square().mean() + output["prior_logits"].square().mean()).backward()
    assert batch["rgb"].grad is not None
    assert next(evaluator.rssm._img_net.parameters()).grad is not None
    assert next(evaluator.rssm._obs_net.parameters()).grad is not None
    assert torch.equal(batch["group_ids"][0], torch.arange(8, device=device))
    assert batch["reset"][0].tolist() == [True] + [False] * 7
    assert torch.count_nonzero(output["predicted_visual_tokens"][:, 0]) == 0

    for value in output.values():
        assert torch.isfinite(value).all()

    assert predictor.action_encoder.weight.grad is not None
    assert predictor.state_encoder.weight.grad is not None
    assert evaluator.token_aggregator.queries.grad is not None
    assert next(evaluator.rssm._deter_net.parameters()).grad is not None
    
    print("test passed")

if __name__ == "__main__":
    test_bwm_vjepa_rssm_flow()
