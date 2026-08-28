from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch import nn

from pefm.data import build_evaluator_dataloader
from pefm.models import RSSM, TokenAggregator, VJEPAObservationAdapter, VJEPARSSMEvaluator
from src.bwm.wan_video_action.data.wan_dataset import RoboTwinUnifiedDataset
from src.bwm.wan_video_action.utils import load_action_stats
from src.bwm.wan_video_action.data.operators import LoadCobotAction, create_video_operator
from src.vjepa2.src.models.vision_transformer import VisionTransformer
from src.vjepa2.src.models.ac_predictor import VisionTransformerPredictorAC

def test_bwm_vjepa_rssm_flow():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    model = OmegaConf.merge(base, size)
    cfg = OmegaConf.create({"device": "cuda:0", "model": model, "batch_size": 1, "num_workers": 0, "time_division_factor": 4, "token_dim": 1408})
    cfg.model.rssm.initial = 'zeros'

    dataset_base_path = "/data1/fangxuebin/boundless-world-model/converted_dataset_task1"
    device = torch.device(cfg.device)
    bwmdataset = RoboTwinUnifiedDataset(
        base_path=dataset_base_path,
        metadata_path=f"{dataset_base_path}/metadata.jsonl",
        main_data_operator=create_video_operator(
            base_path=dataset_base_path,
            num_frames=81
        ),
        special_operator_map={
            "action": LoadCobotAction(
                base_path=dataset_base_path,
                stat=load_action_stats(f"{dataset_base_path}/stat.json"),
            ),},)
    batch = next(iter(build_evaluator_dataloader(bwmdataset, cfg.time_division_factor, cfg.batch_size, False, cfg.num_workers)))
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    videoembedder = VisionTransformer(img_size=(256,256), num_frames=81, embed_dim=1408, num_heads=22, depth=40, use_silu=True, use_rope=True, use_activation_checkpointing=True)
    predictor = VisionTransformerPredictorAC(img_size = 256, num_frames=40, patch_size = 16, embed_dim = 1408, action_embed_dim = 14)

    evaluator = VJEPARSSMEvaluator(
        VJEPAObservationAdapter(videoembedder, input_size=256),
        predictor,
        TokenAggregator(token_dim=cfg.token_dim, embed_dim=256, num_heads=2),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    ).to(device)
    output = evaluator(batch)

    assert batch["rgb"].shape == (cfg.batch_size, 1, 3, 81, 480, 640)
    assert batch["eef"].shape == (cfg.batch_size, 21, 14)
    assert torch.equal(torch.unique(batch["group_ids"][0]), torch.arange(21, device=device))
    assert output["visual_tokens"].shape == (cfg.batch_size, 21, 256, 1408)
    assert output["predicted_visual_tokens"].shape == (cfg.batch_size, 21, 256, 1408)
    assert output["embed"].shape == (cfg.batch_size, 21, 256)
    assert output["predicted_context"].shape == (cfg.batch_size, 21, 256)
    assert output["posterior_logits"].shape == (cfg.batch_size, 21, 32, 16)
    assert output["prior_logits"].shape == (cfg.batch_size, 21, 32, 16)
    assert output["posterior_reconstruction"].shape == (cfg.batch_size, 21, 256)
    assert output["prior_prediction"].shape == (cfg.batch_size, 21, 256)
    losses = evaluator.compute_loss(
        output, batch["reset"], cfg.model.kl_free,
        cfg.model.loss_scales.dyn, cfg.model.loss_scales.rep,
    )
    assert set(losses) == {
        "token_prediction", "posterior_reconstruction", "prior_prediction",
        "kl_dynamics", "kl_representation", "kl", "total",
    }
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())

    changed_t0 = dict(output)
    changed_t0["predicted_visual_tokens"] = output["predicted_visual_tokens"].clone()
    changed_t0["predicted_visual_tokens"][:, 0] = 100
    changed_t0["prior_prediction"] = output["prior_prediction"].clone()
    changed_t0["prior_prediction"][:, 0] = 100
    changed_t0["prior_logits"] = output["prior_logits"].clone()
    changed_t0["prior_logits"][:, 0, :, 0] += 10
    changed_losses = evaluator.compute_loss(
        changed_t0, batch["reset"], cfg.model.kl_free,
        cfg.model.loss_scales.dyn, cfg.model.loss_scales.rep,
    )
    for name in ("token_prediction", "prior_prediction", "kl"):
        assert torch.allclose(losses[name], changed_losses[name])

    losses["total"].backward()
    assert next(evaluator.rssm._img_net.parameters()).grad is not None
    assert next(evaluator.rssm._obs_net.parameters()).grad is not None
    assert batch["reset"][0].tolist() == [True] + [False] * 20
    assert torch.count_nonzero(output["predicted_visual_tokens"][:, 0]) == 0

    for value in output.values():
        assert torch.isfinite(value).all()

    assert predictor.action_encoder.weight.grad is not None
    assert predictor.state_encoder.weight.grad is not None
    assert evaluator.token_aggregator.queries.grad is not None
    assert next(evaluator.rssm._deter_net.parameters()).grad is not None
    assert evaluator.posterior_reconstruction_head.weight.grad is not None
    assert evaluator.prior_prediction_head.weight.grad is not None
    assert all(parameter.grad is None for parameter in evaluator.vjepa_adapter.encoder.parameters())
    
    print("test passed")

if __name__ == "__main__":
    test_bwm_vjepa_rssm_flow()
