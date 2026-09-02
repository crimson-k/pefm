"""Minimal positive-pair training loop for the Stage-1 evaluator."""

import argparse
import json
import os
import sys
from pathlib import Path
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pefm.training.config.parser import merge_yaml_and_args
from pefm.data import build_evaluator_dataloader
from pefm.models import RSSM, TokenAggregator, VJEPAObservationAdapter, VJEPARSSMEvaluator
from src.bwm.wan_video_action.data.operators import LoadCobotAction, create_video_operator
from src.bwm.wan_video_action.data.wan_dataset import RoboTwinUnifiedDataset
from src.bwm.wan_video_action.utils import load_action_stats, set_global_seed
from src.vjepa2.src.models.ac_predictor import VisionTransformerPredictorAC
from src.vjepa2.src.models.vision_transformer import VisionTransformer

def make_loader(cfg, metadata_name, first_episode, last_episode, shuffle):
    metadata_path = Path(cfg.dataset) / metadata_name
    with metadata_path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    indices = [
        i for i, row in enumerate(rows)
        if first_episode <= int(row.get("source_episode_index", row["episode_index"])) <= last_episode
        and int(row["end_frame"]) - int(row.get("start_frame", 0)) + 1 >= 81
    ]
    assert indices, f"No episodes {first_episode}-{last_episode} in {metadata_path}"
    dataset = RoboTwinUnifiedDataset(
        base_path=cfg.dataset, metadata_path=str(metadata_path), sample_indices=indices,
        main_data_operator=create_video_operator(base_path=cfg.dataset, num_frames=81),
        special_operator_map={"action": LoadCobotAction(
            base_path=cfg.dataset, stat=load_action_stats(f"{cfg.dataset}/stat.json"),
        )},
    )
    return build_evaluator_dataloader(
        dataset, 4, cfg.batch_size, shuffle=shuffle, num_workers=cfg.workers,
    )

def build_model(cfg):
    checkpoint = torch.load(cfg.vjepa_checkpoint, map_location="cpu", weights_only=False)
    clean = lambda state: {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    encoder_state, predictor_state = clean(checkpoint["encoder"]), clean(checkpoint["predictor"])
    del checkpoint
    encoder = VisionTransformer(
        img_size=(256, 256), num_frames=81, embed_dim=1408, depth=40,
        num_heads=22, mlp_ratio=48 / 11, use_silu=False, use_rope=True,
        use_activation_checkpointing=True,
    )
    encoder.load_state_dict(encoder_state, strict=True)
    del encoder_state
    predictor = VisionTransformerPredictorAC(
        img_size=256, num_frames=40, patch_size=16, embed_dim=1408, action_embed_dim=14,
    )
    current = predictor.state_dict()
    compatible = {k: v for k, v in predictor_state.items() if k in current and v.shape == current[k].shape}
    loaded = predictor.load_state_dict(compatible, strict=False)
    assert set(loaded.missing_keys) == {
        "action_encoder.weight", "state_encoder.weight", "extrinsics_encoder.weight",
    }
    return VJEPARSSMEvaluator(
        VJEPAObservationAdapter(encoder, input_size=256), predictor,
        TokenAggregator(token_dim=1408, embed_dim=256, num_heads=4),
        RSSM(cfg.model.rssm, embed_size=256, act_dim=14),
    ).to(cfg.device)

def run_epoch(model, loader, cfg, optimizer, epoch, split, accelerator):
    training = optimizer is not None
    model.train(training)
    core = accelerator.unwrap_model(model)
    totals, count = {}, 0
    for batch_id, batch in enumerate(loader):
        batch = {key: value.to(cfg.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            output = model(batch)
            losses = core.compute_loss(
                output, batch["reset"], cfg.model.kl_free,
                cfg.model.loss_scales.dyn, cfg.model.loss_scales.rep,
            )
        grad_norm = 0.0
        if training and not cfg.predictor_warmup:
            accelerator.backward(losses["total"])
            grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
            optimizer.step()
        elif training and cfg.predictor_warmup:
            accelerator.backward(losses["token_prediction"])
            grad_norm = float(accelerator.clip_grad_norm_(
                [parameter for parameter in core.context_predictor.parameters() if parameter.requires_grad],
                cfg.max_grad_norm))
            optimizer.step()
        names = list(losses)
        reduced = accelerator.reduce(torch.stack([losses[name].detach() for name in names]), "mean")
        logged = dict(zip(names, reduced))
        count += 1
        for name, value in logged.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        values = " ".join(f"{name}={float(value):.5f}" for name, value in logged.items())
        if accelerator.is_main_process:
            print(f"[{split}] epoch={epoch} batch={batch_id} grad_norm={grad_norm:.4f} {values}", flush=True)
        if cfg.max_batches and count >= cfg.max_batches:
            break
    return {name: value / count for name, value in totals.items()}

def evaluate(model, loader, cfg, accelerator):
    model.eval()
    core = accelerator.unwrap_model(model)
    collected = []

    def metrics(output, valid):
        values = [
            (output["predicted_visual_tokens"] - output["visual_tokens"]).abs().mean((-1, -2)),
            (output["prior_prediction"] - output["embed"]).square().mean(-1),
            torch.distributions.kl_divergence(
                core.rssm.get_dist(output["posterior_logits"]),
                core.rssm.get_dist(output["prior_logits"]),
            ),
        ]
        return torch.stack([(value * valid).sum(1) / valid.sum(1) for value in values], -1)

    with torch.inference_mode():
        for batch_id, batch in enumerate(loader):
            batch = {key: value.to(cfg.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            shifted = batch["eef"].clone()
            shifted[:, 3] = batch["eef"][:, 2]
            shifted[:, 4:] = batch["eef"][:, 3:-1]
            shuffled = batch["eef"].roll(1, 0)
            valid = (~batch["reset"]).to(batch["eef"].dtype)
            valid[:, :4] = 0
            outputs = []
            for eef in (batch["eef"], shifted, shuffled):
                variant = {**batch, "eef": eef}
                outputs.append(model(variant))
            values = torch.cat([metrics(output, valid) for output in outputs], -1)
            collected.append(accelerator.gather_for_metrics(values).cpu())

    values = torch.cat(collected)
    names = ("token_prediction", "prior_prediction", "raw_kl")
    result = {"samples": len(values)}
    for index, variant in enumerate(("correct", "shifted", "shuffled")):
        result[variant] = dict(zip(names, values[:, index * 3:(index + 1) * 3].mean(0).tolist()))
    for index, variant in ((1, "shifted"), (2, "shuffled")):
        wrong, correct = values[:, index * 3:(index + 1) * 3], values[:, :3]
        result[f"{variant}_gap"] = dict(zip(names, (wrong - correct).mean(0).tolist()))
        result[f"{variant}_win_rate"] = dict(zip(names, (wrong > correct).float().mean(0).tolist()))
    return result

def save_state(path, model, optimizer, cfg, epoch):
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "config": OmegaConf.to_container(cfg, resolve=True), "seed": int(cfg.seed),
    }
    temporary = str(path) + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)

def save_checkpoint(output, model, optimizer, cfg, epoch):
    latest = output / "latest.pt"
    save_state(latest, model, optimizer, cfg, epoch)
    numbered = output / f"epoch_{epoch:04d}.pt"
    if numbered.exists():
        numbered.unlink()
    os.link(latest, numbered)  # Hard link: versioned checkpoint without duplicating 7.7 GB.

def main():
    accelerator = Accelerator()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file. CLI args override YAML config.")
    args = parser.parse_args()
    if args.config is not None:
        args = merge_yaml_and_args(args.config, args)
    base, size = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml"), OmegaConf.load(args.rssm_model_size)
    for unused in ("encoder", "decoder"): del base[unused]
    cfg = OmegaConf.create({**vars(args), "model": OmegaConf.merge(base, size)})
    cfg.model.rssm.initial = "zeros"
    resume = torch.load(cfg.resume, map_location="cpu", weights_only=False) if cfg.resume else None
    assert not (resume and cfg.init_checkpoint), "Use resume or init_checkpoint, not both"
    if resume:
        saved = OmegaConf.create(resume["config"])
        saved.device, saved.epochs, saved.resume, saved.seed = cfg.device, cfg.epochs, cfg.resume, resume["seed"]
        saved.init_checkpoint = None
        saved.checkpoint_every = cfg.checkpoint_every
        saved.eval_mode = cfg.eval_mode
        cfg = saved
    assert cfg.checkpoint_every > 0, "checkpoint_every must be positive"
    cfg.device = str(accelerator.device)
    cfg.model.rssm.device = cfg.device
    set_global_seed(cfg.seed)
    output = Path(cfg.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        config_name = "resolved_soft_eval_config.yaml" if cfg.eval_mode else "resolved_config.yaml"
        OmegaConf.save(cfg, output / config_name, resolve=True)
    accelerator.wait_for_everyone()
    train_loader = make_loader(cfg, "metadata_train.jsonl", 0, 39, True)
    val_loader = make_loader(cfg, "metadata_test.jsonl", 40, 49, False)
    model = build_model(cfg)
    if cfg.init_checkpoint:
        initialized = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initialized["model"])
        del initialized
    for name in cfg.get("freeze", []):
        model.get_submodule(name).requires_grad_(False)
    if cfg.eval_mode:
        assert resume, "eval_mode requires a resume checkpoint"
        model.load_state_dict(resume["model"])
        model.requires_grad_(False)
        model, val_loader = accelerator.prepare(model, val_loader)
        result = evaluate(model, val_loader, cfg, accelerator)
        if accelerator.is_main_process:
            path = output / f"action_eval_soft_epoch_{int(resume['epoch']):04d}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        accelerator.end_training()
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if cfg.predictor_warmup:
        optimizer = torch.optim.AdamW([parameter for parameter in model.context_predictor.parameters() if parameter.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch = 0
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"])
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader,
    )
    for epoch in range(start_epoch, cfg.epochs):
        train_losses = run_epoch(model, train_loader, cfg, optimizer, epoch, "train", accelerator)
        with torch.no_grad():
            val_losses = run_epoch(model, val_loader, cfg, None, epoch, "val", accelerator)
        if accelerator.is_main_process:
            print(f"[epoch] {epoch} train={train_losses} val={val_losses}", flush=True)
            completed = epoch + 1
            if completed % cfg.checkpoint_every == 0 or completed == cfg.epochs:
                save_checkpoint(output, accelerator.unwrap_model(model), optimizer, cfg, completed)
        accelerator.wait_for_everyone()
    accelerator.end_training()
if __name__ == "__main__":
    main()
