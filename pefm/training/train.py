"""Train the V-JEPA RSSM teacher used by the later latent/Main branch."""

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
from pefm.utils.graceful_exit import GracefulExit
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from dataloader import RawCobotAction, build_RGB_dataloader
from pefm.models import build_evaluator
from pefm.export import (
    check_teacher_contract,
    export_frozen_evaluator,
    teacher_contract,
)
from src.bwm.wan_video_action.data.operators import create_video_operator
from src.bwm.wan_video_action.data.wan_dataset import RoboTwinUnifiedDataset
from src.bwm.wan_video_action.utils import load_action_stats, set_global_seed

NUM_FRAMES = 81
HISTORY_FRAMES = 9
TIME_DIVISION_FACTOR = 4


def loss_reset_for_teacher(batch):
    """Keep the RSSM rollout intact but supervise only future groups."""
    reset = batch["reset"].clone()
    group_ids = batch["group_ids"]
    history_groups = int(group_ids[0, HISTORY_FRAMES - 1].item()) + 1
    reset[:, :history_groups] = True
    return reset


def mask_teacher_losses(output, losses, reset):
    """Apply the future-only mask to the reconstruction term as well."""
    valid = (~reset).to(output["embed"])
    target_embed = output["embed"].detach()
    reconstruction_error = (
        output["posterior_reconstruction"] - target_embed
    ).square().mean(-1)
    losses["posterior_reconstruction"] = (
        reconstruction_error * valid
    ).sum() / valid.sum().clamp_min(1)
    losses["total"] = (
        losses["posterior_reconstruction"]
        + losses["prior_prediction"]
        + losses["kl"]
    )
    return losses


def make_loader(cfg, metadata_name, first_episode, last_episode, shuffle):
    metadata_path = Path(cfg.dataset) / metadata_name
    with metadata_path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    indices = [
        i for i, row in enumerate(rows)
        if first_episode <= int(row.get("source_episode_index", row["episode_index"])) <= last_episode
        and int(row["end_frame"]) - int(row.get("start_frame", 0)) + 1 >= NUM_FRAMES
    ]
    assert indices, f"No episodes {first_episode}-{last_episode} in {metadata_path}"
    dataset = RoboTwinUnifiedDataset(
        base_path=cfg.dataset, metadata_path=str(metadata_path), sample_indices=indices,
        main_data_operator=create_video_operator(base_path=cfg.dataset, num_frames=NUM_FRAMES),
        special_operator_map={"action": RawCobotAction(
            base_path=cfg.dataset, stat=load_action_stats(f"{cfg.dataset}/stat.json"),
        )},
    )
    return build_RGB_dataloader(
        dataset, TIME_DIVISION_FACTOR, cfg.batch_size, shuffle=shuffle, num_workers=cfg.workers,
    )

def run_epoch(model, loader, cfg, optimizer, epoch, split, accelerator, stop=None):
    training = optimizer is not None
    model.train(training)
    core = accelerator.unwrap_model(model)
    totals, count = {}, 0
    for batch_id, batch in enumerate(loader):
        batch = {key: value.to(cfg.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if training: optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            output = model(batch)
            loss_reset = loss_reset_for_teacher(batch)
            losses = core.compute_loss(
                output, loss_reset, cfg.model.kl_free,
                cfg.model.loss_scales.dyn, cfg.model.loss_scales.rep,
            )
            losses = mask_teacher_losses(output, losses, loss_reset)
        grad_norm = 0.0
        if training:
            accelerator.backward(losses["total"])
            grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm))
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
        if stop is not None:
            stop.batch = count
            if stop.sync():
                break
        if cfg.max_batches and count >= cfg.max_batches:
            break
    return {name: value / count for name, value in totals.items()}

def save_state(path, model, optimizer, cfg, epoch, **progress):
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    resolved_config["semantic_teacher"] = teacher_contract(cfg)
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "config": resolved_config, "seed": int(cfg.seed),
        "semantic_teacher": teacher_contract(cfg),
        **progress,
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
    if epoch == cfg.epochs:
        frozen = export_frozen_evaluator(numbered, output / "frozen_evaluator.pt")
        print(f"[checkpoint] frozen evaluator: {frozen}", flush=True)

def main():
    accelerator = Accelerator()
    stop = GracefulExit(accelerator)
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
    if cfg.init_checkpoint:
        raise ValueError(
            "The Stage 2 Teacher must train RSSM, TokenAggregator and heads from "
            "scratch; init_checkpoint is disabled"
        )
    if resume:
        check_teacher_contract(resume, cfg)
    if resume:
        saved = OmegaConf.create(resume["config"])
        saved.device, saved.epochs, saved.resume, saved.seed = cfg.device, cfg.epochs, cfg.resume, resume["seed"]
        saved.init_checkpoint = None
        saved.checkpoint_every = cfg.checkpoint_every
        cfg = saved
    assert cfg.checkpoint_every > 0, "checkpoint_every must be positive"
    cfg.device = str(accelerator.device)
    cfg.model.rssm.device = cfg.device
    set_global_seed(cfg.seed)
    output = Path(cfg.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    accelerator.wait_for_everyone()
    train_loader = make_loader(cfg, "metadata_train.jsonl", 0, 39, True)
    val_loader = make_loader(cfg, "metadata_test.jsonl", 40, 49, False)
    model = build_evaluator(cfg, cfg.vjepa_checkpoint)
    # The V-JEPA encoder defines the observation semantics.  It is always
    # frozen; the RSSM checkpoint produced here is the teacher state space
    # that a later latent/Main branch must reuse.
    model.vjepa_adapter.encoder.requires_grad_(False)
    for name in cfg.get("freeze", []):
        model.get_submodule(name).requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        if epoch >= int(cfg.get("freeze_aggregator_epoch", cfg.epochs + 1)):
            accelerator.unwrap_model(model).token_aggregator.requires_grad_(False)
        train_losses = run_epoch(model, train_loader, cfg, optimizer, epoch, "train", accelerator, stop)
        if stop.requested:
            stop.save(output, model, optimizer, cfg, epoch, "train", save_state)
            break
        with torch.no_grad():
            val_losses = run_epoch(model, val_loader, cfg, None, epoch, "val", accelerator, stop)
        if stop.requested:
            stop.save(output, model, optimizer, cfg, epoch, "val", save_state)
            break
        if accelerator.is_main_process:
            print(f"[epoch] {epoch} train={train_losses} val={val_losses}", flush=True)
            completed = epoch + 1
            if completed % cfg.checkpoint_every == 0 or completed == cfg.epochs:
                save_checkpoint(output, accelerator.unwrap_model(model), optimizer, cfg, completed)
        accelerator.wait_for_everyone()
    accelerator.end_training()
if __name__ == "__main__":
    main()
