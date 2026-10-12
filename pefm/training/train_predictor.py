"""阶段 A：缓存固定目标，统计训练集尺度，并训练动作条件 GRU。"""

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from pefm.export import load_frozen_visual_teacher
from pefm.export.stage2 import export_predictor, load_predictor, teacher_identity
from pefm.models import ActionPredictor
from pefm.models.vjepa_adapter import align_row_major_tokens
from pefm.utils.graceful_exit import GracefulExit
from .train_adapter import loader, _move


@torch.no_grad()
def cache_targets(args):
    teacher = load_frozen_visual_teacher(args.teacher_bundle, device=args.device, dtype=getattr(torch, args.teacher_dtype))
    identity = teacher_identity(args.teacher_bundle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.jsonl").open("x") as manifest:
        for index, raw in enumerate(loader(args.train_manifest)):
            batch = _move(raw["teacher_batch"], args.device)
            tokens = teacher.vjepa_adapter(batch["rgb"], batch["group_ids"])
            grid = align_row_major_tokens(tokens, teacher.rgb_grid, teacher.aligned_grid)
            embed = teacher.token_aggregator(grid.flatten(2, 4).contiguous()).cpu()
            payload = {"embed": embed, "action_segments": batch["action_segments"].cpu(),
                       "eef": batch["eef"].cpu(), "task": raw["task"],
                       "teacher_sha256": identity, "teacher_dtype": args.teacher_dtype,
                       "sample_index": raw["sample_index"],
                       "source_video": raw["source_video"],
                       "frame_indices": raw["frame_indices"]}
            path = output / f"sample_{index:06d}.pt"
            torch.save(payload, path)
            manifest.write(json.dumps({"path": path.name}) + "\n")
            manifest.flush()
            print(f"[cache] {index + 1} {path}", flush=True)


def fit_statistics(model, records):
    # 只用训练集的 3..20 组；统计量随 predictor 保存，B/C 不重新拟合。
    values = torch.cat([r["embed"][:, 3:21].flatten(0, 1) for r in records]).double()
    std = values.std(0, unbiased=False)
    if not torch.isfinite(values).all() or float(std.max()) < 1e-6:
        raise ValueError("Frozen embeddings are non-finite or constant")
    model.mean.copy_(values.mean(0).float())
    model.std.copy_(std.clamp_min(1e-6).float())
    return {"variance_mean": float(values.var(0, unbiased=False).mean()),
            "std_min": float(std.min()), "std_max": float(std.max())}


def targets_at_horizon(embed, horizon):
    return torch.stack([embed[:, t + 1:t + 1 + horizon]
                        for t in range(3, 21 - horizon)], dim=1)


@torch.no_grad()
def evaluate(model, records, horizon=1):
    totals = {}
    # 同任务内轮换轨迹，保持每条动作片段自身的时间顺序。
    task_indices = {}
    for index, record in enumerate(records):
        task_indices.setdefault(record["task"], []).append(index)
    if any(len(indices) < 2 for indices in task_indices.values()):
        raise ValueError("Action shuffle diagnostic needs at least two held-out records per task")
    partners = {index: indices[(offset + 1) % len(indices)]
                for indices in task_indices.values() for offset, index in enumerate(indices)}
    device = next(model.parameters()).device
    for index, record in enumerate(records):
        batch = _move(record, device)
        embed, actions = batch["embed"], batch["action_segments"]
        target = targets_at_horizon(embed, horizon)
        predictions = {
            "prediction": model.pairs(embed, actions, horizon),
            "mean": model.mean.expand_as(target),
            "persistence": embed[:, 3:21-horizon].unsqueeze(2).expand_as(target),
            "time_shift": model.pairs(embed, actions.roll(1, 1), horizon),
            "task_shuffle": model.pairs(embed, records[partners[index]]["action_segments"].to(device), horizon),
        }
        for name, prediction in predictions.items():
            totals[name] = totals.get(name, 0.) + float(model.distance(prediction, target).mean())
    return {k: v / len(records) for k, v in totals.items()}


@torch.no_grad()
def motion_probe(train, valid):
    """同一个线性 ridge 读出器检验双臂位置与下一组位移。"""
    def xy(records):
        x = torch.cat([r["embed"][:, 3:20].flatten(0, 1) for r in records]).double()
        position = torch.cat([r["eef"][:, 3:20, [0, 1, 2, 7, 8, 9]].flatten(0, 1)
                              for r in records]).double()
        delta = torch.cat([(r["eef"][:, 4:21, [0, 1, 2, 7, 8, 9]] -
                            r["eef"][:, 3:20, [0, 1, 2, 7, 8, 9]]).flatten(0, 1)
                           for r in records]).double()
        return x, torch.cat((position, delta), -1)
    x, y = xy(train)
    vx, vy = xy(valid)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x, vx = (x - mean) / std, (vx - mean) / std
    x = torch.cat((x, torch.ones(len(x), 1)), 1)
    vx = torch.cat((vx, torch.ones(len(vx), 1)), 1)
    weights = torch.linalg.solve(x.T @ x + 1e-3 * torch.eye(x.shape[1]), x.T @ y)
    error, baseline = (vx @ weights - vy).square(), (y.mean(0) - vy).square()
    return {"position_mse": float(error[:, :6].mean()), "position_mean_mse": float(baseline[:, :6].mean()),
            "displacement_mse": float(error[:, 6:].mean()), "displacement_mean_mse": float(baseline[:, 6:].mean())}


def run_epoch(model, batches, cfg, optimizer, epoch, split, accelerator, stop=None):
    """Phase A optimizes only standardized future embedding MSE."""
    training = optimizer is not None
    model.train(training)
    core = accelerator.unwrap_model(model)
    total, count = 0., 0
    for batch_id, raw in enumerate(batches):
        record = _move(raw, cfg.device)
        embedding, actions = record["embed"].detach(), record["action_segments"]
        horizon = int(cfg.horizon)
        segments = torch.stack([actions[:, t + 1:t + 1 + horizon]
                                for t in range(3, 21 - horizon)], dim=1)
        with torch.set_grad_enabled(training):
            prediction = model(embedding[:, 3:21 - horizon], segments)
            loss = core.distance(prediction, targets_at_horizon(embedding, horizon)).mean()
            if training:
                optimizer.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
        total += float(accelerator.reduce(loss.detach(), "mean"))
        count += 1
        accelerator.print(f"[{split}] epoch={epoch} batch={batch_id} pred={float(loss.detach()):.5f}",
                          flush=True)
        if stop is not None:
            stop.batch = count
            if stop.sync():
                break
        if cfg.get("max_batches", 0) and count >= cfg.max_batches:
            break
    if not count:
        raise ValueError("No predictor batches")
    return {"pred": total / count, "total": total / count}


def save_state(path, model, optimizer, cfg, epoch, **progress):
    """A resumable predictor checkpoint never contains visual or stochastic weights."""
    export_predictor(path, model, cfg.teacher_sha256, epoch=epoch,
                     teacher_dtype=cfg.teacher_dtype,
                     horizon=int(cfg.horizon), seed=int(cfg.seed),
                     optimizer=optimizer.state_dict(),
                     config=OmegaConf.to_container(cfg, resolve=True), **progress)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", choices=("cache", "train"), default="train")
    parser.add_argument("--config")
    parser.add_argument("--train-manifest")
    parser.add_argument("--valid-manifest")
    parser.add_argument("--teacher-bundle")
    parser.add_argument("--teacher-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, choices=(1, 2, 4), default=1)
    action_mode = parser.add_mutually_exclusive_group()
    action_mode.add_argument("--no-actions", dest="no_actions", action="store_true")
    action_mode.add_argument("--actions", dest="no_actions", action="store_false")
    parser.set_defaults(no_actions=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--max-grad-norm", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--max-batches", type=int, default=0)
    config_arg, _ = parser.parse_known_args(argv)
    if config_arg.config:
        config = OmegaConf.to_container(OmegaConf.load(config_arg.config), resolve=True)
        allowed = {action.dest for action in parser._actions}
        unknown = set(config) - allowed
        if unknown:
            parser.error(f"Unsupported Stage 2 config keys: {sorted(unknown)}. "
                         "Use the fixed-embedding predictor config, not the old Teacher config.")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if not args.train_manifest or not args.output:
        parser.error("--train-manifest and --output are required")
    if args.phase == "cache" and not args.teacher_bundle:
        parser.error("cache requires --teacher-bundle")
    if args.phase == "train" and not args.valid_manifest:
        parser.error("train requires --valid-manifest")
    if args.phase not in ("cache", "train"):
        parser.error("phase must be cache or train")
    if args.horizon not in (1, 2, 4) or args.epochs < 1 or args.hidden_dim < 1:
        parser.error("Require horizon in {1,2,4}, epochs >= 1 and hidden-dim >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if args.phase == "cache":
        cache_targets(args)
        return
    train, valid = list(loader(args.train_manifest)), list(loader(args.valid_manifest))
    identity = train[0]["teacher_sha256"]
    if any(r["teacher_sha256"] != identity for r in train + valid):
        raise ValueError("All target caches must use the same frozen Teacher")
    assert all(r["teacher_dtype"] == args.teacher_dtype and r["embed"].dtype == getattr(torch, args.teacher_dtype)
               for r in train + valid), "Target cache precision differs from Phase A"
    cfg = OmegaConf.create(vars(args))
    cfg.teacher_sha256 = identity
    accelerator = Accelerator(cpu=args.device == "cpu")
    cfg.device = str(accelerator.device)
    stop = GracefulExit(accelerator)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume:
        if resume["teacher_sha256"] != identity or resume["horizon"] != args.horizon:
            raise ValueError("Resume requires the same Teacher and prediction horizon")
        if resume["hidden_dim"] != args.hidden_dim or resume["use_actions"] != (not args.no_actions):
            raise ValueError("Resume predictor architecture differs from config")
        if resume["embed_dim"] != train[0]["embed"].shape[-1]:
            raise ValueError("Resume embedding dimension differs from target caches")
        model = load_predictor(resume)
        cfg.seed = int(resume["seed"])
        torch.manual_seed(cfg.seed)
        statistics = resume["config"]["statistics"]
        probe = resume["config"]["motion_probe"]
    else:
        model = ActionPredictor(train[0]["embed"].shape[-1], args.hidden_dim, not args.no_actions)
        statistics = fit_statistics(model, train)
        probe = motion_probe(train, valid)
    cfg.statistics, cfg.motion_probe = statistics, probe
    accelerator.print(json.dumps({"statistics": statistics, "motion_probe": probe}), flush=True)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"])
    train_loader, valid_loader = loader(args.train_manifest, True), loader(args.valid_manifest)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader)
    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    accelerator.wait_for_everyone()
    best = float(resume["config"].get("best", float("inf"))) if resume else float("inf")
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, cfg, optimizer, epoch, "train", accelerator, stop)
        if stop.requested:
            stop.save(output, model, optimizer, cfg, epoch, "train", save_state)
            break
        valid_metrics = run_epoch(model, valid_loader, cfg, None, epoch, "val", accelerator, stop)
        if stop.requested:
            stop.save(output, model, optimizer, cfg, epoch, "val", save_state)
            break
        core = accelerator.unwrap_model(model)
        diagnostics = evaluate(core, valid, args.horizon)
        record = {"epoch": epoch + 1, "train": train_metrics, "valid": valid_metrics,
                  "diagnostics": diagnostics}
        improved = diagnostics["prediction"] < best
        best = min(best, diagnostics["prediction"])
        cfg.best = best
        if accelerator.is_main_process:
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(record) + "\n")
            save_state(output / "latest.pt", core, optimizer, cfg, epoch + 1)
            if improved:
                export_predictor(output / "predictor_best.pt", core, identity,
                                 teacher_dtype=args.teacher_dtype,
                                 epoch=epoch + 1, horizon=args.horizon, metrics=record,
                                 statistics=statistics, motion_probe=probe, seed=cfg.seed)
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
