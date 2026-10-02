"""阶段 A：缓存固定目标，统计训练集尺度，并训练动作条件 GRU。"""

import argparse
import json
from pathlib import Path

import torch

from pefm.export import load_frozen_evaluator
from pefm.export.stage2 import export_predictor, teacher_identity
from pefm.models import ActionPredictor
from pefm.models.vjepa_adapter import align_row_major_tokens
from .train_adapter import loader, _move


@torch.no_grad()
def cache_targets(args):
    teacher = load_frozen_evaluator(args.teacher_bundle, device=args.device, dtype=torch.float32)
    identity = teacher_identity(args.teacher_bundle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "manifest.jsonl").open("x") as manifest:
        for index, raw in enumerate(loader(args.train_manifest)):
            batch = _move(raw["teacher_batch"], args.device)
            tokens = teacher.vjepa_adapter(batch["rgb"], batch["group_ids"])
            grid = align_row_major_tokens(tokens, teacher.rgb_grid, teacher.aligned_grid)
            embed = teacher.token_aggregator(grid.flatten(2, 4)).cpu()
            payload = {"embed": embed, "action_segments": batch["action_segments"].cpu(),
                       "eef": batch["eef"].cpu(), "task": raw["task"],
                       "teacher_sha256": identity}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cache", "train"))
    for name in ("train-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--teacher-bundle")
    parser.add_argument("--valid-manifest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--no-actions", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.phase == "cache":
        if not args.teacher_bundle:
            parser.error("cache requires --teacher-bundle")
        cache_targets(args)
        return
    if not args.valid_manifest:
        parser.error("train requires --valid-manifest")
    train, valid = list(loader(args.train_manifest)), list(loader(args.valid_manifest))
    identity = train[0]["teacher_sha256"]
    if any(r["teacher_sha256"] != identity for r in train + valid):
        raise ValueError("All target caches must use the same frozen Teacher")
    model = ActionPredictor(train[0]["embed"].shape[-1], args.hidden_dim, not args.no_actions)
    statistics = fit_statistics(model, train)
    probe = motion_probe(train, valid)
    print(json.dumps({"statistics": statistics, "motion_probe": probe}), flush=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for index in torch.randperm(len(train)).tolist():
            record = _move(train[index], args.device)
            prediction = model.pairs(record["embed"], record["action_segments"], args.horizon)
            loss = model.distance(prediction, targets_at_horizon(record["embed"], args.horizon)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        metrics = evaluate(model, valid, args.horizon)
        message = {"epoch": epoch, "train": sum(losses) / len(losses), "valid": metrics}
        print(json.dumps(message), flush=True)
        with (output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(message) + "\n")
        if metrics["prediction"] < best:
            best = metrics["prediction"]
            export_predictor(output / "predictor_best.pt", model, identity,
                              epoch=epoch, horizon=args.horizon, metrics=message,
                              statistics=statistics, motion_probe=probe, seed=args.seed)


if __name__ == "__main__":
    main()
