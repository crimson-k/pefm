"""冻结 DiT 的单 token 局部 RGB 读出；每个对照独立拟合共享 ridge 读出器。"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


CONDITIONS = (
    "matched", "space_dx1", "space_dx4", "time_dt1", "time_dt4",
    "spatial_mean", "global_mean", "coords_sigma",
)


def rgb_targets(rgb, grid, patch_size):
    """第 t>0 组对应帧 4t-3…4t；只在各自 RGB 区域内部平均。"""
    time, height, width = grid
    frames = rgb[0, 0].permute(1, 0, 2, 3).float()
    assert len(frames) == 4 * (time - 1) + 1
    assert frames.shape[-2] % height == frames.shape[-1] % width == 0
    assert frames.min() >= -1.00001 and frames.max() <= 1.00001
    frames = (frames + 1) / 2  # 缓存的 RGB 使用 [-1,1]，报告使用 [0,1]。
    cell_height, cell_width = frames.shape[-2] // height, frames.shape[-1] // width
    assert cell_height % patch_size == cell_width % patch_size == 0
    pooled = F.avg_pool2d(frames, (cell_height // patch_size, cell_width // patch_size))
    grouped = pooled[1:].reshape(time - 1, 4, *pooled.shape[1:]).mean(1)
    target = grouped.reshape(time - 1, 3, height, patch_size, width, patch_size)
    target = target.permute(0, 2, 4, 1, 3, 5).reshape(time - 1, height, width, -1)
    changes = (frames[1:] - frames[:-1]).abs()
    changes = F.avg_pool2d(changes, (cell_height, cell_width)).mean(1)
    motion = changes.reshape(time - 1, 4, height, width).mean(1)
    return target[2:], motion[2:]  # 只评测未来组 t=3…20，不含条件帧。


def prepare(args):
    splits = {}
    for split in ("train", "valid"):
        manifest = args.capture_root / split / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if args.max_samples:
            rows = rows[:args.max_samples]
        hidden, target, motion, metadata = [], [], [], []
        for index, row in enumerate(rows):
            raw = torch.load(manifest.parent / row["path"], mmap=True, map_location="cpu", weights_only=False)
            assert raw["selected_block"] == 15 and raw["num_views"] == 1
            grid = tuple(raw["dit_grid"])
            assert grid == (21, 6, 8) and raw["valid_len"] == np.prod(grid)
            y, m = rgb_targets(raw["teacher_batch"]["rgb"], grid, args.patch_size)
            hidden.append(raw["hidden"][0].reshape(*grid, -1).clone())
            target.append(y)
            motion.append(m)
            metadata.append({"source_video": raw["source_video"], "task": raw["task"],
                             "sigma": float(raw["noise_sigma"].mean()),
                             "frame_indices": raw["frame_indices"],
                             "action_encoder_sha256": raw["action_encoder_sha256"]})
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(f"[prepare] {split} {index + 1}/{len(rows)}", flush=True)
        splits[split] = {"hidden": torch.stack(hidden), "target": torch.stack(target),
                         "motion": torch.stack(motion), "metadata": metadata,
                         "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    train_videos = {m["source_video"] for m in splits["train"]["metadata"]}
    valid_videos = {m["source_video"] for m in splits["valid"]["metadata"]}
    assert not train_videos.intersection(valid_videos)
    assert len({m["action_encoder_sha256"] for s in splits.values() for m in s["metadata"]}) == 1
    torch.save(splits, args.output / "prepared.pt")


def features(hidden, metadata, condition):
    """共同目标为 t=3…16；时间平移不循环，空间交换的距离固定为 1/4 格。"""
    future = hidden[:, 3:17]
    if condition == "matched":
        x = future
    elif condition.startswith("space_dx"):
        columns = torch.arange(future.shape[3], device=hidden.device) ^ int(condition[-1])
        x = future[:, :, :, columns]
    elif condition.startswith("time_dt"):
        offset = int(condition[-1])
        x = hidden[:, 3 + offset:17 + offset]
    elif condition == "spatial_mean":
        x = future.mean((2, 3), keepdim=True).expand_as(future)
    elif condition == "global_mean":
        x = hidden.mean((1, 2, 3), keepdim=True).expand_as(future)
    elif condition == "coords_sigma":
        x = torch.zeros_like(future)
    else:
        raise ValueError(condition)
    n, time, height, width, dim = x.shape
    # 独热空间/时间坐标比直线坐标基线更能拟合固定位置的背景模板。
    space = F.one_hot(torch.arange(height * width, device=x.device), height * width)
    time_ids = F.one_hot(torch.arange(time, device=x.device), time)
    space = space.reshape(1, 1, height, width, -1).expand(n, time, -1, -1, -1)
    time_ids = time_ids.reshape(1, time, 1, 1, -1).expand(n, -1, height, width, -1)
    sigma = torch.tensor([m["sigma"] for m in metadata], device=x.device)
    sigma = sigma.reshape(n, 1, 1, 1, 1).expand(n, time, height, width, 1)
    return torch.cat((x, space, time_ids, sigma, sigma.square()), dim=-1).reshape(-1, dim + height * width + time + 2)


def tuning_split(metadata, seed):
    tasks = defaultdict(set)
    for m in metadata:
        tasks[m["task"]].add(m["source_video"])
    rng = np.random.default_rng(seed)
    held_out = set()
    for videos in tasks.values():
        videos = sorted(videos)
        assert len(videos) >= 2
        held_out.update(rng.choice(videos, max(1, len(videos) // 5), replace=False).tolist())
    return torch.tensor([m["source_video"] not in held_out for m in metadata])


def normal_equations(x, y):
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
    x = torch.cat(((x - mean) / std, torch.ones(len(x), 1, device=x.device)), 1)
    gram = (x.T @ x / len(x)).double()
    rhs = (x.T @ y / len(x)).double()
    return gram, rhs, mean, std


def solve(gram, rhs, ridge):
    penalty = torch.eye(len(gram), device=gram.device, dtype=gram.dtype) * ridge
    penalty[-1, -1] = 0  # 截距不受正则化。
    return torch.linalg.solve(gram + penalty, rhs).float()


def predict(x, weight, mean, std):
    result = []
    for batch in x.split(8192):
        batch = torch.cat(((batch - mean) / std, torch.ones(len(batch), 1, device=x.device)), 1)
        result.append(batch @ weight)
    return torch.cat(result)


def aggregate(errors, motion, metadata):
    """先在源视频内部合并窗口，随后对视频等权平均，避免长视频支配结果。"""
    videos = {}
    for video in sorted({m["source_video"] for m in metadata}):
        ids = [i for i, m in enumerate(metadata) if m["source_video"] == video]
        error = errors[ids]
        mask = motion[ids]
        moving = error[mask]
        videos[video] = {"task": metadata[ids[0]]["task"], "mse": float(error.mean()),
                         "motion_mse": float(moving.mean()) if len(moving) else None,
                         "motion_tokens": int(mask.sum()), "tokens": error.numel()}
    return videos


def paired_comparisons(video_metrics, draws, seed):
    rng = np.random.default_rng(seed)
    names = [name for name in CONDITIONS if name != "matched"]
    videos = sorted(video_metrics["matched"])
    tasks = sorted({video_metrics["matched"][v]["task"] for v in videos})
    task_ids = [[i for i, v in enumerate(videos) if video_metrics["matched"][v]["task"] == t] for t in tasks]
    indices = np.concatenate([rng.choice(ids, (draws, len(ids)), replace=True) for ids in task_ids], axis=1)
    report = {}
    for metric in ("mse", "motion_mse"):
        matched = np.array([video_metrics["matched"][v][metric] for v in videos], dtype=float)
        assert np.isfinite(matched).all()
        report[metric] = {}
        for name in names:
            other = np.array([video_metrics[name][v][metric] for v in videos])
            difference = other - matched
            boot = difference[indices].mean(1)
            # 两个指标、各 7 个预先声明的比较，共同使用 Bonferroni 区间。
            q = .05 / (2 * len(names)) / 2
            interval = np.quantile(boot, [q, 1 - q]).tolist()
            report[metric][name] = {"other_minus_matched": float(difference.mean()),
                                   "relative_reduction": float(difference.mean() / other.mean()),
                                   "familywise_95ci": interval, "videos_matched_better": int((difference > 0).sum())}
    return report


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    labels = [name.replace("_", " ") for name in CONDITIONS]
    for ax, key, title in zip(axes, ("video_mse", "motion_video_mse"), ("All local RGB targets", "Motion regions")):
        values = [report["conditions"][name][key] for name in CONDITIONS]
        ax.bar(range(len(values)), values, color=["#2563eb"] + ["#94a3b8"] * (len(values) - 1))
        ax.set_xticks(range(len(values)), labels, rotation=35, ha="right")
        ax.set_ylabel("Normalized local RGB MSE (equal video weight)")
        ax.set_title(title)
    fig.suptitle("Frozen Wan DiT block 15: separately fitted shared linear readouts")
    fig.savefig(output / "local_readout.png", dpi=180)
    fig.savefig(output / "local_readout.pdf")
    plt.close(fig)


def run(args):
    data = torch.load(args.output / "prepared.pt", mmap=True, map_location="cpu", weights_only=False)
    for values in data.values():
        values["target"] = values["target"][:, :14]
        values["motion"] = values["motion"][:, :14]
    fit_windows = tuning_split(data["train"]["metadata"], args.seed)
    train_y = data["train"]["target"].flatten(0, 3).to(args.device)
    valid_y = data["valid"]["target"].flatten(0, 3).to(args.device)
    shape = data["valid"]["target"].shape[:-1]
    points = int(np.prod(shape[1:]))
    fit = fit_windows.repeat_interleave(points).to(args.device)
    # 归一化统计只来自最终训练集；内部调参另用 fit 子集自己的统计。
    y_mean, y_std = train_y.mean(0), train_y.std(0).clamp_min(1e-5)
    tune_mean, tune_std = train_y[fit].mean(0), train_y[fit].std(0).clamp_min(1e-5)
    threshold = max(float(torch.quantile(data["train"]["motion"].flatten(), .75)), 1 / 255)
    motion = data["valid"]["motion"] >= threshold
    hidden = {s: F.layer_norm(d["hidden"].to(args.device).float(), (d["hidden"].shape[-1],)) for s, d in data.items()}
    report = {"target": "RGB [0,1], group-averaged 4 frames, independent local area averages",
              "future_groups": list(range(3, 17)), "patch_size": args.patch_size,
              "spatial_controls": "source column = target column xor 1/4 (absolute distance exactly 1/4 cells)",
              "temporal_controls": "source t = target t + 1/4; no wrapping; targets t=3..16",
              "selected_block": 15, "seed": args.seed, "motion_threshold": threshold,
              "normalization": "fixed per-output training mean/std; internal tuning uses fit-only statistics",
              "metric_weighting": "equal source-video weight; sigma/time summaries descriptive",
              "ridge_grid": args.ridges, "train_windows": len(data["train"]["metadata"]),
              "valid_windows": len(data["valid"]["metadata"]), "tuning_fit_windows": int(fit_windows.sum()),
              "source_manifest_sha256": {s: d["manifest_sha256"] for s, d in data.items()},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "conditions": {}}
    video_metrics, rows = {}, []
    for name in CONDITIONS:
        print(f"[fit] {name}", flush=True)
        x = features(hidden["train"], data["train"]["metadata"], name)
        gram, rhs, mean, std = normal_equations(x[fit], (train_y[fit] - tune_mean) / tune_std)
        tuning = []
        for ridge in args.ridges:
            weight = solve(gram, rhs, ridge)
            estimate = predict(x[~fit], weight, mean, std)
            error = float((estimate - (train_y[~fit] - tune_mean) / tune_std).square().mean())
            tuning.append({"ridge": ridge, "mse": error})
        ridge = min(tuning, key=lambda v: v["mse"])["ridge"]
        del gram, rhs
        gram, rhs, mean, std = normal_equations(x, (train_y - y_mean) / y_std)
        weight = solve(gram, rhs, ridge)
        train_error = float((predict(x, weight, mean, std) - (train_y - y_mean) / y_std).square().mean())
        del x, gram, rhs
        vx = features(hidden["valid"], data["valid"]["metadata"], name)
        estimate = predict(vx, weight, mean, std)
        error = (estimate - (valid_y - y_mean) / y_std).square().mean(-1).reshape(shape).cpu()
        rgb_error = ((estimate * y_std + y_mean - valid_y).square().mean(-1).reshape(shape)).cpu()
        assert torch.isfinite(error).all()
        video_metrics[name] = aggregate(error, motion, data["valid"]["metadata"])
        overall = np.mean([v["mse"] for v in video_metrics[name].values()])
        moving = np.mean([v["motion_mse"] for v in video_metrics[name].values()])
        report["conditions"][name] = {"ridge": ridge, "tuning": tuning, "train_mse": train_error,
                                      "video_mse": float(overall), "motion_video_mse": float(moving),
                                      "rgb_mse": float(rgb_error.mean()), "per_time_mse": error.mean((0, 2, 3)).tolist()}
        for i, m in enumerate(data["valid"]["metadata"]):
            rows.append({"condition": name, **m, "mse": float(error[i].mean()),
                         "motion_mse": float(error[i][motion[i]].mean()) if motion[i].any() else None})
        torch.save({"condition": name, "weight": weight.cpu(), "mean": mean.cpu(), "std": std.cpu(),
                    "target_mean": y_mean.cpu(), "target_std": y_std.cpu(), "ridge": ridge}, args.output / f"readout_{name}.pt")
        print(f"[result] {name} ridge={ridge:g} mse={overall:.6f} motion={moving:.6f}", flush=True)
        del vx, estimate
    report["video_metrics"] = video_metrics
    report["paired_comparisons"] = paired_comparisons(video_metrics, args.bootstrap, args.seed)
    report["bootstrap_draws"] = args.bootstrap
    for name in CONDITIONS:
        selected = [r for r in rows if r["condition"] == name]
        report["conditions"][name]["sigma_bins"] = {}
        for b in range(3):
            group = [r for r in selected if min(int(r["sigma"] * 3), 2) == b]
            by_video = defaultdict(list)
            for row in group:
                by_video[row["source_video"]].append(row)
            if not group:
                continue
            means = [np.mean([r["mse"] for r in values]) for values in by_video.values()]
            moving = [np.mean([r["motion_mse"] for r in values if r["motion_mse"] is not None])
                      for values in by_video.values() if any(r["motion_mse"] is not None for r in values)]
            report["conditions"][name]["sigma_bins"][str(b)] = {
                "windows": len(group), "videos": len(by_video), "mse": float(np.mean(means)),
                "motion_videos": len(moving), "motion_mse": float(np.mean(moving)) if moving else None}
        report["conditions"][name]["tasks"] = {
            task: {"videos": len(group), "mse": float(np.mean([v["mse"] for v in group])),
                   "motion_mse": float(np.mean([v["motion_mse"] for v in group]))}
            for task in sorted({v["task"] for v in video_metrics[name].values()})
            if (group := [v for v in video_metrics[name].values() if v["task"] == task])}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.output / "window_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    plot_report(report, args.output)
    print("[complete]", args.output / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--ridges", type=float, nargs="+", default=[.001, .01, .1, 1., 10., 100.])
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    with torch.no_grad():
        (prepare if args.phase == "prepare" else run)(args)


if __name__ == "__main__":
    main()
