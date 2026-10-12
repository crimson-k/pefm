"""阶段 B：固定视觉目标和预测器，先对齐当前特征，再加入未来预测。"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from pefm.export import load_frozen_visual_teacher
from pefm.export.stage2 import export_stage2_adapter, load_predictor, load_stage2_adapter, teacher_identity
from pefm.models import SpatiallyAlignedAdapter, TemporalSemanticDistillation


class HiddenBatchDataset(Dataset):
    def __init__(self, manifest):
        manifest = Path(manifest)
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.paths = [manifest.parent / (r["path"] if isinstance(r, dict) else r) for r in records]
        self.targets = [manifest.parent / r["target_path"]
                        if isinstance(r, dict) and "target_path" in r else None for r in records]
        if not self.paths:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        raw = torch.load(self.paths[index], map_location="cpu", weights_only=False, mmap=True)
        if self.targets[index] is not None:
            target = torch.load(self.targets[index], map_location="cpu", weights_only=False, mmap=True)
            assert raw["sample_index"] == target["sample_index"]
            assert raw["source_video"] == target["source_video"]
            assert raw["frame_indices"] == target["frame_indices"]
            assert torch.equal(raw["teacher_batch"]["action_segments"], target["action_segments"])
            raw["target_embedding"] = target["embed"]
            raw["teacher_sha256"] = target["teacher_sha256"]
            raw["teacher_dtype"] = target["teacher_dtype"]
            # 固定目标已缓存时，无需读取或传输每个样本约 48 MB 的 RGB。
            raw["teacher_batch"].pop("rgb")
        return raw


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return value


def loader(manifest, shuffle=False):
    return DataLoader(HiddenBatchDataset(manifest), batch_size=None, shuffle=shuffle)


def run_adapter_epoch(model, batches, device, pred_weight, optimizer=None,
                      selected_block=None, noise_sampling=None):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    for raw in batches:
        batch = _move(raw, device)
        if selected_block is not None and batch["selected_block"] != selected_block:
            raise ValueError("Captured hidden uses a different DiT block")
        if noise_sampling is not None and batch["noise_sampling"] != noise_sampling:
            raise ValueError("Captured noise distribution differs from Phase B/C contract")
        sigma = batch["noise_sigma"]
        if "target_embedding" in batch:
            assert batch["teacher_sha256"] == model.teacher_sha256, "Cached target uses a different Teacher"
            assert batch["teacher_dtype"] == model.teacher_dtype, "Cached target precision differs from Phase B"
        with torch.set_grad_enabled(training):
            result = model(batch["hidden"].detach(), batch["dit_grid"], batch["teacher_batch"],
                           sigma, batch["valid_len"], target_embedding=batch.get("target_embedding"))
            loss = result["align"] + pred_weight * result["pred"]
            if training and float(result["active_fraction"]) > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred = model.predictor
            estimate, target = result["estimated"], result["target"]
            clean_prediction = pred.pairs(target, batch["teacher_batch"]["action_segments"]).squeeze(2)
            wrong_prediction = pred.pairs(estimate, batch["teacher_batch"]["action_segments"].roll(1, 1)).squeeze(2)
            agreement_per_step = pred.distance(result["predicted"], clean_prediction)
            metrics = {
                "align": result["align"], "pred": result["pred"], "total": loss,
                "active_fraction": result["active_fraction"],
                "paired_mse": pred.distance(estimate[:, 3:20], target[:, 3:20]).mean(),
                "shuffled_mse": pred.distance(estimate[:, 3:20], target[:, 3:20].roll(1, 1)).mean(),
                "gt_start_pred_mse": pred.distance(clean_prediction, target[:, 4:21]).mean(),
                "estimated_start_pred_mse": result["pred_per_step"].mean(),
                "prediction_agreement_mse": agreement_per_step.mean(),
                "estimated_persistence_mse": pred.distance(estimate[:, 3:20], target[:, 4:21]).mean(),
                "estimated_wrong_action_mse": pred.distance(wrong_prediction, target[:, 4:21]).mean(),
            }
            # capture 当前每条记录只含一个噪声量，按固定三档分别报告读出误差。
            noise_bin = min(int(float(sigma.float().mean()) * 3), 2)
            scoped = {f"sigma_bin_{noise_bin}_{name}": value for name, value in metrics.items()
                      if name not in ("total", "active_fraction")}
            scoped.update({f"task_{raw['task']}_{name}": value for name, value in metrics.items()
                           if name not in ("total", "active_fraction")})
            for name, value in {**metrics, **scoped}.items():
                total, n = totals.get(name, (0., 0))
                totals[name] = (total + float(value), n + 1)
        count += 1
    if not count:
        raise ValueError("No Adapter batches")
    metrics = {k: value / n for k, (value, n) in totals.items()}
    if metrics["active_fraction"] == 0:
        raise ValueError("No captures in the configured auxiliary sigma interval")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-bundle", "predictor-bundle", "train-manifest", "valid-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dit-dim", type=int, required=True)
    parser.add_argument("--selected-block", type=int, required=True)
    parser.add_argument("--noise-sampling", type=json.loads, required=True)
    parser.add_argument("--sigma-max", type=float, default=1.)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--align-epochs", type=int, default=2)
    parser.add_argument("--pred-weight", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--init-adapter", help="Warm-start Adapter weights; uses a fresh optimizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if not 0 < args.align_epochs < args.epochs or not 0 < args.sigma_max <= 1:
        parser.error("Require 0 < align-epochs < epochs and 0 < sigma-max <= 1")
    torch.manual_seed(args.seed)
    identity = teacher_identity(args.teacher_bundle)
    predictor_bundle = torch.load(args.predictor_bundle, map_location="cpu", weights_only=False)
    if predictor_bundle["teacher_sha256"] != identity:
        raise ValueError("Predictor was fitted to a different frozen Teacher")
    assert predictor_bundle["teacher_dtype"] == args.teacher_dtype
    teacher = load_frozen_visual_teacher(args.teacher_bundle, device=args.device, dtype=getattr(torch, args.teacher_dtype))
    token_dim = teacher.token_aggregator.attn.kdim
    start_epoch = 0
    if args.init_adapter:
        initial = torch.load(args.init_adapter, map_location="cpu", weights_only=False)
        assert initial["teacher_sha256"] == identity
        assert initial["sigma_max"] == args.sigma_max
        assert all(torch.equal(value, predictor_bundle["model"][key])
                   for key, value in initial["predictor"]["model"].items())
        adapter = load_stage2_adapter(args.init_adapter, device=args.device, dit_dim=args.dit_dim,
                                      selected_block=args.selected_block, noise_sampling=args.noise_sampling)
        assert adapter.target_grid == teacher.aligned_grid
        start_epoch = int(initial["epoch"])
    else:
        adapter = SpatiallyAlignedAdapter(args.dit_dim, token_dim, teacher.aligned_grid).to(args.device)
    model = TemporalSemanticDistillation(teacher, adapter, load_predictor(predictor_bundle, args.device),
                                         args.sigma_max)
    model.teacher_sha256 = identity
    model.teacher_dtype = args.teacher_dtype
    del teacher
    train_batches, valid_batches = loader(args.train_manifest, True), loader(args.valid_manifest)
    if all(path is not None for batches in (train_batches, valid_batches) for path in batches.dataset.targets):
        model.visual.cpu()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundle_args = dict(predictor_bundle=predictor_bundle, selected_block=args.selected_block,
                       teacher_sha256=identity, sigma_max=args.sigma_max,
                       noise_sampling=args.noise_sampling, init_adapter=args.init_adapter, lr=args.lr)
    export_stage2_adapter(output / "adapter_initial.pt", adapter, **bundle_args, epoch=start_epoch)
    best = float("inf")
    if args.init_adapter:
        # warm-start 时重新验证当前数据与预测损失权重下的起始模型。
        valid = run_adapter_epoch(model, valid_batches, args.device, args.pred_weight,
                                  selected_block=args.selected_block, noise_sampling=args.noise_sampling)
        best = valid["total"]
        initial_metrics = {"epoch": start_epoch, "valid": valid}
        print(json.dumps({"initial_evaluation": initial_metrics}), flush=True)
        export_stage2_adapter(output / "adapter_best.pt", adapter, **bundle_args,
                              epoch=start_epoch, metrics=initial_metrics)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        weight = 0. if epoch <= args.align_epochs else args.pred_weight
        train = run_adapter_epoch(model, train_batches, args.device, weight,
                                  optimizer, args.selected_block, args.noise_sampling)
        valid = run_adapter_epoch(model, valid_batches, args.device, args.pred_weight,
                                  selected_block=args.selected_block, noise_sampling=args.noise_sampling)
        record = {"epoch": epoch, "pred_weight": weight, "train": train, "valid": valid}
        print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if epoch > args.align_epochs and valid["total"] < best:
            best = valid["total"]
            export_stage2_adapter(output / "adapter_best.pt", adapter, **bundle_args,
                                  epoch=epoch, metrics=record)
        export_stage2_adapter(output / "adapter_latest.pt", adapter, **bundle_args,
                              epoch=epoch, metrics=record,
                              optimizer=optimizer.state_dict(), config=vars(args))


if __name__ == "__main__":
    main()
