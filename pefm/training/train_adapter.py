"""Phase-B training entry point for the PEFM SpatiallyAlignedAdapter.

The manifest points to ``.pt`` files produced by the BWM-side hidden capture.
Each file contains ``hidden``, ``dit_grid``, ``valid_len`` and either a nested
``teacher_batch`` or the four Teacher batch fields directly.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from pefm.export import export_stage2_adapter, load_frozen_evaluator
from pefm.models import SpatiallyAlignedAdapter, TemporalSemanticDistillation


REQUIRED_TEACHER_FIELDS = ("rgb", "group_ids", "action_delta", "reset")


class HiddenBatchDataset(Dataset):
    """Read one serialized hidden/GT batch per manifest line."""

    def __init__(self, manifest):
        manifest = Path(manifest)
        root = manifest.parent
        self.paths = []
        with manifest.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    path = record["path"] if isinstance(record, dict) else record
                    path = Path(path)
                    self.paths.append(str(path if path.is_absolute() else root / path))
        if not self.paths:
            raise ValueError(f"Empty Stage 2 hidden manifest: {manifest}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        payload = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a dict hidden batch, got {type(payload).__name__}")
        return payload


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _unpack_batch(batch):
    hidden = batch["hidden"]
    teacher_batch = batch.get("teacher_batch")
    if teacher_batch is None:
        teacher_batch = {key: batch[key] for key in REQUIRED_TEACHER_FIELDS}
    if "dit_grid" not in batch or "valid_len" not in batch:
        raise KeyError("Stage 2 hidden batches must provide dit_grid and valid_len explicitly")
    dit_grid = tuple(int(value) for value in batch["dit_grid"])
    valid_len = int(batch["valid_len"])
    num_views = int(batch.get("num_views", 1))
    return hidden, dit_grid, valid_len, num_views, teacher_batch


def _scalar_metrics(result):
    values = {}
    for name, value in result.items():
        if torch.is_tensor(value) and value.ndim == 0:
            values[name] = float(value.detach().cpu())
    return values


def run_adapter_epoch(
    distiller, loader, device, optimizer=None,
    token_scale=1.0, embedding_scale=1.0, posterior_scale=1.0,
):
    """Run one Phase-B epoch and return averaged loss/diagnostic scalars."""
    training = optimizer is not None
    distiller.train(training)
    totals, count = {}, 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        hidden, dit_grid, valid_len, num_views, teacher_batch = _unpack_batch(batch)
        hidden = hidden.detach()  # Phase B keeps Wan DiT outside the optimizer.
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            result = distiller(
                hidden=hidden,
                dit_grid=dit_grid,
                valid_len=valid_len,
                teacher_batch=teacher_batch,
                num_views=num_views,
                phase="adapter",
                token_scale=token_scale,
                embedding_scale=embedding_scale,
                posterior_scale=posterior_scale,
            )
        losses = result["adapter"]
        if training:
            losses["total"].backward()
            optimizer.step()
        for name, value in _scalar_metrics(losses).items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    if not count:
        raise ValueError("Stage 2 Adapter epoch received no batches")
    return {name: value / count for name, value in totals.items()}


def _teacher_contract(bundle_path):
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    contract = bundle.get("semantic_teacher")
    if contract is None:
        config = bundle.get("config", {})
        contract = config.get("semantic_teacher") if isinstance(config, Mapping) else None
    if contract is None:
        raise ValueError("Teacher bundle has no semantic_teacher contract")
    return contract


def train_adapter(
    teacher_bundle,
    train_loader,
    output,
    *,
    dit_dim,
    selected_block,
    device,
    dtype,
    epochs=1,
    lr=1e-4,
    weight_decay=0.01,
    valid_loader=None,
    noise_sampling=None,
    target_grid=(15, 20),
    num_views=1,
    hidden_dim=None,
    token_scale=1.0,
    embedding_scale=1.0,
    posterior_scale=1.0,
):
    """Train and export the best Adapter using a caller-provided DataLoader."""
    if tuple(target_grid) != (15, 20) or int(num_views) != 1:
        raise ValueError("Phase B currently supports only target_grid=(15,20), num_views=1")
    teacher_contract = _teacher_contract(teacher_bundle)
    teacher = load_frozen_evaluator(teacher_bundle, device=device, dtype=dtype)
    adapter = SpatiallyAlignedAdapter(
        dit_dim=dit_dim,
        vjepa_dim=1408,
        target_grid=target_grid,
        num_views=num_views,
        hidden_dim=hidden_dim,
    ).to(device=device, dtype=dtype)
    distiller = TemporalSemanticDistillation(teacher, adapter, history_groups=3)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=lr, weight_decay=weight_decay,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best_score = None
    best_path = output / "best_adapter.pt"
    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_adapter_epoch(
            distiller, train_loader, device, optimizer,
            token_scale=token_scale,
            embedding_scale=embedding_scale,
            posterior_scale=posterior_scale,
        )
        valid_metrics = None
        if valid_loader is not None:
            with torch.no_grad():
                valid_metrics = run_adapter_epoch(
                    distiller, valid_loader, device,
                    token_scale=token_scale,
                    embedding_scale=embedding_scale,
                    posterior_scale=posterior_scale,
                )
        metrics = valid_metrics or train_metrics
        score = metrics["correct_cosine"] - max(
            metrics["shuffled_spatial_cosine"], metrics["shuffled_time_view_cosine"],
        )
        path = output / f"adapter_epoch_{epoch:04d}.pt"
        export_stage2_adapter(
            path,
            adapter,
            selected_block=selected_block,
            teacher_contract=teacher_contract,
            noise_sampling=noise_sampling,
            epoch=epoch,
            metrics={"train": train_metrics, "valid": valid_metrics},
        )
        if best_score is None or score > best_score:
            best_score = score
            export_stage2_adapter(
                best_path,
                adapter,
                selected_block=selected_block,
                teacher_contract=teacher_contract,
                noise_sampling=noise_sampling,
                epoch=epoch,
                metrics={"train": train_metrics, "valid": valid_metrics},
            )
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train the PEFM Stage-2 Adapter.")
    parser.add_argument("--config", default=None, help="YAML config; explicit CLI values override it.")
    parser.add_argument("--teacher-bundle", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--valid-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dit-dim", type=int, default=None)
    parser.add_argument("--selected-block", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--target-grid", type=int, nargs=2, default=None)
    parser.add_argument("--num-views", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--noise-sampling", default=None,
                        help="Optional JSON object overriding the config contract.")
    parser.add_argument("--token-scale", type=float, default=None)
    parser.add_argument("--embedding-scale", type=float, default=None)
    parser.add_argument("--posterior-scale", type=float, default=None)
    args = parser.parse_args()
    config = {}
    if args.config:
        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True) or {}
    values = dict(config)
    values.update({key: value for key, value in vars(args).items()
                   if key != "config" and value is not None})
    defaults = {
        "device": "cuda", "dtype": "bfloat16", "epochs": 1,
        "lr": 1e-4, "weight_decay": 0.01, "target_grid": [15, 20],
        "num_views": 1, "hidden_dim": None, "noise_sampling": None,
        "token_scale": 1.0, "embedding_scale": 1.0, "posterior_scale": 1.0,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    required = ("teacher_bundle", "train_manifest", "output", "dit_dim", "selected_block")
    missing = [key for key in required if values.get(key) is None]
    if missing:
        parser.error("missing required settings: " + ", ".join(missing))
    if int(values.get("history_groups", 3)) != 3:
        parser.error("Stage 2 requires history_groups=3")
    if args.noise_sampling:
        values["noise_sampling"] = json.loads(args.noise_sampling)
    dtype = getattr(torch, values["dtype"])
    train_loader = DataLoader(
        HiddenBatchDataset(values["train_manifest"]), batch_size=1, shuffle=True,
        collate_fn=lambda items: items[0],
    )
    valid_loader = None
    if values.get("valid_manifest"):
        valid_loader = DataLoader(
            HiddenBatchDataset(values["valid_manifest"]), batch_size=1, shuffle=False,
            collate_fn=lambda items: items[0],
        )
    train_adapter(
        values["teacher_bundle"],
        train_loader,
        values["output"],
        dit_dim=int(values["dit_dim"]),
        selected_block=int(values["selected_block"]),
        device=values["device"],
        dtype=dtype,
        epochs=int(values["epochs"]),
        lr=float(values["lr"]),
        weight_decay=float(values["weight_decay"]),
        target_grid=tuple(values["target_grid"]),
        num_views=int(values["num_views"]),
        hidden_dim=values["hidden_dim"],
        noise_sampling=values["noise_sampling"],
        token_scale=float(values["token_scale"]),
        embedding_scale=float(values["embedding_scale"]),
        posterior_scale=float(values["posterior_scale"]),
        valid_loader=valid_loader,
    )


if __name__ == "__main__":
    main()
