"""Checkpoint contracts for the PEFM-side Stage 2 Adapter."""

import os
from collections.abc import Mapping
from pathlib import Path

import torch

from pefm.models import SpatiallyAlignedAdapter, TemporalSemanticDistillation

from .frozen import load_frozen_evaluator


FORMAT = "pefm-spatial-adapter-v1"
TIME_GROUPS = 21
HISTORY_GROUPS = 3
TARGET_GRID = (15, 20)


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(path) + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return path


def _check_teacher_contract(actual, expected):
    if expected is None:
        return
    if not isinstance(actual, Mapping):
        raise ValueError("Adapter bundle has no teacher_contract")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"Teacher contract mismatch for {key}: "
                f"bundle={actual.get(key)!r}, expected={value!r}"
            )


def _check_adapter_contract(bundle, *, dit_dim=None, selected_block=None,
                            teacher_contract=None, noise_sampling=None):
    if not isinstance(bundle, Mapping):
        raise ValueError("Adapter bundle must be a mapping")
    if bundle.get("format") != FORMAT:
        raise ValueError(f"Unsupported Stage 2 Adapter format: {bundle.get('format')!r}")
    if int(bundle.get("time_groups", -1)) != TIME_GROUPS:
        raise ValueError("Adapter bundle must use time_groups=21")
    if int(bundle.get("history_groups", -1)) != HISTORY_GROUPS:
        raise ValueError("Adapter bundle must use history_groups=3")
    if tuple(bundle.get("target_grid", ())) != TARGET_GRID:
        raise ValueError("Adapter bundle must target the V-JEPA 15x20 grid")
    if int(bundle.get("num_views", -1)) != 1:
        raise ValueError("PEFM Stage 2 currently supports num_views=1 only")
    for key in ("dit_dim", "vjepa_dim", "selected_block", "teacher_contract", "noise_sampling"):
        if key not in bundle:
            raise ValueError(f"Adapter bundle is missing contract field: {key}")
    if not isinstance(bundle["teacher_contract"], Mapping):
        raise ValueError("Adapter bundle teacher_contract must be a mapping")
    if not isinstance(bundle.get("model"), Mapping):
        raise ValueError("Adapter bundle is missing model weights")
    if dit_dim is not None and int(bundle.get("dit_dim", -1)) != int(dit_dim):
        raise ValueError(
            f"DiT hidden dim mismatch: bundle={bundle.get('dit_dim')}, expected={dit_dim}"
        )
    if selected_block is not None and int(bundle.get("selected_block", -1)) != int(selected_block):
        raise ValueError(
            f"Selected DiT block mismatch: bundle={bundle.get('selected_block')}, "
            f"expected={selected_block}"
        )
    _check_teacher_contract(bundle.get("teacher_contract"), teacher_contract)
    if noise_sampling is not None and bundle.get("noise_sampling") != _plain(noise_sampling):
        raise ValueError("Adapter noise_sampling contract does not match")


def export_stage2_adapter(
    path,
    adapter,
    *,
    selected_block,
    teacher_contract,
    noise_sampling=None,
    epoch=None,
    metrics=None,
):
    """Save only Adapter weights plus the contracts needed by Phase C."""
    if int(adapter.num_views) != 1 or tuple(adapter.target_grid) != TARGET_GRID:
        raise ValueError("Stage 2 Adapter export requires one view and a 15x20 target grid")
    if not isinstance(teacher_contract, Mapping):
        raise ValueError("teacher_contract must be a mapping")
    state = {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }
    bundle = {
        "format": FORMAT,
        "model": state,
        "dit_dim": int(adapter.dit_dim),
        "vjepa_dim": int(adapter.vjepa_dim),
        "hidden_dim": int(adapter.hidden_dim),
        "target_grid": list(TARGET_GRID),
        "num_views": 1,
        "time_groups": TIME_GROUPS,
        "history_groups": HISTORY_GROUPS,
        "selected_block": int(selected_block),
        "teacher_contract": _plain(teacher_contract),
        "noise_sampling": _plain(noise_sampling),
    }
    if epoch is not None:
        bundle["epoch"] = int(epoch)
    if metrics is not None:
        bundle["metrics"] = _plain(metrics)
    return _write_bundle(path, bundle)


def load_stage2_adapter(
    path,
    *,
    device,
    dtype=None,
    dit_dim=None,
    selected_block=None,
    teacher_contract=None,
    noise_sampling=None,
    freeze=False,
):
    """Load an Adapter after checking all Stage 2 compatibility fields."""
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    _check_adapter_contract(
        bundle,
        dit_dim=dit_dim,
        selected_block=selected_block,
        teacher_contract=teacher_contract,
        noise_sampling=noise_sampling,
    )
    adapter = SpatiallyAlignedAdapter(
        dit_dim=int(bundle["dit_dim"]),
        vjepa_dim=int(bundle["vjepa_dim"]),
        hidden_dim=int(bundle.get("hidden_dim", bundle["vjepa_dim"])),
        target_grid=tuple(bundle["target_grid"]),
        num_views=int(bundle["num_views"]),
    )
    adapter.load_state_dict(bundle["model"], strict=True)
    adapter.to(device=device)
    if dtype is not None:
        adapter.to(dtype=dtype)
    if freeze:
        adapter.requires_grad_(False)
        adapter.eval()
    return adapter


def _teacher_metadata(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    contract = bundle.get("semantic_teacher")
    if contract is None:
        config = bundle.get("config", {})
        contract = config.get("semantic_teacher") if isinstance(config, Mapping) else None
    if contract is None:
        raise ValueError("Teacher bundle has no semantic_teacher contract")
    return contract


def load_stage2_distiller(
    teacher_bundle,
    adapter_bundle,
    device,
    dtype,
    *,
    dit_dim=None,
    selected_block=None,
    noise_sampling=None,
):
    """Load the frozen Teacher and Adapter used by Phase C."""
    teacher_contract = _teacher_metadata(teacher_bundle)
    teacher = load_frozen_evaluator(teacher_bundle, device=device, dtype=dtype)
    adapter = load_stage2_adapter(
        adapter_bundle,
        device=device,
        dtype=dtype,
        dit_dim=dit_dim,
        selected_block=selected_block,
        teacher_contract=teacher_contract,
        noise_sampling=noise_sampling,
        freeze=True,
    )
    distiller = TemporalSemanticDistillation(teacher, adapter, history_groups=HISTORY_GROUPS)
    distiller.freeze_adapter()
    return distiller
