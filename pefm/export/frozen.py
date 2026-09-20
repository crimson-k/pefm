"""Export and load the frozen evaluator consumed by BWM training."""

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pefm.models import build_evaluator

FORMAT = "pefm-frozen-evaluator-v1"


def export_frozen_evaluator(checkpoint_path, bundle_path):
    """Remove training state and save a self-contained evaluator bundle."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bundle = {
        "format": FORMAT,
        "model": checkpoint["model"],
        "config": checkpoint["config"],
        "epoch": int(checkpoint["epoch"]),
        "seed": int(checkpoint["seed"]),
        "semantic_teacher": checkpoint.get("semantic_teacher"),
    }
    path = Path(bundle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(path) + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return path


def load_frozen_evaluator(bundle_path, device, dtype):
    """Load BWM's frozen evaluator while retaining gradients to its RGB input."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("format") != FORMAT:
        raise ValueError(f"Unsupported PEFM bundle format: {bundle.get('format')!r}")

    cfg = OmegaConf.create(bundle["config"])
    contract = bundle.get("semantic_teacher") or cfg.get("semantic_teacher")
    if contract is not None and int(contract.get("time_groups", 0)) != 21:
        raise ValueError(f"Stage 2 requires a 21-group Teacher, got {contract!r}")
    cfg.device = "cpu"
    cfg.model.rssm.device = "cpu"
    evaluator = build_evaluator(cfg)
    state = bundle.pop("model")
    evaluator.load_state_dict(state, strict=True, assign=True)
    del bundle, state
    evaluator.to(device=device, dtype=dtype)
    evaluator.rssm._device = torch.device(device)
    evaluator.requires_grad_(False)
    evaluator.eval()
    return evaluator
