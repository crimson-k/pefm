"""Export and load the frozen evaluator consumed by BWM training."""

import os
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pefm.models import build_evaluator

FORMAT = "pefm-frozen-evaluator-v1"
TEACHER_CONTRACT_VERSION = 2
NUM_FRAMES = 81
HISTORY_FRAMES = 9
TIME_DIVISION_FACTOR = 4


def teacher_contract(cfg):
    """Describe the exact Teacher architecture used by Stage 2."""
    rssm = cfg.model.rssm
    rssm_fields = (
        "stoch", "deter", "hidden", "discrete", "img_layers", "obs_layers",
        "dyn_layers", "blocks", "act", "unimix_ratio", "initial",
    )
    return {
        "contract_version": TEACHER_CONTRACT_VERSION,
        "state_space": "shared_rssm_v1",
        "categorical_shape": [int(rssm.stoch), int(rssm.discrete)],
        "embed_size": 256,
        "token_dim": 1408,
        "token_aggregator": {
            "token_dim": 1408,
            "embed_dim": 256,
            "num_heads": 4,
            "num_queries": 4,
        },
        "rssm": {
            key: (float(getattr(rssm, key)) if key == "unimix_ratio"
                  else str(getattr(rssm, key)) if key in ("act", "initial")
                  else int(getattr(rssm, key)))
            for key in rssm_fields
        },
        "num_frames": NUM_FRAMES,
        "history_frames": HISTORY_FRAMES,
        "time_groups": 1 + (NUM_FRAMES - 1) // TIME_DIVISION_FACTOR,
        "vjepa_encoder_frozen": True,
    }


def check_teacher_contract(checkpoint, cfg):
    """Reject missing or incompatible Teacher metadata before loading weights."""
    contract = checkpoint.get("semantic_teacher")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "Teacher checkpoint is missing the semantic_teacher contract; "
            "old or unverified Teacher checkpoints cannot be reused"
        )
    expected = teacher_contract(cfg)
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(
                f"Teacher contract mismatch for {key}: "
                f"checkpoint={contract.get(key)!r}, expected={value!r}"
            )


def export_frozen_evaluator(checkpoint_path, bundle_path):
    """Remove training state and save a self-contained evaluator bundle."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint.get("config", {}))
    check_teacher_contract(checkpoint, config)
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
    check_teacher_contract(bundle, cfg)
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
