"""Load only V-JEPA and Aggregator weights, including from old Teacher bundles."""

import os
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf

from pefm.models import build_visual_teacher

FORMAT = "pefm-frozen-visual-teacher-v1"
LEGACY_FORMAT = "pefm-frozen-evaluator-v1"
NUM_FRAMES = 81
HISTORY_FRAMES = 9
TIME_DIVISION_FACTOR = 4


def teacher_contract(cfg):
    """The shared, fixed embedding space; no dynamics architecture is required."""
    aggregator = cfg.get("token_aggregator", {})
    return {
        "contract_version": 3,
        "state_space": "fixed_embedding_v1",
        "embed_size": int(aggregator.get("embed_dim", 256)),
        "token_dim": 1408,
        "token_aggregator": {
            "token_dim": 1408,
            "embed_dim": int(aggregator.get("embed_dim", 256)),
            "num_heads": int(aggregator.get("num_heads", 4)),
            "num_queries": int(aggregator.get("num_queries", 4)),
        },
        "rgb_grid": [int(v) for v in cfg.get("rgb_grid", [15, 20])],
        "aligned_grid": [int(v) for v in cfg.get("aligned_grid", cfg.get("rgb_grid", [15, 20]))],
        "num_frames": NUM_FRAMES,
        "history_frames": HISTORY_FRAMES,
        "time_groups": 21,
        "vjepa_encoder_frozen": True,
    }


def check_teacher_contract(checkpoint, cfg):
    contract = checkpoint.get("semantic_teacher")
    if not isinstance(contract, Mapping):
        raise ValueError("Teacher checkpoint is missing the semantic_teacher contract")
    expected = teacher_contract(cfg)
    # Old bundles define the same visual embedding space. Validate that part
    # without requiring their discarded RSSM architecture or constructing it.
    legacy = (contract.get("contract_version") == 2
              and contract.get("state_space") == "shared_rssm_v1")
    for key, value in expected.items():
        if legacy and key in ("contract_version", "state_space"):
            continue
        if contract.get(key) != value:
            raise ValueError(f"Teacher contract mismatch for {key}: "
                             f"checkpoint={contract.get(key)!r}, expected={value!r}")


def visual_state(state):
    """Strictly load the visual submodules while discarding old dynamics heads."""
    result = {k: v for k, v in state.items()
              if k.startswith(("vjepa_adapter.", "token_aggregator."))}
    if not result:
        raise ValueError("Teacher checkpoint has no V-JEPA/Aggregator weights")
    return result


def export_frozen_evaluator(checkpoint_path, bundle_path):
    """Extract a visual-only bundle from a verified Teacher checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint.get("config", {}))
    check_teacher_contract(checkpoint, config)
    bundle = {
        "format": FORMAT, "model": visual_state(checkpoint["model"]),
        "config": {"device": "cpu", **{
            k: v for k, v in teacher_contract(config).items()
            if k in ("rgb_grid", "aligned_grid", "token_aggregator")}},
        "epoch": int(checkpoint["epoch"]), "seed": int(checkpoint["seed"]),
        "semantic_teacher": teacher_contract(config),
    }
    path = Path(bundle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(path) + ".tmp"
    torch.save(bundle, temporary)
    os.replace(temporary, path)
    return path


def load_frozen_visual_teacher(bundle_path, device, dtype=torch.float32):
    """Reuse the trained G; never instantiate or load a dynamics model."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("format") not in (FORMAT, LEGACY_FORMAT):
        raise ValueError(f"Unsupported PEFM Teacher format: {bundle.get('format')!r}")
    cfg = OmegaConf.create(bundle["config"])
    check_teacher_contract(bundle, cfg)
    cfg.device = "cpu"
    teacher = build_visual_teacher(cfg)
    teacher.load_state_dict(visual_state(bundle["model"]), strict=True, assign=True)
    del bundle
    return teacher.to(device=device, dtype=dtype).requires_grad_(False).eval()


# Existing bundle-loading calls now return only the fixed visual target branch.
load_frozen_evaluator = load_frozen_visual_teacher
