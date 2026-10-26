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
    aggregator = {
        "type": "cross_attention",
        "token_dim": 48,
        "embed_dim": 512,
        "num_heads": 4,
        "num_queries": 4,
    }
    contract = {
        "contract_version": TEACHER_CONTRACT_VERSION,
        "state_space": "shared_rssm_v1",
        "categorical_shape": [int(rssm.stoch), int(rssm.discrete)],
        "embed_size": 512,
        "token_dim": 48,
        "token_aggregator": aggregator,
        "interaction_heads": {
            "hidden": 256, "relation": 3, "event": 8, "motion": 5,
            "shared_prior_posterior": True,
            "object_loss_reduction": "sum",
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
    }
    contract.update(
        observation="wan_vae_clean_latents",
        vae_encoder_frozen=True,
        semantic_target={
            "type": "vjepa_spatial_mean", "dim": int(cfg.get("semantic_dim", 1408)),
            "encoder_frozen": True,
        },
        vae={
            "type": "WanVideoVAE38", "channels": 48,
            "spatial_compression": 16, "temporal_compression": 4,
            "rgb_size": [int(cfg.height), int(cfg.width)],
            "resize_mode": str(cfg.resize_mode),
        },
    )
    if cfg.get("reward_label_reduction") == "masked_or":
        contract["interaction_heads"].update(
            object_label_reduction="masked_or", object_loss_reduction="none",
            relation_loss="bce_with_logits", event_loss="bce_with_logits",
            motion_loss="bce_with_logits",
        )
    return contract


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
    """Load the evaluator while retaining gradients to its observation input."""
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
    action_delta_stats = cfg.get("action_delta_stats")
    evaluator.action_delta_stats = (
        OmegaConf.to_container(action_delta_stats, resolve=True)
        if action_delta_stats is not None else None
    )
    evaluator.requires_grad_(False)
    evaluator.eval()
    return evaluator
