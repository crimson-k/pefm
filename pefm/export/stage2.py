"""GRU predictor 与带噪声条件的 Adapter checkpoint。"""

import hashlib
from pathlib import Path

import torch

from pefm.models import ActionPredictor, SpatiallyAlignedAdapter, TemporalSemanticDistillation
from .frozen import load_frozen_visual_teacher

FORMAT = "pefm-embedding-adapter-v1"
PREDICTOR_FORMAT = "pefm-embedding-predictor-v1"


def teacher_identity(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)
    return path


def _state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def export_predictor(path, predictor, teacher_sha256, **metadata):
    return _write_bundle(path, dict(
        format=PREDICTOR_FORMAT, model=_state(predictor),
        embed_dim=predictor.embed_dim, hidden_dim=predictor.hidden_dim,
        use_actions=predictor.use_actions, teacher_sha256=teacher_sha256,
        **metadata,
    ))


def load_predictor(bundle, device="cpu"):
    if isinstance(bundle, (str, Path)):
        bundle = torch.load(bundle, map_location="cpu", weights_only=False)
    if bundle["format"] != PREDICTOR_FORMAT:
        raise ValueError("Expected an embedding GRU predictor checkpoint")
    predictor = ActionPredictor(bundle["embed_dim"], bundle["hidden_dim"], bundle["use_actions"])
    predictor.load_state_dict(bundle["model"])
    return predictor.to(device)


def export_stage2_adapter(path, adapter, *, predictor_bundle, selected_block,
                          teacher_sha256, sigma_max=1.0, noise_sampling=None, **metadata):
    if predictor_bundle["teacher_sha256"] != teacher_sha256:
        raise ValueError("Predictor and Adapter must use the same frozen Teacher")
    return _write_bundle(path, dict(
        format=FORMAT, model=_state(adapter), dit_dim=adapter.dit_dim,
        vjepa_dim=adapter.vjepa_dim, hidden_dim=adapter.hidden_dim,
        target_grid=adapter.target_grid, selected_block=selected_block,
        predictor=predictor_bundle, teacher_sha256=teacher_sha256,
        sigma_max=float(sigma_max), noise_sampling=noise_sampling, **metadata,
    ))


def load_stage2_adapter(path, *, device, dtype=None, dit_dim=None, selected_block=None,
                        noise_sampling=None, freeze=False):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle["format"] != FORMAT:
        raise ValueError("Expected an embedding Adapter checkpoint; RSSM Adapters cannot be reused")
    for key, expected in (("dit_dim", dit_dim), ("selected_block", selected_block),
                          ("noise_sampling", noise_sampling)):
        if expected is not None and bundle[key] != expected:
            raise ValueError(f"Adapter contract mismatch: {key}")
    adapter = SpatiallyAlignedAdapter(bundle["dit_dim"], bundle["vjepa_dim"],
                                      bundle["target_grid"], hidden_dim=bundle["hidden_dim"])
    adapter.load_state_dict(bundle["model"])
    adapter.to(device=device, dtype=dtype)
    if freeze:
        adapter.requires_grad_(False).eval()
    return adapter


def load_stage2_distiller(teacher_bundle, adapter_bundle, device, dtype, *,
                          dit_dim=None, selected_block=None, noise_sampling=None):
    bundle = torch.load(adapter_bundle, map_location="cpu", weights_only=False)
    if bundle["teacher_sha256"] != teacher_identity(teacher_bundle):
        raise ValueError("Frozen Teacher differs from the one used to fit the predictor")
    teacher = load_frozen_visual_teacher(teacher_bundle, device=device, dtype=dtype)
    adapter = load_stage2_adapter(adapter_bundle, device=device, dtype=dtype,
                                  dit_dim=dit_dim, selected_block=selected_block,
                                  noise_sampling=noise_sampling, freeze=True)
    predictor = load_predictor(bundle["predictor"], device=device)
    return TemporalSemanticDistillation(teacher, adapter, predictor, bundle["sigma_max"]).freeze_adapter()
