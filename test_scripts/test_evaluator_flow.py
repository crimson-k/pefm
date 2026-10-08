"""CPU checks for the fixed visual target branch and old Teacher extraction."""

import sys
from unittest.mock import patch

import tempfile
import unittest
from pathlib import Path
import torch
from omegaconf import OmegaConf

from pefm.export.frozen import (LEGACY_FORMAT, FORMAT, export_frozen_evaluator,
                                load_frozen_visual_teacher, teacher_contract)
from pefm.models import VisualTeacher, VJEPAObservationAdapter, TokenAggregator, SpatiallyAlignedAdapter
from pefm.models.vjepa_adapter import align_row_major_tokens, row_major_token_grid
from src.vjepa2.src.models.vision_transformer import vit_tiny


def tiny_teacher():
    return VisualTeacher(VJEPAObservationAdapter(vit_tiny(img_size=(32, 32), num_frames=2,
                                                         use_rope=True)),
                         TokenAggregator(192, 8, num_heads=2), rgb_grid=(2, 2), aligned_grid=(1, 1))


def test_visual_targets_do_not_read_future_frames():
    torch.set_num_threads(1)
    teacher = tiny_teacher().requires_grad_(False).eval()
    rgb = torch.randn(1, 1, 3, 81, 32, 32)
    groups = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(1, 21).repeat_interleave(4)))
    with torch.no_grad():
        first = teacher({"rgb": rgb, "group_ids": groups})
        changed = rgb.clone()
        changed[:, :, :, 41:] = torch.randn_like(changed[:, :, :, 41:]) * 5
        second = teacher({"rgb": changed, "group_ids": groups})
        prefix = teacher({"rgb": rgb[:, :, :, :41], "group_ids": groups[:41]})
    assert first["embed"].shape == (1, 21, 8)
    torch.testing.assert_close(first["embed"][:, :11], second["embed"][:, :11])
    torch.testing.assert_close(first["embed"][:, :11], prefix["embed"])
    assert not hasattr(teacher, "rssm")
    assert not any(name.startswith("src.r2dreamer") for name in sys.modules)
    coordinates = torch.arange(1200, dtype=torch.float32).reshape(1, 1, 1200, 1)
    assert row_major_token_grid(coordinates)[0, 0, 0, 1, 0, 0] == 40
    assert align_row_major_tokens(coordinates)[0, 0, 0, 0, 0, 0] == 20.5


def test_old_teacher_loads_visual_weights_only(tmp_path):
    torch.set_num_threads(1)
    original = tiny_teacher().eval()
    cfg = OmegaConf.create({"device": "cpu"})
    contract = teacher_contract(cfg)
    contract.update(contract_version=2, state_space="shared_rssm_v1",
                    categorical_shape=[32, 16], rssm={"ignored": True})
    state = original.state_dict()
    state.update({"rssm.unused": torch.randn(3), "prior_prediction_head.unused": torch.randn(3),
                  "posterior_reconstruction_head.unused": torch.randn(3)})
    bundle = {"format": LEGACY_FORMAT, "model": state,
              "config": OmegaConf.to_container(cfg), "semantic_teacher": contract,
              "epoch": 10, "seed": 42}
    old_path, new_path = tmp_path / "old.pt", tmp_path / "visual.pt"
    torch.save(bundle, old_path)
    # Production factory still builds the original V-JEPA architecture. Use a
    # small real encoder here to verify filtering, strict loading and freezing.
    with patch("pefm.export.frozen.build_visual_teacher", side_effect=lambda cfg: tiny_teacher()):
        loaded = load_frozen_visual_teacher(old_path, "cpu")
        export_frozen_evaluator(old_path, new_path)
        converted = load_frozen_visual_teacher(new_path, "cpu")
        for teacher in (loaded, converted):
            assert not hasattr(teacher, "rssm")
            assert all(not p.requires_grad for p in teacher.parameters())
            for key, value in original.state_dict().items():
                torch.testing.assert_close(teacher.state_dict()[key], value)
        assert torch.load(new_path, weights_only=False)["format"] == FORMAT
        assert all(k.startswith(("vjepa_adapter.", "token_aggregator."))
                   for k in converted.state_dict())
        del bundle["model"]["token_aggregator.queries"]
        torch.save(bundle, old_path)
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "Missing key"):
            load_frozen_visual_teacher(old_path, "cpu")


def test_spatial_adapter_requires_explicit_view_layout():
    adapter = SpatiallyAlignedAdapter(dit_dim=32, vjepa_dim=8, target_grid=(15, 20),
                                      num_views=2, hidden_dim=32)
    hidden = torch.randn(1, 21 * 60 * 40, 32)
    tokens = adapter(hidden, dit_grid=(21, 60, 40), sigma=torch.tensor([0.5]))
    assert tokens.shape == (1, 21, 2, 15, 20, 8)
    assert not hasattr(adapter, "aggregator")
    with unittest.TestCase().assertRaisesRegex(ValueError, "not divisible"):
        adapter(hidden, dit_grid=(21, 60, 40), sigma=torch.tensor([0.5]), num_views=7)


class VisualTeacherTest(unittest.TestCase):
    def test_causal_targets(self):
        test_visual_targets_do_not_read_future_frames()

    def test_legacy_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            test_old_teacher_loads_visual_weights_only(Path(directory))

    def test_spatial_layout(self):
        test_spatial_adapter_requires_explicit_view_layout()


if __name__ == "__main__":
    unittest.main()
