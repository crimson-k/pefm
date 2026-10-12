"""检查局部 RGB 的时空索引、对照输入及已知局部信号上的完整读出。"""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve().parents[1] / "pefm/training/verify_local_tokens.py"
spec = importlib.util.spec_from_file_location("local_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class LocalTokenTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_rgb_targets_use_exact_local_cells_and_frame_groups(self):
        t = torch.arange(81).reshape(81, 1, 1) / 200
        y = torch.arange(6).repeat_interleave(4).reshape(1, 24, 1) / 20
        x = torch.arange(8).repeat_interleave(4).reshape(1, 1, 32) / 30
        frames = (t + y + x).unsqueeze(0).expand(3, -1, -1, -1)
        target, motion = probe.rgb_targets((2 * frames - 1)[None, None], (21, 6, 8), 2)
        self.assertEqual(tuple(target.shape), (18, 6, 8, 12))
        torch.testing.assert_close(target[0, 2, 3], torch.full((12,), 10.5 / 200 + 2 / 20 + 3 / 30))
        torch.testing.assert_close(target[-1, 0, 0], torch.full((12,), 78.5 / 200))
        torch.testing.assert_close(motion, torch.full((18, 6, 8), .005))
        changed = frames.clone()
        changed[:, 9:13, :4, :4] += .1
        after, _ = probe.rgb_targets((2 * changed - 1)[None, None], (21, 6, 8), 2)
        difference = after - target
        self.assertEqual(int((difference.abs().sum(-1) > 1e-6).sum()), 1)
        torch.testing.assert_close(difference[0, 0, 0], torch.full((12,), .1))

    def test_controls_use_one_token_and_identical_target_coordinates(self):
        hidden = torch.arange(21 * 6 * 8 * 3).reshape(1, 21, 6, 8, 3).float()
        metadata = [{"sigma": .4}]
        values = {name: probe.features(hidden, metadata, name) for name in probe.CONDITIONS}
        expected = {"matched": (3, 0), "space_dx1": (3, 1), "space_dx4": (3, 4),
                    "time_dt1": (4, 0), "time_dt4": (7, 0)}
        for name, (t, x) in expected.items():
            torch.testing.assert_close(values[name][0, :3], hidden[0, t, 0, x])
        for name, value in values.items():
            self.assertEqual(tuple(value.shape), (14 * 6 * 8, 3 + 48 + 14 + 2))
            torch.testing.assert_close(value[:, 3:], values["matched"][:, 3:])
        self.assertEqual(float(values["coords_sigma"][:, :3].abs().sum()), 0)
        torch.testing.assert_close(values["time_dt4"][-1, :3], hidden[0, 20, 5, 7])

    def test_video_aggregation_does_not_count_windows_as_independent_videos(self):
        errors = torch.tensor([1., 3., 6.]).reshape(3, 1, 1, 1)
        metadata = [{"source_video": v, "task": "test"} for v in ("a", "a", "b")]
        values = probe.aggregate(errors, torch.ones_like(errors, dtype=torch.bool), metadata)
        self.assertEqual(values["a"]["mse"], 2)
        self.assertEqual(values["b"]["mse"], 6)

    def test_full_ridge_pipeline_detects_known_local_information(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            data = {}
            mapping = torch.randn(12, 6)
            for split, count in (("train", 5), ("valid", 2)):
                hidden = torch.randn(count, 21, 6, 8, 12)
                target = F.layer_norm(hidden[:, 3:], (12,)) @ mapping
                data[split] = {"hidden": hidden, "target": target,
                               "motion": torch.rand(count, 18, 6, 8), "manifest_sha256": "synthetic",
                               "metadata": [{"source_video": f"{split}_{i}", "task": "test",
                                             "sigma": .2 + i / 10} for i in range(count)]}
            torch.save(data, output / "prepared.pt")
            args = SimpleNamespace(output=output, seed=1, device="cpu", patch_size=8,
                                   ridges=[.001, .1], bootstrap=2000)
            probe.run(args)
            report = json.loads((output / "report.json").read_text())
            self.assertLess(report["conditions"]["matched"]["video_mse"], 1e-4)
            for name in probe.CONDITIONS[1:]:
                self.assertGreater(report["conditions"][name]["video_mse"], .5)
                self.assertGreater(report["paired_comparisons"]["mse"][name]["familywise_95ci"][0], 0)


if __name__ == "__main__":
    unittest.main()
