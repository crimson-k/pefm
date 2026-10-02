"""CPU 验证：动作时序、冻结梯度、统计量与 checkpoint 一致性。"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from pefm.data import stage2_teacher_batch
from pefm.export.stage2 import export_predictor, export_stage2_adapter, load_predictor, load_stage2_adapter
from pefm.models import ActionPredictor, SpatiallyAlignedAdapter, TemporalSemanticDistillation, TokenAggregator
from pefm.training.train_predictor import fit_statistics, targets_at_horizon
from pefm.training.train_adapter import run_adapter_epoch


class TinyVisual(nn.Module):
    def forward(self, rgb, group_ids):
        return rgb


def make_model():
    teacher = SimpleNamespace(vjepa_adapter=TinyVisual(),
                              token_aggregator=TokenAggregator(4, 6, num_heads=2),
                              rgb_grid=(2, 2), aligned_grid=(2, 2))
    return TemporalSemanticDistillation(teacher, SpatiallyAlignedAdapter(5, 4, (2, 2)),
                                        ActionPredictor(6, 8))


class SemanticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def batch(self):
        return {"rgb": torch.randn(2, 21, 4, 4), "group_ids": None,
                "action_segments": torch.randn(2, 21, 4, 14)}

    def test_frame_actions_preserve_four_frame_order(self):
        eef = torch.zeros(1, 81, 14)
        eef[0, :, 0] = torch.arange(81).square()
        result = stage2_teacher_batch(torch.zeros(1, 1, 3, 81, 2, 2), eef)
        torch.testing.assert_close(result["action_segments"][0, 4, :, 0], torch.tensor([25., 27., 29., 31.]))
        self.assertEqual(tuple(result["action_segments"].shape), (1, 21, 4, 14))
        with self.assertRaises(ValueError):
            stage2_teacher_batch(torch.zeros(1, 1, 3, 81, 2, 2), eef[:, ::4])

    def test_pairs_and_open_loop(self):
        model = ActionPredictor(6, 8)
        embed, actions = torch.randn(2, 21, 6), torch.randn(2, 21, 4, 14)
        first = model.pairs(embed, actions)
        self.assertEqual(tuple(first.shape), (2, 17, 1, 6))
        torch.testing.assert_close(first[:, 0], model(embed[:, 3], actions[:, 4:5]))
        rollout = model.pairs(embed, actions, 4)[:, 0]
        changed = embed.clone()
        changed[:, 4:] += 100
        torch.testing.assert_close(rollout, model.pairs(changed, actions, 4)[:, 0])
        self.assertFalse(torch.allclose(rollout, model.pairs(embed, actions.flip(2), 4)[:, 0]))
        no_action = ActionPredictor(6, 8, use_actions=False)
        torch.testing.assert_close(no_action.pairs(embed, actions), no_action.pairs(embed, actions * 20))

    def test_each_loss_reaches_only_current_groups(self):
        for term in ("align", "pred"):
            model = make_model().freeze_adapter()
            hidden = torch.randn(2, 84, 5, requires_grad=True)
            output = model(hidden, (21, 2, 2), self.batch(), torch.tensor([.2]), 84)
            self.assertEqual(tuple(output[term + "_per_step"].shape), (2, 17))
            output[term].backward()
            grad = hidden.grad.reshape(2, 21, -1).norm(dim=-1)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue((grad[:, 3:20] > 0).all())
            self.assertEqual(float(grad[:, :3].sum() + grad[:, 20:].sum()), 0.)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertFalse(output["target"].requires_grad)

    def test_adapter_is_group_local_and_sigma_conditioned(self):
        model = make_model()
        hidden = torch.randn(1, 84, 5)
        first = model.adapter(hidden, (21, 2, 2), torch.tensor([.1]), 84)
        changed = hidden.clone().reshape(1, 21, 4, 5)
        changed[:, 10] += torch.randn(4, 5)
        second = model.adapter(changed.flatten(1, 2), (21, 2, 2), torch.tensor([.1]), 84)
        torch.testing.assert_close(first[:, :10], second[:, :10])
        torch.testing.assert_close(first[:, 11:], second[:, 11:])
        noisy = model.adapter(hidden, (21, 2, 2), torch.tensor([.8]), 84)
        self.assertFalse(torch.allclose(first, noisy))

    def test_phase_b_only_updates_adapter_and_masks_noise(self):
        model = make_model()
        batch = {"hidden": torch.randn(2, 84, 5), "dit_grid": (21, 2, 2), "valid_len": 84,
                 "teacher_batch": self.batch(), "noise_sigma": torch.tensor([.2])}
        frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
        before = [p.clone() for p in model.adapter.parameters()]
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=.01)
        metrics = run_adapter_epoch(model, [batch], "cpu", 1., optimizer)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.adapter.parameters())))
        self.assertTrue(all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen))
        self.assertIn("gt_start_pred_mse", metrics)
        model.sigma_max = .1
        with self.assertRaisesRegex(ValueError, "No captures"):
            run_adapter_epoch(model, [batch], "cpu", 1.)

    def test_predictor_training_and_checkpoint_roundtrip(self):
        model = ActionPredictor(6, 8)
        embed = torch.randn(4, 21, 6)
        stats = fit_statistics(model, [{"embed": embed}])
        self.assertGreater(stats["variance_mean"], 0)
        actions = torch.randn(4, 21, 4, 14)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        target = targets_at_horizon(embed, 1)
        initial = float(model.distance(model.pairs(embed, actions), target).mean().detach())
        for _ in range(25):
            loss = model.distance(model.pairs(embed, actions), target).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()), initial * .8)
        adapter = SpatiallyAlignedAdapter(5, 4, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            export_predictor(path / "p.pt", model, "teacher")
            bundle = torch.load(path / "p.pt", weights_only=False)
            loaded = load_predictor(bundle)
            torch.testing.assert_close(model.pairs(embed, actions), loaded.pairs(embed, actions))
            export_stage2_adapter(path / "a.pt", adapter, predictor_bundle=bundle,
                                  selected_block=15, teacher_sha256="teacher", sigma_max=.4)
            restored = load_stage2_adapter(path / "a.pt", device="cpu", selected_block=15)
            for a, b in zip(adapter.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)
            with self.assertRaises(ValueError):
                load_stage2_adapter(path / "a.pt", device="cpu", selected_block=14)
            with self.assertRaises(ValueError):
                export_stage2_adapter(path / "bad.pt", adapter, predictor_bundle=bundle,
                                      selected_block=15, teacher_sha256="wrong")


if __name__ == "__main__":
    unittest.main()
