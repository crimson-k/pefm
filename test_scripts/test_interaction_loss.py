"""GT layout, weighted classification losses, masks and gradient checks."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from pefm.data import EvaluatorDataset
from pefm.data.dataloader import merge_object_labels
from pefm.models import build_evaluator


@pytest.fixture(scope="module")
def model():
    base = OmegaConf.load("src/r2dreamer/configs/model/_base_.yaml")
    size = OmegaConf.load("src/r2dreamer/configs/model/size12M.yaml")
    cfg = OmegaConf.create({"device": "cpu", "model": OmegaConf.merge(base, size)})
    cfg.model.rssm.initial = "zeros"
    return build_evaluator(cfg)


def zero_logits():
    return {
        f"{branch}_{name}_logits": torch.zeros(1, 4, dim, requires_grad=True)
        for branch in ("posterior", "prior")
        for name, dim in (("relation", 3), ("event", 8), ("motion", 5))
    }


def labels():
    target = torch.zeros(1, 4, 16)
    target[..., 0] = 1
    target[..., 3] = 1
    return target


def test_weighted_bce_for_all_heads_and_branch_sum(model):
    output = zero_logits()
    target = labels().requires_grad_()
    losses = model.compute_interaction_loss(
        output, torch.tensor([[True, False, False, False]]), target,
        weights={"relation": 2.0, "event": 3.0, "motion": 5.0},
    )
    torch.testing.assert_close(losses["relation"], torch.tensor(2 * math.log(2)))
    torch.testing.assert_close(losses["event"], torch.tensor(2 * math.log(2)))
    torch.testing.assert_close(losses["motion"], torch.tensor(2 * math.log(2)))
    torch.testing.assert_close(losses["interaction"], torch.tensor(20 * math.log(2)))
    losses["interaction"].backward()
    assert target.grad is None
    for logits in output.values():
        assert logits.grad[:, :1].abs().sum() == 0
        assert logits.grad[:, 1:].abs().sum() > 0


@pytest.mark.parametrize("event_index", range(8))
def test_all_eight_events_use_their_own_label_and_mask(model, event_index):
    output = zero_logits()
    target = labels()
    target[..., 3:11] = 0
    target[:, 1, 3 + event_index] = 1
    target[:, 1, 11] = 1
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, 1, 3 + event_index] = True
    mask[:, 1, 11] = True
    losses = model.compute_interaction_loss(
        output, torch.zeros(1, 4, dtype=torch.bool), target, mask,
    )
    losses["interaction"].backward()
    for branch in ("posterior", "prior"):
        event_grad = output[f"{branch}_event_logits"].grad
        motion_grad = output[f"{branch}_motion_logits"].grad
        assert event_grad[:, 1, event_index].item() < 0
        assert event_grad.count_nonzero() == 1
        assert motion_grad[:, 1, 0].item() < 0
        assert motion_grad.count_nonzero() == 1


def test_old_four_event_labels_are_rejected(model):
    with pytest.raises(ValueError, match="Expected interaction_target"):
        model.compute_interaction_loss(
            zero_logits(), torch.zeros(1, 4, dtype=torch.bool), torch.zeros(1, 4, 13),
        )


@pytest.mark.parametrize("positive_indices", [[], [0, 7]])
def test_single_arm_rejects_zero_or_multiple_known_events(model, positive_indices):
    target = labels()
    target[:, 1, 3:11] = 0
    target[:, 1, [3 + index for index in positive_indices]] = 1
    with pytest.raises(ValueError, match="Single-arm event targets"):
        model.compute_interaction_loss(
            zero_logits(), torch.zeros(1, 4, dtype=torch.bool), target,
        )


def test_partially_masked_single_arm_still_rejects_multiple_known_events(model):
    target = labels()
    target[:, 1, 10] = 1
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 1, 4] = False
    with pytest.raises(ValueError, match="Single-arm event targets"):
        model.compute_interaction_loss(
            zero_logits(), torch.zeros(1, 4, dtype=torch.bool), target, mask,
        )


def test_explicit_dual_arm_steps_allow_multiple_events_in_a_mixed_batch(model):
    target = labels().expand(2, -1, -1).clone()
    target[1, 1, 10] = 1
    output = {key: value.detach().expand(2, -1, -1).clone() for key, value in zero_logits().items()}
    reset = torch.zeros(2, 4, dtype=torch.bool)
    dual_arm = torch.zeros_like(reset)
    dual_arm[1, 1] = True
    losses = model.compute_interaction_loss(output, reset, target, event_is_dual_arm=dual_arm)
    assert torch.isfinite(losses["interaction"])
    dual_arm[1, 1] = False
    with pytest.raises(ValueError, match="Single-arm event targets"):
        model.compute_interaction_loss(output, reset, target, event_is_dual_arm=dual_arm)


def test_dual_arm_flags_must_match_batch_time_shape(model):
    with pytest.raises(ValueError, match="event_is_dual_arm must have shape"):
        model.compute_interaction_loss(
            zero_logits(), torch.zeros(1, 4, dtype=torch.bool), labels(),
            event_is_dual_arm=torch.ones(1, dtype=torch.bool),
        )


def test_two_objects_or_labels_and_compute_each_head_loss_once(model):
    output = zero_logits()
    first = labels()
    second = labels()
    second[..., :3] = torch.tensor([0, 0, 1])
    second[..., 3:11] = torch.nn.functional.one_hot(torch.full((1, 4), 7), 8)
    second[..., 15] = 1
    reset = torch.tensor([[True, False, False, False]])
    target = torch.stack((first, second), dim=2)
    combined = model.compute_interaction_loss(output, reset, target)
    union = torch.maximum(first, second)
    merged = model.compute_interaction_loss(
        output, reset, union, interaction_is_multi_object=torch.ones_like(reset),
    )
    for name in combined:
        torch.testing.assert_close(combined[name], merged[name])
    actual_grads = torch.autograd.grad(combined["interaction"], tuple(output.values()), retain_graph=True)
    expected_grads = torch.autograd.grad(merged["interaction"], tuple(output.values()))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)
    # 两物体的不同正例都应推动同一预测变大，不能被另一物体的 0 抵消。
    for branch in (0, 1):
        assert (actual_grads[branch * 3][:, 1:, [0, 2]] < 0).all()
        assert (actual_grads[branch * 3 + 1][:, 1:, [0, 7]] < 0).all()


def test_duplicate_positive_stays_one_and_does_not_double_loss_or_gradient(model):
    output = zero_logits()
    reset = torch.zeros(1, 4, dtype=torch.bool)
    target = torch.stack((labels(), labels()), dim=2)
    union, mask = merge_object_labels(target, torch.ones_like(target, dtype=torch.bool))
    torch.testing.assert_close(union, labels())
    assert mask.all() and union.max() == 1
    combined = model.compute_interaction_loss(output, reset, target)
    single = model.compute_interaction_loss(output, reset, labels())
    for name in combined:
        torch.testing.assert_close(combined[name], single[name])
    actual_grads = torch.autograd.grad(combined["interaction"], tuple(output.values()), retain_graph=True)
    expected_grads = torch.autograd.grad(single["interaction"], tuple(output.values()))
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)


def test_or_mask_retains_known_positive_and_requires_all_objects_for_negative():
    target = torch.tensor([[[1.0, 0, 0, 1, 0], [float("nan"), 0, 1, 1, float("nan")]]])
    mask = torch.tensor([[[True, True, True, True, True], [False, True, True, True, False]]])
    merged, valid = merge_object_labels(target, mask)
    torch.testing.assert_close(merged, torch.tensor([[1.0, 0, 1, 1, 0]]))
    torch.testing.assert_close(valid, torch.tensor([[True, True, True, True, False]]))


def test_unknown_object_does_not_create_false_negative_gradients(model):
    output = zero_logits()
    target = torch.stack((labels(), torch.full_like(labels(), float("nan"))), dim=2)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, :, 1] = False
    loss = model.compute_interaction_loss(output, torch.zeros(1, 4, dtype=torch.bool), target, mask)
    loss["interaction"].backward()
    for branch in ("posterior", "prior"):
        relation = output[f"{branch}_relation_logits"].grad
        event = output[f"{branch}_event_logits"].grad
        assert (relation[..., 0] < 0).all() and not relation[..., 1:].any()
        assert (event[..., 0] < 0).all() and not event[..., 1:].any()
        assert not output[f"{branch}_motion_logits"].grad.any()


def test_invalid_events_on_reset_steps_are_ignored(model):
    target = labels()
    target[:, 0, 10] = 1
    loss = model.compute_interaction_loss(
        zero_logits(), torch.tensor([[True, False, False, False]]), target,
    )["interaction"]
    assert torch.isfinite(loss)


def test_unknown_labels_are_excluded_and_do_not_create_nan(model):
    output = zero_logits()
    target = labels()
    mask = torch.ones_like(target, dtype=torch.bool)
    target[:, 0] = float("nan")
    target[:, 2] = float("nan")
    mask[:, 2] = False
    target[:, 1, 3] = float("nan")
    mask[:, 1, 3] = False
    loss = model.compute_interaction_loss(
        output, torch.tensor([[True, False, False, False]]), target, mask,
    )["interaction"]
    torch.testing.assert_close(loss, torch.tensor(6 * math.log(2)))
    loss.backward()
    for logits in output.values():
        assert torch.isfinite(logits.grad).all()
        assert logits.grad[:, 0].abs().sum() == 0
        assert logits.grad[:, 2].abs().sum() == 0
    assert output["posterior_event_logits"].grad[:, 1, 0].abs().sum() == 0


def test_fully_masked_batch_has_zero_interaction_loss(model):
    output = zero_logits()
    loss = model.compute_interaction_loss(
        output, torch.zeros(1, 4, dtype=torch.bool), labels(),
        torch.zeros(1, 4, 16, dtype=torch.bool),
    )["interaction"]
    assert loss.item() == 0
    loss.backward()
    assert all(logits.grad.abs().sum() == 0 for logits in output.values())


def test_relation_requires_one_hot_and_unknown_labels_require_mask(model):
    target = labels()
    target[:, 1, 1] = 1
    with pytest.raises(ValueError, match="exactly one positive"):
        model.compute_interaction_loss(zero_logits(), torch.zeros(1, 4, dtype=torch.bool), target)
    target = labels()
    target[:, 1, 3] = -1
    with pytest.raises(ValueError, match="mask unknown"):
        model.compute_interaction_loss(zero_logits(), torch.zeros(1, 4, dtype=torch.bool), target)


@pytest.mark.parametrize("dual_arm", [None, False, True, [False, True]])
def test_dataset_passes_grouped_gt_and_masks(dual_arm):
    target = labels()[0, :2]
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, 3] = False
    sample = {
        "video": torch.zeros(1, 3, 5, 32, 32), "action": torch.zeros(5, 14),
        "interaction_target": target.tolist(), "interaction_mask": mask.tolist(),
    }
    if dual_arm is not None:
        sample["event_is_dual_arm"] = dual_arm
    batch = EvaluatorDataset([sample])[0]
    torch.testing.assert_close(batch["interaction_target"], target)
    torch.testing.assert_close(batch["interaction_mask"], mask)
    expected_dual_arm = torch.tensor(False if dual_arm is None else dual_arm).expand(2)
    torch.testing.assert_close(batch["event_is_dual_arm"], expected_dual_arm)
    assert batch["interaction_target"].shape[0] == batch["action_delta"].shape[0]
    sample["interaction_target"] = torch.zeros(2, 13).tolist()
    with pytest.raises(ValueError, match="Expected grouped"):
        EvaluatorDataset([sample])[0]
    sample["interaction_target"] = target.tolist()
    sample["event_is_dual_arm"] = [False, True, False]
    with pytest.raises(ValueError, match="Expected event_is_dual_arm"):
        EvaluatorDataset([sample])[0]
