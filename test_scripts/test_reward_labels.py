"""验证 V23 分类映射、原始帧运动、质量 mask 和多物体读取。"""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pefm.data.reward_labels import EVENTS, load_source_manifest, read_reward_labels, robot_motion_labels
from pefm.data import EvaluatorDataset
import torch


def write_sources(tmp_path, targets=("bottle",)):
    ids = np.tile(np.arange(8, dtype=np.uint8)[:, None], (1, len(targets)))
    if len(targets) > 1:
        ids[:, 1] = ids[::-1, 0]
    source = tmp_path / "labels.npz"
    np.savez_compressed(
        source, class_id=ids, one_hot=np.eye(8, dtype=np.uint8)[ids],
        frame_index=np.arange(8), time_seconds=np.arange(8) / 30,
        target_ids=np.array(targets), class_names=np.array(EVENTS), fps=np.array(30.0),
        robot_moving=np.zeros(8, dtype=bool),
        annotation_uncertain=np.zeros_like(ids, dtype=bool), failed=np.zeros_like(ids, dtype=bool),
        uncertain=np.zeros_like(ids, dtype=np.int8), verification_state=np.full(ids.shape, "pass"),
        verification_uncertain=np.zeros_like(ids, dtype=np.int8),
        verification_rejected=np.zeros_like(ids, dtype=np.int8),
    )
    eef = np.zeros((8, 14))
    eef[:, 6] = np.arange(8) * 0.1
    state = tmp_path / "state.parquet"
    pq.write_table(pa.table({"observation.state": eef.tolist(), "frame_index": np.arange(8)}), state)
    return source, state


def test_all_event_classes_map_to_relation_and_object_motion(tmp_path):
    source, state = write_sources(tmp_path)
    result = read_reward_labels(source, state)
    np.testing.assert_array_equal(result["relation"].argmax(-1)[:, 0], [0, 0, 0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(result["event"][:, 0], np.eye(8))
    np.testing.assert_array_equal(result["motion"][:, 0, 4], [0, 1, 1, 0, 1, 0, 1, 0])
    assert result["interaction_target"].shape == (8, 1, 16)
    assert result["interaction_mask"].all()
    assert not result["label_acceptance"]
    assert "control opening" in str(result["gripper_source"])


def test_per_target_and_noncontiguous_frame_selection_preserve_alignment(tmp_path):
    source, state = write_sources(tmp_path, targets=("burger", "fries"))
    result = read_reward_labels(source, state)
    assert result["interaction_target"].shape == (8, 2, 16)
    selected = read_reward_labels(source, state, frame_indices=[5, 1, 5], target_id="fries")
    np.testing.assert_array_equal(selected["interaction_target"], result["interaction_target"][[5, 1, 5], 1])
    np.testing.assert_array_equal(selected["frame_index"], [5, 1, 5])
    with pytest.raises(ValueError, match="Unknown target"):
        read_reward_labels(source, state, target_id="unknown")
    with pytest.raises(ValueError, match="frame_indices"):
        read_reward_labels(source, state, frame_indices=[-1])


def test_unknown_review_is_not_false_and_does_not_mask_robot_state(tmp_path):
    source, state = write_sources(tmp_path)
    with np.load(source) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["uncertain"][1] = -1
    arrays["verification_state"] = arrays["verification_state"].astype("U32")
    arrays["verification_state"][1] = "stale_or_missing_report"
    arrays["annotation_uncertain"][2] = True
    arrays["failed"][3] = True
    arrays["verification_rejected"][4] = 1
    np.savez_compressed(source, **arrays)
    result = read_reward_labels(source, state)
    assert not result["relation_mask"][1:5].any()
    assert not result["event_mask"][1:5].any()
    assert not result["motion_mask"][1:5, :, 4].any()
    assert result["motion_mask"][..., :4].all()
    assert result["uncertain"][1, 0] == -1
    candidates = read_reward_labels(source, state, include_unverified=True)
    assert candidates["event_mask"][1].all()
    assert not candidates["event_mask"][2:5].any()
    assert not candidates["label_acceptance"]


def test_completed_episode_layout_preserves_objects_quality_and_source_relation(tmp_path):
    folder = tmp_path / "episodes" / "train_task_000000"
    folder.mkdir(parents=True)
    source, state = write_sources(folder, targets=("burger", "fries"))
    with np.load(source) as data:
        arrays = {key: data[key] for key in data.files}
    del arrays["verification_state"], arrays["verification_uncertain"]
    arrays["uncertain"] = arrays["uncertain"].astype(bool)
    arrays["uncertain"][1, 0] = True
    arrays["failed"][2, 1] = True
    arrays["verification_rejected"][3, 0] = 1
    arrays["annotation_uncertain"][4, 1] = True
    arrays["relation"] = np.array([0, 0, 0, 0, 1, 1, 2, 2])[arrays["class_id"]]
    arrays["object_moving"] = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=bool)[arrays["class_id"]]
    np.savez_compressed(source, **arrays)
    annotation = folder / "annotations.json"
    annotation.write_text(json.dumps({"source_row": {"video": "videos/task.mp4", "action": "data/task.parquet"}}))
    completed = {
        "status": "complete",
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, annotation)},
    }
    marker = folder / "completed.json"
    marker.write_text(json.dumps(completed))
    manifest = load_source_manifest(folder.parent)
    assert manifest["source_format"] == "episodes"
    assert manifest["files"][0]["file"] == "train_task_000000/labels.npz"
    assert manifest["files"][0]["source_video"] == "videos/task.mp4"
    result = read_reward_labels(source, state)
    assert result["interaction_target"].shape == (8, 2, 16)
    assert result["source_status"] == "complete"
    assert "verification_state" not in result
    assert "label_acceptance" not in result
    np.testing.assert_array_equal(result["source_relation"], arrays["relation"])
    np.testing.assert_array_equal(result["event"], arrays["one_hot"])
    expected_valid = np.ones((8, 2), dtype=bool)
    expected_valid[[1, 2, 3, 4], [0, 1, 0, 1]] = False
    np.testing.assert_array_equal(result["event_mask"].all(-1), expected_valid)
    np.testing.assert_array_equal(result["motion_mask"][..., 4], expected_valid)
    assert result["motion_mask"][..., :4].all()
    selected = read_reward_labels(source, state, frame_indices=[5, 1], target_id="fries")
    np.testing.assert_array_equal(selected["source_relation"], arrays["relation"][[5, 1], 1])
    completed["status"] = "incomplete"
    marker.write_text(json.dumps(completed))
    with pytest.raises(ValueError, match="not complete"):
        read_reward_labels(source, state)
    completed["status"] = "complete"
    marker.write_text(json.dumps(completed))
    annotation.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_source_manifest(folder.parent)
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_reward_labels(source, state)


def test_motion_uses_incoming_frames_rotation_geometry_and_both_grippers():
    eef = np.zeros((3, 14))
    eef[:, 6] = [0.1, 0.2, 0.2]
    eef[:, 13] = [0.8, 0.7, 0.7]
    eef[1:, 0] = 0.001
    eef[:, 10] = [np.pi - 0.01, -np.pi + 0.01, -np.pi + 0.01]
    labels, mask = robot_motion_labels(eef)
    np.testing.assert_array_equal(labels, [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    assert mask.all()
    labels, mask = robot_motion_labels(eef[:1])
    assert not labels.any() and not mask.any()


def test_masks_missing_pose_but_keeps_confirmed_motion_from_other_arm():
    eef = np.zeros((3, 14))
    eef[:, 0] = np.nan
    eef[2, 7] = 0.01
    labels, mask = robot_motion_labels(eef)
    assert not mask[1, 0]
    assert labels[2, 0] and mask[2, 0]


def test_rejects_reordered_classes_inconsistent_onehot_and_state_frames(tmp_path):
    source, state = write_sources(tmp_path)
    with np.load(source) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["class_names"] = arrays["class_names"][::-1]
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="event order"):
        read_reward_labels(source, state)
    arrays["class_names"] = np.array(EVENTS)
    arrays["one_hot"][0] = 0
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="one_hot"):
        read_reward_labels(source, state)
    arrays["one_hot"] = np.eye(8, dtype=np.uint8)[arrays["class_id"]]
    arrays["frame_index"][0] = 99
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="frame indices differ"):
        read_reward_labels(source, state)


@pytest.mark.parametrize("targets", [("bottle",), ("burger", "fries")])
def test_dataset_uses_actual_group_endpoints_and_or_merges_objects(tmp_path, targets):
    source, state = write_sources(tmp_path, targets=targets)
    result = read_reward_labels(source, state)
    np.savez_compressed(tmp_path / "train_task_000000.npz", **result)
    sample = {
        "video": torch.zeros(1, 3, 5, 32, 32), "action": torch.zeros(5, 14),
        "split": "train", "task": "task", "episode_index": 0, "start_frame": 3,
    }
    batch = EvaluatorDataset([sample], reward_labels=tmp_path)[0]
    ids = [3, 7]
    np.testing.assert_array_equal(batch["frame_indices"].numpy(), [3, 4, 5, 6, 7])
    expected_target = result["interaction_target"][ids].max(axis=1)
    expected_mask = result["interaction_mask"][ids].all(axis=1)
    np.testing.assert_array_equal(batch["interaction_target"].numpy(), expected_target)
    np.testing.assert_array_equal(batch["interaction_mask"].numpy(), expected_mask)
    assert batch["interaction_mask"].all()
    assert batch["interaction_is_multi_object"].eq(len(targets) > 1).all()
    assert batch["interaction_target"].max() == 1
    if len(targets) > 1:
        assert (batch["interaction_target"][:, 3:11].sum(-1) == 2).all()
