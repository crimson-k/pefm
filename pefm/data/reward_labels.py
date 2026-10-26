"""读取 V23 事件标签和 Parquet 状态，生成关系(3)、事件(8)、运动(5)标签。

用法：python -m pefm.data.reward_labels --output outputs/reward_labels/v23
可用 --episode train_adjust_bottle_000000 只读一集，--target-id bottle 选择物体。
默认保留物体轴 [F,K,D]；指定 target_id 后为 [F,D]。frame_indices 可传入
训练实际采样的时间组末帧；先在原始连续帧上算运动，再取这些帧，避免跨帧误算。
默认仅将复核确定的视觉标签设为有效；--include-unverified 也使用尚未复核的
非疑点候选。支持旧导出 manifest 和新版 episodes/*/labels.npz；保留源质量字段，
完成标记只说明源处理完成，不额外声明人工验收通过。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .dataloader import relative_eef_actions

WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = WORKSPACE / "interaction_teacher/annotation/support/policy_research/v23_formal300_unaccepted_onehot_export_20261024"
DEFAULT_DATASET = WORKSPACE / "boundless-world-model/converted_dataset_bwm_6tasks_256x192"
RELATIONS = ("未接触", "接触", "抓持")
EVENTS = ("空移，物体静止", "空移，物体运动", "机械臂静止，物体运动", "双方静止",
          "接触，物体运动", "接触，物体静止", "抓持，物体随动", "抓持，物体静止")
MOTIONS = ("末端平移", "末端旋转", "夹爪张开", "夹爪闭合", "物体运动")
RELATION_IDS = np.array([0, 0, 0, 0, 1, 1, 2, 2])
OBJECT_MOVING = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=np.uint8)


def load_source_manifest(source):
    """读取旧 manifest，或根据新版每集完成标记建立输入清单。"""
    source = Path(source)
    if (source / "export_manifest.json").is_file():
        return json.loads((source / "export_manifest.json").read_text())
    entries = []
    for folder in sorted(path for path in source.iterdir() if path.is_dir()):
        completed = json.loads((folder / "completed.json").read_text())
        if completed["status"] != "complete":
            raise ValueError(f"Episode is not complete: {folder.name}")
        for name, expected in completed["sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source checksum mismatch: {folder / name}")
        annotation = json.loads((folder / "annotations.json").read_text())
        entries.append({
            "episode": folder.name, "file": f"{folder.name}/labels.npz",
            "sha256": completed["sha256"]["labels.npz"],
            "completed_sha256": hashlib.sha256((folder / "completed.json").read_bytes()).hexdigest(),
            "annotations_sha256": completed["sha256"]["annotations.json"],
            "source_video": annotation["source_row"]["video"],
            "source_action": annotation["source_row"]["action"],
        })
    if not entries:
        raise ValueError(f"No episode label files found in {source}")
    return {"source_format": "episodes", "source_status": "complete", "files": entries}


def robot_motion_labels(eef, translation_m=1e-4, rotation_rad=math.radians(0.1),
                        gripper_delta=1e-4):
    """按原始相邻帧计算属性，两臂分别判断后取或；夹爪使用记录的控制开度。

    平移和旋转阈值沿用 V23。第 0 帧沿用其 0→1 转移定义。
    两臂同时分别张开、闭合时，两个夹爪标签可以同时为真。
    """
    eef = torch.as_tensor(eef, dtype=torch.float64)
    if eef.ndim != 2 or eef.shape[1] != 14:
        raise ValueError(f"Expected observation.state [F,14], got {tuple(eef.shape)}")
    if min(translation_m, rotation_rad, gripper_delta) <= 0:
        raise ValueError("Motion thresholds must be positive")
    if len(eef) < 2:
        return np.zeros((len(eef), 4), dtype=np.uint8), np.zeros((len(eef), 4), dtype=bool)
    delta = relative_eef_actions(eef).reshape(-1, 2, 7)
    delta[0] = delta[1].clone()
    measurements = torch.stack((
        delta[..., :3].norm(dim=-1), delta[..., 3:6].norm(dim=-1),
        delta[..., 6], -delta[..., 6],
    ), dim=-1)
    known = torch.isfinite(measurements)
    thresholds = torch.tensor([translation_m, rotation_rad, gripper_delta, gripper_delta])
    positive = ((measurements >= thresholds) & known).any(dim=1)
    valid = positive | known.all(dim=1)
    return positive.numpy().astype(np.uint8), valid.numpy()


def read_reward_labels(onehot_path, state_path, frame_indices=None, target_id=None,
                       include_unverified=False, translation_m=1e-4,
                       rotation_rad=math.radians(0.1), gripper_delta=1e-4):
    """返回三类标签、逐维 mask、16 维拼接标签和未修改的源质量字段。"""
    with np.load(onehot_path, allow_pickle=False) as source:
        result = {key: source[key] for key in source.files}
    frame_keys = set(result) - {"target_ids", "class_names", "fps"}
    ids = result["class_id"]
    if tuple(result["class_names"]) != EVENTS:
        raise ValueError(f"Unexpected V23 event order in {onehot_path}")
    if ids.ndim != 2 or not np.isin(ids, np.arange(8)).all():
        raise ValueError(f"Invalid event class IDs in {onehot_path}")
    if not np.array_equal(result["one_hot"], np.eye(8, dtype=np.uint8)[ids]):
        raise ValueError(f"one_hot does not match class_id in {onehot_path}")
    candidate_export = "verification_state" in result
    if not candidate_export:
        marker = json.loads(Path(onehot_path).with_name("completed.json").read_text())
        if marker["status"] != "complete":
            raise ValueError(f"Episode is not complete: {onehot_path}")
        if hashlib.sha256(Path(onehot_path).read_bytes()).hexdigest() != marker["sha256"]["labels.npz"]:
            raise ValueError(f"Source checksum mismatch: {onehot_path}")
        result["source_status"] = np.array(marker["status"])
    if "relation" in result:
        if not np.array_equal(result["relation"], RELATION_IDS[ids]):
            raise ValueError(f"Source relation does not match event classes in {onehot_path}")
        result["source_relation"] = result["relation"]
        frame_keys.add("source_relation")
    if "object_moving" in result and not np.array_equal(result["object_moving"], OBJECT_MOVING[ids]):
        raise ValueError(f"Source object_moving does not match event classes in {onehot_path}")
    table = pq.read_table(state_path, columns=["observation.state", "frame_index"])
    if not np.array_equal(table["frame_index"].to_numpy(), result["frame_index"]):
        raise ValueError(f"State and label frame indices differ: {state_path}")
    robot, robot_mask = robot_motion_labels(
        table["observation.state"].to_pylist(), translation_m, rotation_rad, gripper_delta,
    )
    visual_valid = (
        ~result["annotation_uncertain"] & ~result["failed"]
        & (result["verification_rejected"] != 1)
    )
    if candidate_export:
        visual_valid &= result["verification_uncertain"] != 1
    if candidate_export and not include_unverified:
        # -1 表示未确认，不能按 False 处理，也不能伪装为已验收标签。
        visual_valid &= (result["uncertain"] == 0) & (result["verification_state"] == "pass")
    elif not candidate_export:
        visual_valid &= result["uncertain"] == 0
    relation = np.eye(3, dtype=np.uint8)[RELATION_IDS[ids]]
    event = result["one_hot"]
    motion = np.concatenate((
        np.broadcast_to(robot[:, None, :], (*ids.shape, 4)), OBJECT_MOVING[ids][..., None],
    ), axis=-1)
    relation_mask = np.broadcast_to(visual_valid[..., None], relation.shape).copy()
    event_mask = np.broadcast_to(visual_valid[..., None], event.shape).copy()
    motion_mask = np.concatenate((
        np.broadcast_to(robot_mask[:, None, :], (*ids.shape, 4)), visual_valid[..., None],
    ), axis=-1)
    result.update(
        relation=relation, event=event, motion=motion, relation_mask=relation_mask,
        event_mask=event_mask, motion_mask=motion_mask,
        interaction_target=np.concatenate((relation, event, motion), axis=-1).astype(np.float32),
        interaction_mask=np.concatenate((relation_mask, event_mask, motion_mask), axis=-1),
    )
    frame_keys.update(("relation", "event", "motion", "relation_mask", "event_mask", "motion_mask",
                       "interaction_target", "interaction_mask"))
    if target_id is not None:
        targets = result["target_ids"].tolist()
        if target_id not in targets:
            raise ValueError(f"Unknown target {target_id!r}; available targets: {targets}")
        index = targets.index(target_id)
        for key in frame_keys:
            if result[key].ndim >= 2:
                result[key] = result[key][:, index]
        result["target_ids"] = np.array([target_id])
    if frame_indices is not None:
        indices = np.asarray(frame_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or ((indices < 0) | (indices >= len(ids))).any():
            raise ValueError("frame_indices must be a 1-D array of valid source frame indices")
        for key in frame_keys:
            result[key] = result[key][indices]
    result.update(
        relation_names=np.array(RELATIONS), event_names=np.array(EVENTS), motion_names=np.array(MOTIONS),
        source_format=np.array("candidate_export" if candidate_export else "episodes"),
        gripper_source=np.array("observation.state control opening delta; not measured finger width"),
        translation_m=np.array(translation_m), rotation_rad=np.array(rotation_rad),
        gripper_delta=np.array(gripper_delta), include_unverified=np.array(include_unverified),
    )
    if candidate_export:
        result.update(diagnostic_only=np.array(True), label_acceptance=np.array(False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, help="写出每集 NPZ 和 manifest；省略时只读取并汇总")
    parser.add_argument("--episode", nargs="+", help="例如 train_adjust_bottle_000000")
    parser.add_argument("--target-id")
    parser.add_argument("--frames", nargs="+", type=int, help="原始帧编号，可传入21个时间组末帧")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--translation-m", type=float, default=1e-4)
    parser.add_argument("--rotation-rad", type=float, default=math.radians(0.1))
    parser.add_argument("--gripper-delta", type=float, default=1e-4)
    args = parser.parse_args()
    manifest = load_source_manifest(args.source)
    entries = {entry["episode"]: entry for entry in manifest["files"]}
    episodes = sorted(entries if args.episode is None else args.episode)
    rows = {}
    for name in ("metadata_train.jsonl", "metadata_test.jsonl"):
        for line in (args.dataset / name).read_text().splitlines():
            row = json.loads(line)
            key = f"{row['split']}_{row['task']}_{int(row['source_episode_index']):06d}"
            rows[key] = row
    summary = {
        "source": str(args.source.resolve()), "dataset": str(args.dataset.resolve()),
        "relation_names": RELATIONS, "event_names": EVENTS, "motion_names": MOTIONS,
        "source_format": manifest.get("source_format", "candidate_export"),
        "source_status": manifest.get("source_status", manifest.get("status")),
        "diagnostic_only": manifest.get("diagnostic_only"), "label_acceptance": manifest.get("label_acceptance"),
        "include_unverified": args.include_unverified, "files": [],
    }
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        entry = entries[episode]
        label_path = args.source / entry["file"]
        for key, field in (("source_video", "video"), ("source_action", "action")):
            if key in entry and entry[key] != rows[episode][field]:
                raise ValueError(f"Episode {episode} uses a different source {field}")
        if hashlib.sha256(label_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source NPZ hash mismatch: {label_path}")
        result = read_reward_labels(
            label_path, args.dataset / rows[episode]["action"], args.frames, args.target_id,
            args.include_unverified, args.translation_m, args.rotation_rad, args.gripper_delta,
        )
        filename = f"{episode}.npz"
        if args.output:
            np.savez_compressed(args.output / filename, **result)
        summary["files"].append({
            "episode": episode, "file": filename, "target_ids": result["target_ids"].tolist(),
            "source_file": entry["file"], "source_sha256": entry["sha256"],
            "completed_sha256": entry.get("completed_sha256"),
            "annotations_sha256": entry.get("annotations_sha256"),
            "source_video": rows[episode]["video"], "source_action": rows[episode]["action"],
            "shape": list(result["interaction_target"].shape),
            "valid_per_dimension": result["interaction_mask"].reshape(-1, 16).sum(0).tolist(),
        })
    if args.output:
        (args.output / "reward_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "episodes": len(episodes), "frames": sum(row["shape"][0] for row in summary["files"]),
        "targets": sum(len(row["target_ids"]) for row in summary["files"]),
        "valid_per_dimension": np.sum([row["valid_per_dimension"] for row in summary["files"]], axis=0).tolist(),
        "output": str(args.output) if args.output else None,
        "label_acceptance": summary["label_acceptance"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
