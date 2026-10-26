"""Thin evaluator view over BWM's RoboTwinUnifiedDataset."""

import torch
from torch.utils.data import DataLoader, Dataset
from src.bwm.wan_video_action.data.operators import LoadCobotAction


def _group_ids(num_frames: int, time_division_factor) -> torch.Tensor:
    if num_frames < 1 or (num_frames - 1) % time_division_factor:
        raise ValueError(f"Expected 1+4k frames, got {num_frames}")
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(1, (num_frames - 1) // time_division_factor + 1).repeat_interleave(time_division_factor)])


class RawCobotAction(LoadCobotAction):
    """Read the absolute EEF pose before percentile clipping and normalization."""

    def _normalize_bound(self, data, *args, **kwargs):
        return data


def _euler_xyz_to_quat(euler):
    roll, pitch, yaw = (angle / 2 for angle in euler.unbind(-1))
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), dim=-1)


def _relative_rotvec(previous, current):
    q0 = _euler_xyz_to_quat(previous)
    q1 = _euler_xyz_to_quat(current)
    w0, v0 = q0[..., :1], -q0[..., 1:]  # inverse of unit quaternion
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w0 * w1 - (v0 * v1).sum(-1, keepdim=True)
    v = w0 * v1 + w1 * v0 + torch.linalg.cross(v0, v1)
    sign = torch.where(w < 0, -1.0, 1.0)
    w, v = sign * w, sign * v
    norm = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, w)
    return v * (angle / norm.clamp_min(1e-8))


def relative_eef_actions(eef):
    """Group-to-group xyz, body-frame rotation vector, and gripper deltas."""
    previous, current = eef[..., :-1, :], eef[..., 1:, :]
    parts = []
    for start in (3, 10):
        parts.extend((
            current[..., start - 3:start] - previous[..., start - 3:start],
            _relative_rotvec(previous[..., start:start + 3], current[..., start:start + 3]),
            current[..., start + 3:start + 4] - previous[..., start + 3:start + 4],
        ))
    delta = torch.cat(parts, dim=-1)
    return torch.cat((torch.zeros_like(eef[..., :1, :]), delta), dim=-2)


def normalize_action_delta(action_delta, stats):
    """Map each relative-action dimension to the train-set p01--p99 range."""
    if stats is None:
        return action_delta
    p01 = torch.as_tensor(
        stats["p01"], device=action_delta.device, dtype=action_delta.dtype,
    )
    p99 = torch.as_tensor(
        stats["p99"], device=action_delta.device, dtype=action_delta.dtype,
    )
    span = p99 - p01
    normalized = 2.0 * (action_delta - p01) / span.clamp_min(1e-6) - 1.0
    normalized = normalized.clamp(-1.0, 1.0)
    return torch.where(span > 1e-6, normalized, action_delta.clamp(-1.0, 1.0))


def merge_object_labels(target, mask):
    """沿物体轴 OR 合并 [...,K,16]；有确定正例即为 1，全体确定才可为 0。"""
    mask = mask.to(device=target.device, dtype=torch.bool)
    if (((target != 0) & (target != 1)) & mask).any():
        raise ValueError("Valid interaction labels must be binary; mask unknown labels")
    positive = ((target == 1) & mask).any(dim=-2)
    return positive.to(dtype=target.dtype), positive | mask.all(dim=-2)


class EvaluatorDataset(Dataset):
    """Convert one BWM sample into aligned RGB and grouped EEF tensors."""

    def __init__(self, bwm_dataset: Dataset, time_division_factor: int = 4,
                 action_delta_stats=None, reward_labels=None):
        self.bwm_dataset = bwm_dataset
        self.time_division_factor = time_division_factor
        self.action_delta_stats = action_delta_stats
        self.reward_labels = reward_labels

    def __len__(self):
        return len(self.bwm_dataset)

    def __getitem__(self, index):
        sample = self.bwm_dataset[index]
        rgb = torch.as_tensor(sample["video"], dtype=torch.float32)
        eef = torch.as_tensor(sample["action"], dtype=torch.float32)
        if eef.ndim == 3 and eef.shape[0] == 1:
            eef = eef[0]
        if rgb.ndim != 5 or eef.ndim != 2 or rgb.shape[2] != eef.shape[0]:
            raise ValueError(f"Expected video (V,C,F,H,W) and action (F,A), got {rgb.shape}, {eef.shape}")

        group_ids = _group_ids(rgb.shape[2], self.time_division_factor)
        # Each Wan group uses its final frame as the transition endpoint.
        endpoints = torch.cat((torch.zeros(1, dtype=torch.long),
                               torch.arange(self.time_division_factor, rgb.shape[2], self.time_division_factor)))
        grouped_eef = eef[endpoints]
        start_frame = int(sample.get("start_frame", 0))
        frame_indices = sample.get("frame_indices", range(start_frame, start_frame + rgb.shape[2]))
        action_delta = normalize_action_delta(
            relative_eef_actions(grouped_eef), self.action_delta_stats,
        )
        result = {
            "sample_id": sample.get("sample_id", index),
            "task": sample.get("task", ""),
            "episode": int(sample.get("episode_index", -1)),
            "split": sample.get("split", ""),
            "rgb": rgb,
            "eef": grouped_eef,
            "action_delta": action_delta,
            "frame_indices": torch.as_tensor(frame_indices, dtype=torch.long),
            "group_ids": group_ids,
            "reset": torch.arange(int(group_ids[-1]) + 1) == 0,
        }
        if self.reward_labels is not None:
            from pathlib import Path
            import numpy as np

            episode = int(sample.get("source_episode_index", sample["episode_index"]))
            key = f"{sample['split']}_{sample['task']}_{episode:06d}"
            source_frames = torch.as_tensor(frame_indices, dtype=torch.long)[endpoints].numpy()
            with np.load(Path(self.reward_labels) / f"{key}.npz", allow_pickle=False) as labels:
                if not np.array_equal(labels["frame_index"], np.arange(len(labels["frame_index"]))):
                    raise ValueError(f"Reward source must contain all original frames: {key}")
                target = torch.from_numpy(labels["interaction_target"][source_frames])
                mask = torch.from_numpy(labels["interaction_mask"][source_frames])
            # 汉堡、薯条先合并为同一份 multi-hot，再对共享预测计算一次损失。
            sample["interaction_is_multi_object"] = target.shape[1] > 1
            sample["interaction_target"], sample["interaction_mask"] = merge_object_labels(target, mask)
        if "interaction_target" in sample:
            target = torch.as_tensor(sample["interaction_target"], dtype=torch.float32)
            mask = torch.as_tensor(
                sample.get("interaction_mask", torch.ones_like(target, dtype=torch.bool)),
                dtype=torch.bool,
            )
            expected = (int(group_ids[-1]) + 1, 16)
            if target.shape != expected or mask.shape != expected:
                raise ValueError(f"Expected grouped interaction_target/mask {expected}, got {target.shape}/{mask.shape}")
            # 元数据可以为整段指定一个布尔值，也可以逐时间组标记双臂事件。
            dual_arm = torch.as_tensor(sample.get("event_is_dual_arm", False), dtype=torch.bool)
            if dual_arm.ndim == 0:
                dual_arm = dual_arm.expand(expected[0])
            if dual_arm.shape != (expected[0],):
                raise ValueError(f"Expected event_is_dual_arm scalar or shape {(expected[0],)}")
            multi_object = torch.as_tensor(sample.get("interaction_is_multi_object", False), dtype=torch.bool)
            if multi_object.ndim == 0:
                multi_object = multi_object.expand(expected[0])
            if multi_object.shape != (expected[0],):
                raise ValueError(f"Expected interaction_is_multi_object scalar or shape {(expected[0],)}")
            result.update(
                interaction_target=target, interaction_mask=mask, event_is_dual_arm=dual_arm,
                interaction_is_multi_object=multi_object,
            )
        return result


def build_RGB_dataloader(bwm_dataset, time_division_factor, batch_size=4, shuffle=True,
                         num_workers=4, action_delta_stats=None, reward_labels=None):
    return DataLoader(
        EvaluatorDataset(bwm_dataset, time_division_factor, action_delta_stats, reward_labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def stage2_teacher_batch(rgb, eef, time_division_factor=4, action_delta_stats=None):
    """Build the 21-step frozen-Teacher input from one BWM training sample."""
    if rgb.ndim == 5:
        rgb = rgb.unsqueeze(0)
    if eef.ndim == 2:
        eef = eef.unsqueeze(0)
    if rgb.ndim != 6 or eef.ndim != 3 or rgb.shape[0] != eef.shape[0]:
        raise ValueError(f"Expected RGB [B,V,C,F,H,W] and EEF [B,F,14], got {rgb.shape}, {eef.shape}")
    group_ids = _group_ids(rgb.shape[3], time_division_factor).to(rgb.device)
    if eef.shape[1] == rgb.shape[3]:
        endpoints = torch.cat((
            torch.zeros(1, dtype=torch.long, device=eef.device),
            torch.arange(time_division_factor, eef.shape[1], time_division_factor, device=eef.device),
        ))
        grouped_eef = eef[:, endpoints]
    elif eef.shape[1] == 1 + (rgb.shape[3] - 1) // time_division_factor:
        grouped_eef = eef
    else:
        raise ValueError(
            f"EEF length {eef.shape[1]} is neither frame length {rgb.shape[3]} "
            "nor grouped length"
        )
    groups = grouped_eef.shape[1]
    return {
        "rgb": rgb,
        "group_ids": group_ids.expand(rgb.shape[0], -1),
        "action_delta": normalize_action_delta(
            relative_eef_actions(grouped_eef), action_delta_stats,
        ),
        "reset": (torch.arange(groups, device=rgb.device) == 0).expand(rgb.shape[0], -1),
    }
