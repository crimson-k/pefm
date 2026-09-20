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


class EvaluatorDataset(Dataset):
    """Convert one BWM sample into aligned RGB and grouped EEF tensors."""

    def __init__(self, bwm_dataset: Dataset, time_division_factor: int = 4):
        self.bwm_dataset = bwm_dataset
        self.time_division_factor = time_division_factor

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
        frame_indices = sample.get("frame_indices", range(rgb.shape[2]))
        return {
            "sample_id": sample.get("sample_id", index),
            "task": sample.get("task", ""),
            "episode": int(sample.get("episode_index", -1)),
            "split": sample.get("split", ""),
            "rgb": rgb,
            "eef": grouped_eef,
            "action_delta": relative_eef_actions(grouped_eef),
            "frame_indices": torch.as_tensor(frame_indices, dtype=torch.long),
            "group_ids": group_ids,
            "reset": torch.arange(int(group_ids[-1]) + 1) == 0,
        }


def build_RGB_dataloader(bwm_dataset, time_division_factor, batch_size=4, shuffle=True, num_workers=4):
    return DataLoader(
        EvaluatorDataset(bwm_dataset, time_division_factor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def stage2_teacher_batch(rgb, eef, time_division_factor=4):
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
        "action_delta": relative_eef_actions(grouped_eef),
        "reset": (torch.arange(groups, device=rgb.device) == 0).expand(rgb.shape[0], -1),
    }
