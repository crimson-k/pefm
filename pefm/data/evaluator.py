"""Thin evaluator view over BWM's RoboTwinUnifiedDataset."""

import torch
from torch.utils.data import DataLoader, Dataset


def _group_ids(num_frames: int) -> torch.Tensor:
    if num_frames < 1 or (num_frames - 1) % 4:
        raise ValueError(f"Expected 1+4k frames, got {num_frames}")
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.arange(1, (num_frames - 1) // 4 + 1).repeat_interleave(4)])


def _group_mean(sequence: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    groups = sequence.new_zeros((int(group_ids[-1]) + 1, sequence.shape[-1]))
    groups.index_add_(0, group_ids, sequence)
    counts = torch.bincount(group_ids).to(sequence).unsqueeze(-1)
    return groups / counts


class EvaluatorDataset(Dataset):
    """Convert one BWM sample into aligned RGB and grouped EEF tensors."""

    def __init__(self, bwm_dataset: Dataset):
        self.bwm_dataset = bwm_dataset

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

        group_ids = _group_ids(rgb.shape[2])
        frame_indices = sample.get("frame_indices", range(rgb.shape[2]))
        return {
            "sample_id": sample.get("sample_id", index),
            "task": sample.get("task", ""),
            "episode": int(sample.get("episode_index", -1)),
            "split": sample.get("split", ""),
            "rgb": rgb,
            "eef": _group_mean(eef, group_ids),
            "frame_indices": torch.as_tensor(frame_indices, dtype=torch.long),
            "group_ids": group_ids,
            "reset": torch.arange(int(group_ids[-1]) + 1) == 0,
        }


def build_evaluator_dataloader(bwm_dataset, batch_size=4, shuffle=True, num_workers=4):
    """Use normal collation instead of BWM's batch-size-one runner collate."""
    return DataLoader(
        EvaluatorDataset(bwm_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
