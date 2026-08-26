import json
from typing import Iterable, Optional

from diffsynth.core import UnifiedDataset

from ..utils import load_action_stats

from .operators import LoadCobotAction, ResolvePromptEmbPath, create_video_operator


class RoboTwinUnifiedDataset(UnifiedDataset):
    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        repeat: int = 1,
        data_file_keys: Iterable[str] = ("video", "action"),
        main_data_operator=lambda x: x,
        special_operator_map: Optional[dict] = None,
        sample_indices: Optional[Iterable[int]] = None,
        temporal_template_sampling: bool = False,
        temporal_num_frames: Optional[int] = None,
        temporal_num_history_frames: int = 1,
        temporal_align_to_vae_latent: bool = False,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = int(repeat)
        self.data_file_keys = tuple(data_file_keys)
        self.main_data_operator = main_data_operator
        self.special_operator_map = {} if special_operator_map is None else dict(special_operator_map)
        self.sample_indices = None if sample_indices is None else [int(index) for index in sample_indices]
        self.temporal_template_sampling = bool(temporal_template_sampling)
        self.temporal_num_frames = None if temporal_num_frames is None else int(temporal_num_frames)
        self.temporal_num_history_frames = int(temporal_num_history_frames)
        self.temporal_align_to_vae_latent = bool(temporal_align_to_vae_latent)
        self.load_from_cache = False
        self.data = self._load_metadata(metadata_path)
        self._apply_sample_selection()
        print(f"Dataset size: {len(self.data)}, repeat: {self.repeat}, total: {len(self)}")

    def _load_metadata(self, metadata_path: str):
        if metadata_path.endswith(".json"):
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        if metadata_path.endswith(".jsonl"):
            rows = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        rows.append(json.loads(text))
            return rows

    def _apply_sample_selection(self):
        if self.sample_indices is None:
            return
        invalid = [index for index in self.sample_indices if index < 0 or index >= len(self.data)]
        if invalid:
            raise IndexError(f"Sample indices out of range: {invalid} (dataset size: {len(self.data)})")
        self.data = [self.data[index] for index in self.sample_indices]

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, data_id: int):
        data = self.data[int(data_id) % len(self.data)].copy()
        temporal_info = self._build_temporal_sample_info(data)
        frame_indices = None if temporal_info is None else temporal_info.get("frame_indices")
        if temporal_info is not None:
            data.update(temporal_info)
        for key in self.data_file_keys:
            if key in self.special_operator_map:
                source = data[key] if key in data else data
                data[key] = self.special_operator_map[key](
                    self._wrap_frame_range_metadata(data, source, frame_indices=frame_indices)
                )
            elif key in data:
                data[key] = self.main_data_operator(
                    self._wrap_frame_range_metadata(data, data[key], frame_indices=frame_indices)
                )
        return data

    def _resolve_frame_range(self, data: dict):
        start_frame = int(data.get("start_frame", 0))
        if data.get("end_frame") is not None:
            end_frame = int(data["end_frame"])
        elif data.get("length") is not None:
            end_frame = start_frame + int(data["length"]) - 1
        else:
            end_frame = start_frame
        return start_frame, end_frame

    def _build_temporal_sample_info(self, data):
        if not self.temporal_template_sampling or not isinstance(data, dict):
            return None

        num_frames = self.temporal_num_frames
        num_history_frames = self.temporal_num_history_frames
        if num_frames is None or num_frames <= 1:
            return None
        if num_history_frames <= 1 or num_history_frames >= num_frames:
            return None
        if (num_history_frames - 1) % 4 != 0:
            raise ValueError(
                "`num_history_frames - 1` must be divisible by 4 when history_template_sampling is enabled."
            )

        start_frame, end_frame = self._resolve_frame_range(data)
        future_len = int(num_frames - num_history_frames)
        lower = max(1, start_frame)
        upper = end_frame
        raw_length = data.get("raw_length")
        raw_end = None
        if raw_length is not None:
            raw_end = max(0, int(raw_length) - 1)
            if future_len > 0:
                upper = min(upper, raw_end - future_len + 1)
        elif future_len > 0:
            upper = min(upper, end_frame - future_len + 1)

        if self.temporal_align_to_vae_latent:
            aligned_upper = max(lower, upper)
            candidates = [
                frame_id for frame_id in range(lower, aligned_upper + 1)
                if (int(frame_id) - 1) % 4 == 0
            ]
            if not candidates:
                raise ValueError(
                    f"No valid temporal future start found for range [{lower}, {aligned_upper}]"
                )
            future_start = candidates[0]
        else:
            future_start = lower

        history_indices = [0]
        history_tail_start = future_start - (num_history_frames - 1)
        history_indices.extend(
            max(0, history_tail_start + offset) for offset in range(num_history_frames - 1)
        )

        future_indices = [future_start + offset for offset in range(future_len)]
        if raw_end is not None:
            future_indices = [min(frame_id, raw_end) for frame_id in future_indices]
        else:
            future_indices = [min(frame_id, end_frame) for frame_id in future_indices]
        return {
            "frame_indices": list(history_indices + future_indices),
            "temporal_future_start": int(future_start),
        }

    def _wrap_frame_range_metadata(self, data, payload, frame_indices=None):
        if not isinstance(data, dict):
            return payload

        start_frame, end_frame = self._resolve_frame_range(data)

        def wrap_item(item):
            if isinstance(item, str):
                wrapped = {"data": item, "start_frame": start_frame, "end_frame": end_frame}
                if frame_indices is not None:
                    wrapped["frame_indices"] = frame_indices
                return wrapped
            if isinstance(item, dict):
                wrapped = item.copy()
                wrapped["start_frame"] = start_frame
                wrapped["end_frame"] = end_frame
                if frame_indices is not None:
                    wrapped["frame_indices"] = frame_indices
                return wrapped
            return item

        if isinstance(payload, (list, tuple)):
            return [wrap_item(item) for item in payload]
        return wrap_item(payload)


def build_robotwin_train_dataset(args) -> RoboTwinUnifiedDataset:
    keys = tuple(args.data_keys)

    special_operator_map = {}
    if args.modes["text"] != "none":
        for key in ("prompt_emb", "negative_prompt_emb"):
            if key in keys:
                special_operator_map[key] = ResolvePromptEmbPath(base_path=args.dataset_base_path)

    if "action" in keys:
        if not args.action_type:
            raise ValueError("`action_type` is required when loading action data.")
        special_operator_map["action"] = LoadCobotAction(
            base_path=args.dataset_base_path,
            action_type=args.action_type,
            stat=load_action_stats(args.action_stat_path),
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
        )

    return RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=tuple(keys),
        main_data_operator=create_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=args.spatial_division_factor,
            width_division_factor=args.spatial_division_factor,
            num_frames=args.num_frames,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            resize_mode=args.resize_mode,
        ),
        special_operator_map=special_operator_map,
        temporal_template_sampling=bool(args.history_template_sampling),
        temporal_num_frames=int(args.num_frames),
        temporal_num_history_frames=int(args.num_history_frames),
        temporal_align_to_vae_latent=(args.modes["vae"] == "emb"),
    )


def build_robotwin_infer_dataset(args) -> RoboTwinUnifiedDataset:
    special_operator_map = {}
    if "action" in args.data_keys:
        special_operator_map["action"] = LoadCobotAction(
            base_path=args.dataset_base_path,
            action_type=args.action_type,
            stat=load_action_stats(args.action_stat_path),
            num_frames=None,
            align_num_frames=False,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
        )

    return RoboTwinUnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=args.data_keys,
        main_data_operator=create_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=args.spatial_division_factor,
            width_division_factor=args.spatial_division_factor,
            num_frames=1,
            time_division_factor=args.time_division_factor,
            time_division_remainder=args.time_division_remainder,
            resize_mode=args.resize_mode,
        ),
        special_operator_map=special_operator_map,
    )


TRAIN_DATASET_BUILDERS = {
    "robotwin": build_robotwin_train_dataset,
}

INFER_DATASET_BUILDERS = {
    "robotwin": build_robotwin_infer_dataset,
}


def build_train_dataset(args) -> RoboTwinUnifiedDataset:
    return TRAIN_DATASET_BUILDERS[args.dataset_name](args)


def build_infer_dataset(args) -> RoboTwinUnifiedDataset:
    return INFER_DATASET_BUILDERS[args.dataset_name](args)
