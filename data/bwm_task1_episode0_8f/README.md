# BWM episode 0 — 8-frame chunks

This is a copied subset of `converted_dataset_task1` for a short VJEPA-AC predictor smoke test.

- Source: `adjust_bottle/aloha-agilex_clean_50/episode_000000`
- Source length: 140 frames
- Windows: 17 non-overlapping chunks of 8 frames
- Covered frames: 0–135
- Dropped tail: 136–139 (not a complete 8-frame chunk)

Load it with BWM operators configured explicitly for 8 frames:

```python
base_path = "data/bwm_task1_episode0_8f"
dataset = RoboTwinUnifiedDataset(
    base_path=base_path,
    metadata_path=f"{base_path}/metadata.jsonl",
    main_data_operator=create_video_operator(base_path=base_path, num_frames=8),
    special_operator_map={
        "action": LoadCobotAction(
            base_path=base_path,
            stat=load_action_stats(f"{base_path}/stat.json"),
            num_frames=8,
        )
    },
)
```

Each raw sample has `video [1,3,8,480,640]` and `action [1,8,14]`. Do not wrap this short dataset with the current `EvaluatorDataset`: its 81→21 grouper requires a frame count of `1+4k`, and 8 does not satisfy that contract.

For the isolated predictor test, omit `group_ids` so the adapter keeps all eight
timesteps:

```python
sample = dataset[0]
rgb = torch.as_tensor(sample["video"]).unsqueeze(0)  # [B,1,3,8,480,640]
tokens = VJEPAObservationAdapter(encoder, input_size=256)(rgb)
assert tokens.shape == (1, 8, 256, 1408)
```
