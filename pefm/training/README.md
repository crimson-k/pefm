# Stage 2 simplified training

The model follows `notes/stage2_simplified.html` in the parent workspace:

```text
GT RGB -> frozen V-JEPA -> frozen shared Aggregator G -> e*
DiT H, sigma -> Adapter P -> the same G -> estimated e
estimated e -> linear B -> one GRUCell with MLP(action) -> linear D -> predicted e
```

Phase A trains the action MLP, B, GRU and D on clean `e*` only. Phase B trains
P, first on current alignment, then on alignment plus future prediction.
Phase C freezes all auxiliary modules and adds the two errors to BWM's flow
matching loss. Both errors use MSE divided by fixed training-set feature
standard deviations. There are 17 one-step pairs: current groups 3–19 and
future groups 4–20. Each action consists of four ordered frame deltas.

## Phase A

`train.py` and `train_predictor.py` now run the same GRU training implementation.
The default configuration no longer trains a Teacher. Reuse an existing
trained Teacher bundle: loading it extracts only V-JEPA and Aggregator weights.
No RSSM, stochastic state, reconstruction head or KL loss is instantiated.
G remains frozen throughout Stage 2.

Cache training and held-out targets separately from manifests of captured
records containing `teacher_batch` and `task`:

```bash
python -m pefm.training.train cache \
  --teacher-bundle TEACHER.pt --train-manifest TRAIN_CAPTURES/manifest.jsonl \
  --output TRAIN_CACHE
python -m pefm.training.train cache \
  --teacher-bundle TEACHER.pt --train-manifest VALID_CAPTURES/manifest.jsonl \
  --output VALID_CACHE
python -m pefm.training.train --config pefm/training/config/train.yaml \
  --train-manifest TRAIN_CACHE/manifest.jsonl \
  --valid-manifest VALID_CACHE/manifest.jsonl --output outputs/stage2/predictor
```

CLI arguments override YAML values. `latest.pt` includes optimizer state and
can be resumed with `--resume`; `predictor_best.pt` is consumed by Phase B.
SIGUSR1 saves an interrupted checkpoint after the current batch. Old RSSM
optimizer checkpoints cannot resume this GRU training. To fit the matched
no-action control, rerun Phase A with `--no-actions` and a separate output
directory. `--horizon 2` or `4` trains open-loop transitions after the one-step
experiment.

Use at least two held-out trajectories per task for the action shuffle
diagnostic. Train/validation manifests must contain disjoint full trajectories.
The visual encoder processes individual frames and G processes groups
independently, so current targets do not read later frames.

## Phases B and C

```bash
python -m pefm.training.train_adapter \
  --teacher-bundle TEACHER.pt \
  --predictor-bundle outputs/stage2/predictor/predictor_best.pt \
  --train-manifest TRAIN_CAPTURES/manifest.jsonl \
  --valid-manifest VALID_CAPTURES/manifest.jsonl \
  --dit-dim 3072 --selected-block 15 \
  --noise-sampling '{"min_timestep_boundary":0.0,"max_timestep_boundary":1.0}' \
  --output outputs/stage2/adapter
```

Use the actual DiT dimension, captured block and noise distribution. Phase C
loads `TEACHER.pt` and `adapter_best.pt` through `load_stage2_distiller` in BWM.
Keep the same Teacher file across A/B/C: its file hash identifies the fixed
target space. The helper `export_frozen_evaluator` can extract a visual-only
bundle from an old verified checkpoint; select that file before building
caches if using it, since conversion changes the hash.

`load_frozen_evaluator` remains a compatibility name for the visual-only loader;
its output contains embeddings and visual tokens. The old prior/posterior
evaluator API has been removed from this Stage 2 branch.
