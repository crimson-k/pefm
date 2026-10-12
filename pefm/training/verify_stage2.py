"""Stage 2 各阶段的实测门槛；未通过时返回非零状态，停止后续训练。"""

import argparse
import json
import math
from pathlib import Path

import torch

from pefm.export.stage2 import load_predictor, load_stage2_distiller, teacher_identity
from .train_adapter import loader, run_adapter_epoch, _move
from .train_predictor import evaluate, motion_probe, targets_at_horizon


def write_report(root, phase, metrics, checks):
    passed = all(checks.values())
    report = {"phase": phase, "passed": passed, "checks": checks, "metrics": metrics}
    (root / "verifications" / phase).mkdir(parents=True, exist_ok=True)
    (root / "verifications" / phase / f"gate_{phase}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(2)


def verify_data(root, device):
    from pefm.export import load_frozen_visual_teacher
    from pefm.models.vjepa_adapter import align_row_major_tokens

    preflight = json.loads((root / "preflight.json").read_text())
    checks, summary, fingerprints, targets = {}, {}, set(), {}
    for split, metadata_split in (("train", "train"), ("valid", "test")):
        metadata = Path(preflight["dataset"]) / f"metadata_{metadata_split}.jsonl"
        source = [json.loads(line) for line in metadata.read_text().splitlines()]
        records = list(loader(root / "embeddings" / split / "manifest.jsonl"))
        targets[split] = records
        captures = list((root / "hidden/block_15" / split).glob("sample_*.pt"))
        assert len(source) == len(records) == len(captures)
        bins = [0, 0, 0]
        paired = []
        for index, target in enumerate(records):
            path = root / "hidden/block_15" / split / f"sample_{index:06d}.pt"
            raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            assert raw["sample_index"] == target["sample_index"] == index
            assert raw["source_video"] == target["source_video"] == source[index]["video"]
            assert raw["model_paths"] == preflight["dit_initialization"]
            assert raw["model_seed"] == 0
            assert raw["frame_indices"] == target["frame_indices"]
            assert torch.equal(raw["teacher_batch"]["action_segments"], target["action_segments"])
            assert raw["hidden"].shape == (1, 1008, 3072)
            assert raw["teacher_batch"]["rgb"].shape == (1, 1, 3, 81, 192, 256)
            assert target["embed"].shape == (1, 21, 256)
            assert torch.isfinite(raw["hidden"]).all() and torch.isfinite(target["embed"]).all()
            assert raw["noise_sampling"] == {"min_timestep_boundary": 0., "max_timestep_boundary": 1.}
            fingerprints.add(raw["action_encoder_sha256"])
            bins[min(int(float(raw["noise_sigma"]) * 3), 2)] += 1
            paired.append({"path": str(path), "target_path": str(root / "embeddings" / split / path.name)})
        (root / f"adapter_{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in paired))
        summary[split] = {"samples": len(source), "sigma_bins": bins}
        checks[f"{split}_noise_coverage"] = min(bins) >= 10

    checks["same_action_initialization"] = len(fingerprints) == 1
    checks["source_shapes_actions_finite"] = True
    summary["motion_probe"] = motion_probe(targets["train"], targets["valid"])
    probe = summary["motion_probe"]
    checks["motion_is_readable"] = (probe["position_mse"] < probe["position_mean_mse"] or
                                    probe["displacement_mse"] < probe["displacement_mean_mse"])
    # V-JEPA 在此实现中逐帧编码；用截断视频实测当前目标不依赖未来帧。
    teacher_dtype = targets["train"][0]["teacher_dtype"]
    assert all(record["teacher_dtype"] == teacher_dtype and record["embed"].dtype == getattr(torch, teacher_dtype)
               for records in targets.values() for record in records)
    teacher = load_frozen_visual_teacher(root / "visual_teacher.pt", device=device, dtype=getattr(torch, teacher_dtype))
    raw = torch.load(root / "hidden/block_15/valid/sample_000000.pt", weights_only=False)
    target = torch.load(root / "embeddings/valid/sample_000000.pt", weights_only=False)["embed"]
    batch = raw["teacher_batch"]
    with torch.no_grad():
        tokens = teacher.vjepa_adapter(batch["rgb"][:, :, :, :13].to(device),
                                      batch["group_ids"][:, :13].to(device))
        grid = align_row_major_tokens(tokens, teacher.rgb_grid, teacher.aligned_grid)
        prefix = teacher.token_aggregator(grid.flatten(2, 4).contiguous()).cpu()
    summary["prefix_max_abs_error"] = float((prefix - target[:, :4]).abs().max())
    checks["clean_start_has_no_future_frames"] = torch.allclose(prefix, target[:, :4], rtol=1e-4, atol=1e-5)
    write_report(root, "data", summary, checks)


def verify_predictor(root, device):
    records = list(loader(root / "embeddings/valid/manifest.jsonl"))
    bundle = torch.load(root / "predictor/predictor_best.pt", map_location="cpu", weights_only=False)
    predictor = load_predictor(bundle, device).eval()
    no_actions = load_predictor(root / "verifications/predictor/no-actions/predictor_best.pt", device).eval()
    metrics = {f"horizon_{k}": evaluate(predictor, records, k) for k in (1, 2, 4)}
    metrics["no_actions"] = evaluate(no_actions, records, 1)["prediction"]
    metrics["motion_probe"] = bundle["motion_probe"]
    one, probe = metrics["horizon_1"], metrics["motion_probe"]
    checks = {f"beats_{key}": one["prediction"] < .99 * one[key]
              for key in ("mean", "persistence", "time_shift", "task_shuffle")}
    checks["beats_matched_no_action_model"] = one["prediction"] < .99 * metrics["no_actions"]
    checks["motion_is_readable"] = (probe["position_mse"] < probe["position_mean_mse"] or
                                    probe["displacement_mse"] < probe["displacement_mean_mse"])
    checks["finite_multistep_diagnostics"] = all(math.isfinite(value)
        for k in (1, 2, 4) for value in metrics[f"horizon_{k}"].values())
    write_report(root, "predictor", metrics, checks)


def adapter_checks(metrics, tolerance, prefix=""):
    get = lambda name: metrics[prefix + name]
    clean, estimated = get("gt_start_pred_mse"), get("estimated_start_pred_mse")
    return {
        prefix + "future_error_close": estimated <= (1 + tolerance) * clean,
        prefix + "prediction_outputs_close": get("prediction_agreement_mse") <= tolerance * clean,
        prefix + "beats_persistence": estimated < get("estimated_persistence_mse"),
        prefix + "beats_wrong_actions": estimated < get("estimated_wrong_action_mse"),
        prefix + "paired_alignment_better": get("paired_mse") < get("shuffled_mse"),
    }


def verify_adapter(root, device, tolerance, adapter_bundle):
    model = load_stage2_distiller(root / "visual_teacher.pt", adapter_bundle,
                                 device, torch.float32, dit_dim=3072, selected_block=15,
                                 noise_sampling={"min_timestep_boundary": 0., "max_timestep_boundary": 1.})
    model.teacher_sha256 = teacher_identity(root / "visual_teacher.pt")
    metrics = run_adapter_epoch(model, loader(root / "adapter_valid.jsonl"), device, 1.,
                                selected_block=15,
                                noise_sampling={"min_timestep_boundary": 0., "max_timestep_boundary": 1.})
    checks = adapter_checks(metrics, tolerance)
    for index in range(3):
        checks.update(adapter_checks(metrics, tolerance, f"sigma_bin_{index}_"))
    checks["finite"] = all(math.isfinite(value) for value in metrics.values())
    metrics["closeness_tolerance"] = tolerance
    write_report(root, "adapter", metrics, checks)


@torch.no_grad()
def compare_action_response(root, device, adapter_bundle):
    """固定读出起点，只改变 GRU 动作；响应幅度不作为预测合理性的通过门槛。"""
    import numpy as np

    identity = teacher_identity(root / 'visual_teacher.pt')
    model = load_stage2_distiller(root / 'visual_teacher.pt', adapter_bundle,
                                 device, torch.float32, dit_dim=3072, selected_block=15,
                                 noise_sampling={'min_timestep_boundary': 0., 'max_timestep_boundary': 1.})
    model.eval().requires_grad_(False)
    model.visual.cpu()
    records, clean, recovered, actions = [], [], [], []
    for index, raw in enumerate(loader(root / 'adapter_valid.jsonl')):
        assert raw['teacher_sha256'] == identity and raw['selected_block'] == 15
        batch = _move(raw, device)
        result = model(batch['hidden'], batch['dit_grid'], batch['teacher_batch'],
                       batch['noise_sigma'], batch['valid_len'],
                       target_embedding=batch['target_embedding'])
        clean.append(result['target'])
        recovered.append(result['estimated'])
        actions.append(batch['teacher_batch']['action_segments'])
        records.append({'sample_index': raw['sample_index'], 'source_video': raw['source_video'],
                        'task': raw['task'], 'sigma': float(batch['noise_sigma']),
                        'frame_indices': raw['frame_indices']})
        if (index + 1) % 25 == 0:
            print(f'action response: recovered {index + 1} windows', flush=True)
    clean, recovered, actions = (torch.cat(values) for values in (clean, recovered, actions))
    # donor 必须来自同任务的另一个源视频，避免重叠窗口被误当作跨轨迹动作。
    partners = [next(j for offset in range(1, len(records))
                     for j in [(i + offset) % len(records)]
                     if records[j]['task'] == record['task']
                     and records[j]['source_video'] != record['source_video'])
                for i, record in enumerate(records)]
    for record, partner in zip(records, partners):
        record['donor_sample_index'] = records[partner]['sample_index']
        record['donor_source_video'] = records[partner]['source_video']
    predictor = model.predictor
    rows = []
    for horizon in (1, 2, 4):
        target = targets_at_horizon(clean, horizon)
        baseline = {}
        correct = {}
        for condition, start in (('clean', clean), ('recovered', recovered)):
            correct[condition] = predictor.pairs(start, actions, horizon)
            baseline[condition] = {
                'correct_mse': predictor.distance(correct[condition], target).mean((1, 2)),
                'persistence_mse': predictor.distance(
                    start[:, 3:21-horizon, None].expand_as(target), target).mean((1, 2)),
                'mean_mse': predictor.distance(predictor.mean.expand_as(target), target).mean((1, 2)),
            }
        for control, wrong in (('time_shift', actions.roll(1, 1)),
                               ('same_task_other_video', actions[partners])):
            for alpha in (0., .25, .5, 1.):
                changed = actions + alpha * (wrong - actions)
                values = {}
                for condition, start in (('clean', clean), ('recovered', recovered)):
                    prediction = predictor.pairs(start, changed, horizon)
                    metrics = dict(baseline[condition])
                    metrics['response_mse'] = predictor.distance(prediction, correct[condition]).mean((1, 2))
                    metrics['wrong_mse'] = predictor.distance(prediction, target).mean((1, 2))
                    metrics['error_increase'] = metrics['wrong_mse'] - metrics['correct_mse']
                    metrics['prediction_gain'] = metrics['persistence_mse'] - metrics['correct_mse']
                    values[condition] = {k: v.cpu().tolist() for k, v in metrics.items()}
                for i, record in enumerate(records):
                    row = {**record, 'horizon': horizon, 'control': control, 'alpha': alpha}
                    for condition in values:
                        row.update({f'{condition}_{k}': v[i] for k, v in values[condition].items()})
                    assert all(math.isfinite(v) for k, v in row.items()
                               if k.startswith(('clean_', 'recovered_')))
                    if alpha == 0:
                        assert row['clean_response_mse'] == row['recovered_response_mse'] == 0
                    rows.append(row)

    # 描述性区间按源视频重采样，保持六个任务的轨迹数量；donor 配对固定。
    rng = np.random.default_rng(0)
    summaries = []
    for horizon in (1, 2, 4):
        for control in ('time_shift', 'same_task_other_video'):
            for alpha in (0., .25, .5, 1.):
                subset = [r for r in rows if (r['horizon'], r['control'], r['alpha']) ==
                          (horizon, control, alpha)]
                scopes = [('all', subset)]
                scopes += [(f'sigma_bin_{b}', [r for r in subset if min(int(r['sigma']*3), 2) == b])
                           for b in range(3)]
                scopes += [(f'task_{task}', [r for r in subset if r['task'] == task])
                           for task in sorted({r['task'] for r in subset})]
                keys = [k for k in subset[0] if k.startswith(('clean_', 'recovered_'))]
                for scope, selected in scopes:
                    matrix = np.array([[r[k] for k in keys] for r in selected])
                    means = dict(zip(keys, matrix.mean(0).tolist()))
                    for condition in ('clean', 'recovered'):
                        means[f'{condition}_response_rmse'] = math.sqrt(means[f'{condition}_response_mse'])
                        means[f'{condition}_relative_gain'] = means[f'{condition}_prediction_gain'] / means[f'{condition}_persistence_mse']
                    denom = means['clean_response_rmse']
                    means['response_rmse_ratio_recovered_to_clean'] = means['recovered_response_rmse'] / denom if denom else None
                    summary = {'horizon': horizon, 'control': control, 'alpha': alpha, 'scope': scope,
                               'windows': len(selected), 'videos': len({r['source_video'] for r in selected}),
                               'metrics': means}
                    if scope == 'all':
                        videos = sorted({r['source_video'] for r in selected})
                        video_rows = [np.array([i for i, r in enumerate(selected) if r['source_video'] == v]) for v in videos]
                        strata = [[j for j, v in enumerate(videos)
                                   if selected[video_rows[j][0]]['task'] == task]
                                  for task in sorted({r['task'] for r in selected})]
                        boots = []
                        for _ in range(2000):
                            picked = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
                            indices = np.concatenate([video_rows[j] for j in picked])
                            boots.append(matrix[indices].mean(0))
                        low, high = np.percentile(boots, [2.5, 97.5], axis=0)
                        summary['pointwise_cluster_bootstrap_95ci'] = {
                            k: [float(lo), float(hi)] for k, lo, hi in zip(keys, low, high)}
                    summaries.append(summary)
    output = root / 'verifications/action_response'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'action_response_rows.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = {'adapter_bundle': str(adapter_bundle), 'teacher_sha256': identity,
              'windows': len(records), 'videos': len({r['source_video'] for r in records}),
              'conditions': 'Fixed original-action DiT hidden; perturb GRU actions only; K=1 primary, K=2/4 exploratory.',
              'uncertainty': '2000 source-video bootstrap draws stratified by task; pointwise descriptive intervals conditional on fixed donors; no equivalence test.',
              'summaries': summaries}
    (output / 'action_response.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'completed': True, 'windows': len(records), 'videos': report['videos']}), flush=True)


def verify_gradient(root, device, adapter_bundle):
    from scripts.train import wan_parser, build_training_module
    from wan_video_action.parsers import merge_yaml_and_args, resolve_data_keys, prepare_model_config
    from wan_video_action.data import build_train_dataset
    from wan_video_action.loss import flow_match_forward
    from wan_video_action.utils import set_global_seed
    from wan_video_action.stage2_semantic import build_teacher_batch

    bwm = Path('/data1/fangxuebin/dev/wm/stage2-B/boundless-world-model')
    preflight = json.loads((root / "preflight.json").read_text())
    argv = ['--config', str(bwm / 'configs/train/train_wan22_ti2v_5b_action_adaln.yaml'),
            '--model_config_path', str(bwm / 'configs/model/wan2_2_ti2v_5b.yaml'),
            '--model_paths', preflight['dit_initialization'], '--dataset_base_path', preflight['dataset'],
            '--dataset_metadata_path', preflight['dataset'] + '/metadata_train.jsonl',
            '--action_stat_path', preflight['dataset'] + '/stat.json', '--height', '192', '--width', '256',
            '--seed', '0', '--trainable', 'dit', 'action_encoder', '--use_gradient_checkpointing',
            '--evaluator_loss_weight', '0', '--stage2_loss_weight', '0.1',
            '--stage2_teacher_path', str(root / 'visual_teacher.pt'),
            '--stage2_adapter_path', str(adapter_bundle),
            '--stage2_target_cache', str(root / 'embeddings/phase_c_targets_bf16.pt'),
            '--stage2_selected_block', '15']
    parser = wan_parser()
    args = parser.parse_args(argv)
    args = merge_yaml_and_args(args.config, parser, args, argv)
    args = resolve_data_keys(args, stage='train')
    set_global_seed(args.seed, deterministic=args.deterministic)
    model = build_training_module(args, prepare_model_config(args), torch.device(device))
    model.train()
    pipe = model.pipe
    reference = torch.load(root / 'hidden/block_15/train/action_encoder_initial.pt', weights_only=False)
    checks = {'same_action_initialization': all(torch.equal(value.cpu(), reference[key])
                                               for key, value in pipe.action_encoder.state_dict().items())}
    auxiliary = pipe.stage2_distiller
    before = {key: value.detach().cpu().clone() for key, value in auxiliary.state_dict().items()}
    sample = build_train_dataset(args, repeat=1)[0]
    inputs = model.transfer_data_to_device(model.get_pipeline_inputs(sample), pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        inputs = pipe.unit_runner(unit, pipe, *inputs)
    shared, positive, _ = inputs
    output = flow_match_forward(pipe, return_hidden_block=15, timestep_id=500, **shared, **positive)
    hidden = output['model_output']['hidden']
    hidden.retain_grad()
    batch = build_teacher_batch(shared['stage2_rgb'], shared['stage2_eef'])
    with torch.autocast(device_type='cuda', enabled=False):
        result = auxiliary(hidden, output['model_output']['dit_grid'], batch,
                           output['sigma'], output['model_output']['valid_len'],
                           target_embedding=shared['stage2_target_embedding'])
    result['pred'].backward()
    gradients = {name: float(parameter.grad.float().norm())
                 for name, parameter in model.named_parameters() if parameter.grad is not None}
    checks.update({
        'hidden_gradient_finite_nonzero': bool(torch.isfinite(hidden.grad).all() and hidden.grad.float().norm() > 0),
        'dit_gradient_nonzero': any(value > 0 for name, value in gradients.items() if name.startswith('pipe.dit.')),
        'action_encoder_trainable': all(p.requires_grad for p in pipe.action_encoder.parameters()),
        'action_gradient_nonzero': any(value > 0 for name, value in gradients.items()
                                       if name.startswith('pipe.action_encoder.')),
        'all_gradients_finite': all(math.isfinite(value) for value in gradients.values()),
        'auxiliary_frozen': all(not p.requires_grad and p.grad is None for p in auxiliary.parameters()),
        'auxiliary_weights_unchanged': all(torch.equal(value.cpu(), before[key])
                                           for key, value in auxiliary.state_dict().items()),
    })
    metrics = {'pred_mse': float(result['pred'].detach()), 'sigma': float(output['sigma']),
               'hidden_grad_norm': float(hidden.grad.float().norm()),
               'dit_gradient_parameters': sum(name.startswith('pipe.dit.') for name in gradients),
               'action_gradient_norm': sum(value for name, value in gradients.items()
                                            if name.startswith('pipe.action_encoder.'))}
    write_report(root, 'gradient', metrics, checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('data', 'predictor', 'adapter', 'gradient', 'actions'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--closeness-tolerance', type=float, default=.2)
    parser.add_argument('--adapter-bundle', type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    adapter_bundle = (args.adapter_bundle.resolve() if args.adapter_bundle is not None
                      else root / 'adapter/block_15/adapter_best.pt')
    if args.phase == 'data':
        verify_data(root, args.device)
    elif args.phase == 'predictor':
        verify_predictor(root, args.device)
    elif args.phase == 'adapter':
        verify_adapter(root, args.device, args.closeness_tolerance, adapter_bundle)
    elif args.phase == 'actions':
        compare_action_response(root, args.device, adapter_bundle)
    else:
        verify_gradient(root, args.device, adapter_bundle)


if __name__ == '__main__':
    main()
