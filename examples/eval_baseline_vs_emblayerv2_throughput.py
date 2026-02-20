# -*- coding: utf-8 -*-
import argparse
import csv
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import benchmark_dinemblayer_infer as bench


@dataclass
class IterMetrics:
    avg_ms: float
    p99_ms: float
    throughput_qps: float
    h2d_only_ms: float
    h2d_pure_ms: float
    h2d_bubble_ms: float
    h2d_bubble_ratio_pct: float


@dataclass
class UtilMetrics:
    gpu_util_avg_pct: Optional[float]
    sm_util_avg_pct: Optional[float]


def _parse_batch_sizes(text: str) -> List[int]:
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError('batch_sizes cannot be empty')
    return values


def _build_infer_models(model, host_data_meta, device: str):
    infer_models = {}
    input_spans = [model.feature_index[name] for name in model.feature_index]
    input_spans = sorted(input_spans, key=lambda x: x[0])
    infer_models['baseline'] = bench.DINBaselinePerInputSlice(
        model=model,
        input_spans=input_spans,
    ).to(device).eval()
    infer_models['baseline_raw'] = model

    seq_ext = bench.load_emblayer_seq_extension()
    vec_ext = bench.load_emblayer_vec_extension()

    infer_models['joint_v2'] = bench.DINJointInferV2(
        model=model,
        seq_feature_info=host_data_meta['seq_feature_info'],
        input_dim=host_data_meta['input_dim'],
        num_seq_per_sample=host_data_meta['num_seq_per_sample'],
        max_seq_len=host_data_meta['max_seq_len'],
        emblayer_seq_ext=seq_ext,
        emblayer_vec_ext=vec_ext,
    ).to(device).eval()
    return infer_models


def _run_mode_iters(mode: str,
                    model,
                    infer_models,
                    host_data,
                    device: str,
                    iters: int,
                    warmup: int) -> List[float]:
    with torch.no_grad():
        for _ in range(warmup):
            dev_inputs = bench._prepare_inputs(mode, host_data, device, scheduler_side_concat=False)
            _ = bench._forward_mode(mode, model, infer_models, dev_inputs, scheduler_side_concat=False)

    bench._sync_if_cuda(device)
    latencies = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            dev_inputs = bench._prepare_inputs(mode, host_data, device, scheduler_side_concat=False)
            _ = bench._forward_mode(mode, model, infer_models, dev_inputs, scheduler_side_concat=False)
            bench._sync_if_cuda(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
    return latencies


def _calc_iter_metrics(mode: str,
                       model,
                       infer_models,
                       host_data,
                       device: str,
                       iters: int,
                       warmup: int) -> IterMetrics:
    latencies = _run_mode_iters(mode, model, infer_models, host_data, device, iters, warmup)
    avg_ms = float(np.mean(latencies))
    p99_ms = float(np.percentile(latencies, 99))
    throughput_qps = host_data['full'].shape[0] * 1000.0 / avg_ms

    h2d_only_ms = bench.h2d_only_ms(mode, host_data, device, iters, warmup, scheduler_side_concat=False)
    h2d_pure_ms = bench.pure_h2d_only_ms(mode, host_data, device, iters, warmup, scheduler_side_concat=False)
    h2d_bubble_ms = max(h2d_only_ms - h2d_pure_ms, 0.0)
    h2d_bubble_ratio_pct = 0.0 if h2d_only_ms <= 0 else h2d_bubble_ms / h2d_only_ms * 100.0

    return IterMetrics(
        avg_ms=avg_ms,
        p99_ms=p99_ms,
        throughput_qps=throughput_qps,
        h2d_only_ms=h2d_only_ms,
        h2d_pure_ms=h2d_pure_ms,
        h2d_bubble_ms=h2d_bubble_ms,
        h2d_bubble_ratio_pct=h2d_bubble_ratio_pct,
    )


def _collect_dmon_util(run_fn) -> UtilMetrics:
    cmd = ['nvidia-smi', 'dmon', '-s', 'u', '-d', '1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    stop_flag = {'stop': False}

    def _reader():
        if proc.stdout is None:
            return
        for line in proc.stdout:
            lines.append(line.rstrip('\n'))
            if stop_flag['stop']:
                break

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        run_fn()
    finally:
        stop_flag['stop'] = True
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        t.join(timeout=2)

    gpu_vals = []
    sm_vals = []
    row_pattern = re.compile(r'^\s*\d+\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$')
    for line in lines:
        m = row_pattern.match(line)
        if not m:
            continue
        gpu_vals.append(float(m.group(1)))
        sm_vals.append(float(m.group(2)))

    gpu_avg = None if not gpu_vals else float(np.mean(gpu_vals))
    sm_avg = None if not sm_vals else float(np.mean(sm_vals))
    return UtilMetrics(gpu_util_avg_pct=gpu_avg, sm_util_avg_pct=sm_avg)


def _select_best_under_p99(records: List[Dict], p99_target_ms: float) -> Optional[Dict]:
    valid = [r for r in records if r['p99_ms'] <= p99_target_ms]
    if not valid:
        return None
    return max(valid, key=lambda x: x['throughput_qps'])


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(v, digits=4):
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:.{digits}f}'
    return str(v)


def _write_report(report_path: str,
                  p99_target_ms: float,
                  sweep_rows: List[Dict],
                  best_rows: List[Dict]):
    lines = []
    lines.append('# Baseline vs EmblayerV2 大吞吐测试报告')
    lines.append('')
    lines.append(f'- p99 约束: <= {p99_target_ms:.1f} ms')
    lines.append('- 对比模式: baseline vs emblayerV2(joint_v2)')
    lines.append('')

    lines.append('## p99<=100ms 下最大吞吐结果')
    lines.append('')
    lines.append('| Mode | Max Throughput (QPS) | Batch Size | Avg Latency (ms) | p99 (ms) | H2D-only (ms) | Pure H2D (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in best_rows:
        lines.append(
            f"| {row['mode']} | {_fmt(row['throughput_qps'])} | {row['batch_size']} | {_fmt(row['avg_ms'])} | {_fmt(row['p99_ms'])} | "
            f"{_fmt(row['h2d_only_ms'])} | {_fmt(row['h2d_pure_ms'])} | {_fmt(row['h2d_bubble_ms'])} | {_fmt(row['h2d_bubble_ratio_pct'])} | "
            f"{_fmt(row.get('sm_util_avg_pct'))} | {_fmt(row.get('gpu_util_avg_pct'))} |"
        )

    lines.append('')
    lines.append('## Batch Sweep 明细')
    lines.append('')
    lines.append('| Mode | Batch | Avg (ms) | p99 (ms) | QPS | H2D-only (ms) | Pure H2D (ms) | Bubble (ms) | Bubble Ratio (%) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for row in sweep_rows:
        lines.append(
            f"| {row['mode']} | {row['batch_size']} | {_fmt(row['avg_ms'])} | {_fmt(row['p99_ms'])} | {_fmt(row['throughput_qps'])} | "
            f"{_fmt(row['h2d_only_ms'])} | {_fmt(row['h2d_pure_ms'])} | {_fmt(row['h2d_bubble_ms'])} | {_fmt(row['h2d_bubble_ratio_pct'])} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Large-throughput baseline vs emblayerV2 evaluator under p99 SLA')
    parser.add_argument('--batch_sizes', type=str, default='256,512,1024,2048,4096')
    parser.add_argument('--p99_target_ms', type=float, default=100.0)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--monitor_iters', type=int, default=500)
    parser.add_argument('--monitor_warmup', type=int, default=100)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--avg_seq_len', type=float, default=8.0)
    parser.add_argument('--num_sparse', type=int, default=200)
    parser.add_argument('--num_seq', type=int, default=100)
    parser.add_argument('--user_sparse_count', type=int, default=100)
    parser.add_argument('--even_vocab_size', type=int, default=4096)
    parser.add_argument('--odd_vocab_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out_dir', type=str, default='./eval_baseline_vs_emblayerv2_outputs')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.user_sparse_count < 0 or args.user_sparse_count > args.num_sparse:
        raise ValueError('--user_sparse_count must be in [0, num_sparse]')

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    mode_name = {
        'baseline': 'baseline',
        'joint_v2': 'emblayerV2',
    }

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'
    if device == 'cpu':
        raise RuntimeError('This script is intended for GPU throughput evaluation. Please run on CUDA.')

    model, feature_columns = bench.build_din_model(
        device=device,
        max_seq_len=args.max_seq_len,
        num_sparse=args.num_sparse,
        num_seq=args.num_seq,
        even_vocab_size=args.even_vocab_size,
        odd_vocab_size=args.odd_vocab_size,
    )
    model.user_sparse_count_for_emblayer_v2 = int(args.user_sparse_count)

    host_data_meta = bench.build_inputs(
        model=model,
        feature_columns=feature_columns,
        batch_size=batch_sizes[0],
        max_seq_len=args.max_seq_len,
        avg_seq_len=args.avg_seq_len,
        seed=args.seed,
    )
    infer_models = _build_infer_models(model, host_data_meta, device)

    print('device:', device)
    print('modes:', 'baseline, emblayerV2(joint_v2)')
    print('batch_sizes:', batch_sizes)
    print('p99_target_ms:', args.p99_target_ms)

    sweep_rows = []
    per_mode_rows: Dict[str, List[Dict]] = {'baseline': [], 'joint_v2': []}

    for bsz in batch_sizes:
        host_data = bench.build_inputs(
            model=model,
            feature_columns=feature_columns,
            batch_size=bsz,
            max_seq_len=args.max_seq_len,
            avg_seq_len=args.avg_seq_len,
            seed=args.seed,
        )
        for mode in ('baseline', 'joint_v2'):
            metrics = _calc_iter_metrics(
                mode=mode,
                model=model,
                infer_models=infer_models,
                host_data=host_data,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
            )
            row = {
                'mode': mode_name[mode],
                'mode_key': mode,
                'batch_size': bsz,
                'avg_ms': metrics.avg_ms,
                'p99_ms': metrics.p99_ms,
                'throughput_qps': metrics.throughput_qps,
                'h2d_only_ms': metrics.h2d_only_ms,
                'h2d_pure_ms': metrics.h2d_pure_ms,
                'h2d_bubble_ms': metrics.h2d_bubble_ms,
                'h2d_bubble_ratio_pct': metrics.h2d_bubble_ratio_pct,
            }
            sweep_rows.append(row)
            per_mode_rows[mode].append(row)
            print(f"mode={mode_name[mode]} batch={bsz} avg_ms={metrics.avg_ms:.4f} p99_ms={metrics.p99_ms:.4f} qps={metrics.throughput_qps:.2f}")

    best_rows = []
    for mode in ('baseline', 'joint_v2'):
        best = _select_best_under_p99(per_mode_rows[mode], args.p99_target_ms)
        if best is None:
            best_rows.append({
                'mode': mode_name[mode],
                'batch_size': 'N/A',
                'avg_ms': None,
                'p99_ms': None,
                'throughput_qps': None,
                'h2d_only_ms': None,
                'h2d_pure_ms': None,
                'h2d_bubble_ms': None,
                'h2d_bubble_ratio_pct': None,
                'gpu_util_avg_pct': None,
                'sm_util_avg_pct': None,
            })
            continue

        bsz = int(best['batch_size'])
        host_data = bench.build_inputs(
            model=model,
            feature_columns=feature_columns,
            batch_size=bsz,
            max_seq_len=args.max_seq_len,
            avg_seq_len=args.avg_seq_len,
            seed=args.seed,
        )

        def _run_for_monitor():
            _ = _run_mode_iters(
                mode=mode,
                model=model,
                infer_models=infer_models,
                host_data=host_data,
                device=device,
                iters=args.monitor_iters,
                warmup=args.monitor_warmup,
            )

        util = _collect_dmon_util(_run_for_monitor)

        out = dict(best)
        out['gpu_util_avg_pct'] = util.gpu_util_avg_pct
        out['sm_util_avg_pct'] = util.sm_util_avg_pct
        best_rows.append(out)

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    sweep_csv = os.path.join(args.out_dir, 'throughput_sweep.csv')
    best_csv = os.path.join(args.out_dir, 'best_under_p99.csv')
    report_md = os.path.join(args.out_dir, 'offline_report.md')

    _write_csv(
        sweep_csv,
        sweep_rows,
        fieldnames=['mode', 'mode_key', 'batch_size', 'avg_ms', 'p99_ms', 'throughput_qps', 'h2d_only_ms', 'h2d_pure_ms', 'h2d_bubble_ms', 'h2d_bubble_ratio_pct'],
    )
    _write_csv(
        best_csv,
        best_rows,
        fieldnames=['mode', 'batch_size', 'avg_ms', 'p99_ms', 'throughput_qps', 'h2d_only_ms', 'h2d_pure_ms', 'h2d_bubble_ms', 'h2d_bubble_ratio_pct', 'sm_util_avg_pct', 'gpu_util_avg_pct'],
    )
    _write_report(report_md, args.p99_target_ms, sweep_rows, best_rows)

    print('saved:', sweep_csv)
    print('saved:', best_csv)
    print('saved:', report_md)


if __name__ == '__main__':
    main()


# cd /root/DeepCTR-Torch/examples
# /root/autodl-tmp/myenv/bin/python eval_baseline_vs_emblayerv2_throughput.py \
#   --batch_sizes 256,512,1024,2048,4096 \
#   --p99_target_ms 100 \
#   --iters 300 --warmup 50 \
#   --monitor_iters 500 --monitor_warmup 100 \
#   --max_seq_len 128 --avg_seq_len 8 \
#   --num_sparse 200 --num_seq 100 \
#   --user_sparse_count 100 \
#   --even_vocab_size 4096 --odd_vocab_size 64 \
#   --out_dir ./eval_baseline_vs_emblayerv2_outputs