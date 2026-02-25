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
class RunResult:
    mode: str
    arrival_rate: float
    duration_s: float
    total_requests: int
    throughput_qps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    avg_batch_size: float
    max_batch_size: int
    avg_queue_ms: float
    avg_h2d_only_ms: float
    avg_h2d_pure_ms: float
    h2d_bubble_ms: float
    h2d_bubble_ratio_pct: float
    sm_util_avg_pct: Optional[float]
    gpu_util_avg_pct: Optional[float]
    avg_h2d_total_numel: float
    avg_h2d_data_numel: float
    avg_h2d_meta_numel: float
    avg_concat_ms: float


MODE_DISPLAY = {
    'baseline': 'baseline',
    'multislice': 'multislice',
    'joint': 'emblayerV1',
    'joint_v2': 'emblayerV2',
}

MODE_ALIASES = {
    'baseline': 'baseline',
    'multislice': 'multislice',
    'emblayer': 'joint',
    'emblayerv1': 'joint',
    'emblayer_v1': 'joint',
    'v1': 'joint',
    'emblayerv2': 'joint_v2',
    'emblayer_v2': 'joint_v2',
    'v2': 'joint_v2',
}

DEFAULT_MODE_ORDER = ('baseline', 'multislice', 'joint', 'joint_v2')


def _parse_float_list(text: str) -> List[float]:
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            parts = [p.strip() for p in token.split(':')]
            if len(parts) != 3:
                raise ValueError(f"invalid arrival_rates range token: '{token}', expected start:end:step")
            start = float(parts[0])
            end = float(parts[1])
            step = float(parts[2])
            if step <= 0:
                raise ValueError('arrival_rates range step must be > 0')
            if end < start:
                raise ValueError('arrival_rates range end must be >= start')
            cur = start
            while cur <= end + 1e-12:
                values.append(cur)
                cur += step
        else:
            values.append(float(token))
    if not values:
        raise ValueError('arrival_rates cannot be empty')
    return values


def _parse_int_list(text: str) -> List[int]:
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError('batch_sizes cannot be empty')
    return values


def _parse_modes(text: str) -> List[str]:
    if not text:
        return list(DEFAULT_MODE_ORDER)

    raw_tokens = [tok.strip() for tok in text.split(',') if tok.strip()]
    if not raw_tokens:
        return list(DEFAULT_MODE_ORDER)

    modes: List[str] = []
    seen = set()
    for token in raw_tokens:
        key = token.lower()
        if key not in MODE_ALIASES:
            valid = ', '.join(sorted(MODE_ALIASES))
            raise ValueError(f"unknown mode '{token}', expected one of: {valid}")
        mode = MODE_ALIASES[key]
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)

    return modes


def _build_infer_models(model, host_data_meta, device: str, modes: List[str]):
    infer_models = {}
    input_spans = [model.feature_index[name] for name in model.feature_index]
    input_spans = sorted(input_spans, key=lambda x: x[0])

    if 'baseline' in modes:
        infer_models['baseline'] = bench.DINBaselinePerInputSlice(
            model=model,
            input_spans=input_spans,
        ).to(device).eval()
    infer_models['baseline_raw'] = model

    multislice_ext = None
    if 'multislice' in modes:
        multislice_ext = bench.load_multislice_extension()
        infer_models['multislice'] = bench.DINMultiSliceInfer(
            model=model,
            compact_layout=host_data_meta['compact_layout'],
            starts=host_data_meta['starts'],
            span_lengths=host_data_meta['span_lengths'],
            multislice_ext=multislice_ext,
        ).to(device).eval()

    seq_ext = None
    vec_ext = None
    if any(m in ('joint', 'joint_v2') for m in modes):
        seq_ext = bench.load_emblayer_seq_extension()
    if 'joint' in modes or 'joint_v2' in modes:
        vec_ext = bench.load_emblayer_vec_extension()

    if 'joint' in modes:
        infer_models['joint'] = bench.DINJointInfer(
            model=model,
            seq_feature_info=host_data_meta['seq_feature_info'],
            input_dim=host_data_meta['input_dim'],
            num_seq_per_sample=host_data_meta['num_seq_per_sample'],
            max_seq_len=host_data_meta['max_seq_len'],
            emblayer_seq_ext=seq_ext,
            emblayer_vec_ext=vec_ext,
        ).to(device).eval()

    if 'joint_v2' in modes:
        infer_models['joint_v2'] = bench.DINJointInferV2(
            model=model,
            seq_feature_info=host_data_meta['seq_feature_info'],
            input_dim=host_data_meta['input_dim'],
            num_seq_per_sample=host_data_meta['num_seq_per_sample'],
            max_seq_len=host_data_meta['max_seq_len'],
            emblayer_seq_ext=seq_ext,
            emblayer_vec_ext=vec_ext,
        ).to(device).eval()

    missing = [m for m in modes if m not in infer_models]
    if missing:
        raise RuntimeError(f'missing infer models for modes: {missing}')
    return infer_models


def _run_infer(mode: str, model, infer_models, host_data, device: str):
    dev_inputs = bench._prepare_inputs(mode, host_data, device, scheduler_side_concat=False)
    return bench._forward_mode(mode, model, infer_models, dev_inputs, scheduler_side_concat=False)


def _collect_dmon_util(run_fn) -> Tuple[Optional[float], Optional[float]]:
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
    return sm_avg, gpu_avg


def _compute_h2d_cache(mode: str,
                        host_data,
                        device: str,
                        iters: int,
                        warmup: int) -> Tuple[float, float]:
    h2d_only = bench.h2d_only_ms(mode, host_data, device, iters, warmup, scheduler_side_concat=False)
    h2d_pure = bench.pure_h2d_only_ms(mode, host_data, device, iters, warmup, scheduler_side_concat=False)
    return h2d_only, h2d_pure


def _sample_seq_len(rng: np.random.Generator, avg_seq_len: float, max_seq_len: int) -> int:
    length = int(rng.poisson(lam=max(1.0, float(avg_seq_len))))
    if length < 1:
        length = 1
    if length > max_seq_len:
        length = max_seq_len
    return length


def _simulate_baseline_padding(seq_lengths: List[int], max_seq_len: int):
    if not seq_lengths:
        return
    batch_size = len(seq_lengths)
    batch_max = max(seq_lengths)
    if batch_max <= 0:
        return

    tmp = np.zeros((batch_size, batch_max), dtype=np.int32)
    for row, length in enumerate(seq_lengths):
        tmp[row, :length] = 1

    if batch_max < max_seq_len:
        full = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        full[:, :batch_max] = tmp


def _simulate(mode: str,
              model,
              infer_models,
              host_data_cache: Dict[int, Tuple[Dict, float, Dict]],
              device: str,
              arrival_rate: float,
              duration_s: float,
              max_batch_size: int,
              max_wait_ms: float,
              seed: int,
              h2d_cache: Dict[int, Tuple[float, float]]) -> RunResult:
    rng = np.random.default_rng(seed)
    queue = []
    batch_sizes = []
    latencies_ms = []
    queue_wait_ms = []
    h2d_only_samples = []
    h2d_pure_samples = []
    concat_ms_samples = []
    h2d_total_numel_samples = []
    h2d_data_numel_samples = []
    h2d_meta_numel_samples = []

    start = time.perf_counter()
    end = start + duration_s
    next_arrival = start + rng.exponential(1.0 / max(arrival_rate, 1e-9))

    def _run_loop():
        nonlocal next_arrival
        while True:
            now = time.perf_counter()
            if now >= end and not queue:
                break

            while now >= next_arrival and now < end:
                seq_len = _sample_seq_len(rng, model.avg_seq_len, model.max_seq_len)
                queue.append((next_arrival, seq_len))
                next_arrival = next_arrival + rng.exponential(1.0 / max(arrival_rate, 1e-9))

            if not queue:
                time.sleep(min(0.001, max(0.0, next_arrival - now)))
                continue

            oldest = queue[0][0]
            if len(queue) < max_batch_size and (now - oldest) < max_wait_ms / 1000.0 and now < end:
                time.sleep(0.0005)
                continue

            batch_size = min(len(queue), max_batch_size)
            batch_items = [queue.pop(0) for _ in range(batch_size)]
            batch_arrivals = [item[0] for item in batch_items]
            batch_lengths = [item[1] for item in batch_items]

            if mode == 'baseline':
                _simulate_baseline_padding(batch_lengths, model.max_seq_len)

            if batch_size not in host_data_cache:
                t_start = time.perf_counter()
                for _ in range(5):
                    tmp_data = bench.build_inputs(
                        model=model,
                        feature_columns=model.feature_columns,
                        batch_size=batch_size,
                        max_seq_len=model.max_seq_len,
                        avg_seq_len=model.avg_seq_len,
                        seed=seed + batch_size,
                    )
                t_end = time.perf_counter()
                concat_ms = (t_end - t_start) * 1000.0 / 5.0
                numel_breakdown = bench.h2d_numel_breakdown(mode, tmp_data, scheduler_side_concat=False)
                host_data_cache[batch_size] = (tmp_data, concat_ms, numel_breakdown)
            
            host_data, concat_ms, numel_breakdown = host_data_cache[batch_size]

            if batch_size not in h2d_cache:
                h2d_cache[batch_size] = _compute_h2d_cache(mode, host_data, device, iters=10, warmup=3)

            bench._sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = _run_infer(mode, model, infer_models, host_data, device)
            bench._sync_if_cuda(device)
            t1 = time.perf_counter()

            h2d_only, h2d_pure = h2d_cache[batch_size]
            h2d_only_samples.append(h2d_only)
            h2d_pure_samples.append(h2d_pure)
            concat_ms_samples.append(concat_ms)
            h2d_total_numel_samples.append(numel_breakdown['total'])
            h2d_data_numel_samples.append(numel_breakdown['data'])
            h2d_meta_numel_samples.append(numel_breakdown['meta'])

            batch_sizes.append(batch_size)
            done_time = t1
            for arrival_t in batch_arrivals:
                queue_wait_ms.append((t0 - arrival_t) * 1000.0)
                latencies_ms.append((done_time - arrival_t) * 1000.0)

    def _runner():
        _run_loop()

    sm_avg, gpu_avg = _collect_dmon_util(_runner)

    if not latencies_ms:
        raise RuntimeError('No requests processed. Increase duration or arrival_rate.')

    total_requests = len(latencies_ms)
    avg_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))
    throughput_qps = total_requests / duration_s
    avg_batch = float(np.mean(batch_sizes)) if batch_sizes else 0.0
    avg_queue = float(np.mean(queue_wait_ms)) if queue_wait_ms else 0.0

    avg_h2d_only = float(np.mean(h2d_only_samples)) if h2d_only_samples else 0.0
    avg_h2d_pure = float(np.mean(h2d_pure_samples)) if h2d_pure_samples else 0.0
    h2d_bubble = max(avg_h2d_only - avg_h2d_pure, 0.0)
    bubble_ratio = 0.0 if avg_h2d_only <= 0 else h2d_bubble / avg_h2d_only * 100.0

    avg_concat = float(np.mean(concat_ms_samples)) if concat_ms_samples else 0.0
    avg_h2d_total_numel = float(np.mean(h2d_total_numel_samples)) if h2d_total_numel_samples else 0.0
    avg_h2d_data_numel = float(np.mean(h2d_data_numel_samples)) if h2d_data_numel_samples else 0.0
    avg_h2d_meta_numel = float(np.mean(h2d_meta_numel_samples)) if h2d_meta_numel_samples else 0.0

    display_mode = MODE_DISPLAY.get(mode, mode)

    return RunResult(
        mode=display_mode,
        arrival_rate=arrival_rate,
        duration_s=duration_s,
        total_requests=total_requests,
        throughput_qps=throughput_qps,
        avg_ms=avg_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        avg_batch_size=avg_batch,
        max_batch_size=max_batch_size,
        avg_queue_ms=avg_queue,
        avg_h2d_only_ms=avg_h2d_only,
        avg_h2d_pure_ms=avg_h2d_pure,
        h2d_bubble_ms=h2d_bubble,
        h2d_bubble_ratio_pct=bubble_ratio,
        sm_util_avg_pct=sm_avg,
        gpu_util_avg_pct=gpu_avg,
        avg_h2d_total_numel=avg_h2d_total_numel,
        avg_h2d_data_numel=avg_h2d_data_numel,
        avg_h2d_meta_numel=avg_h2d_meta_numel,
        avg_concat_ms=avg_concat,
    )


def _select_best_under_p99(records: List[RunResult], p99_target_ms: float) -> Optional[RunResult]:
    valid = [r for r in records if r.p99_ms <= p99_target_ms]
    if not valid:
        return None
    return max(valid, key=lambda x: x.throughput_qps)


def _write_csv(path: str, rows: List[RunResult]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode', 'arrival_rate', 'duration_s', 'total_requests', 'throughput_qps',
            'avg_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'avg_batch_size', 'max_batch_size',
            'avg_queue_ms', 'avg_h2d_only_ms', 'avg_h2d_pure_ms', 'h2d_bubble_ms', 'h2d_bubble_ratio_pct',
            'sm_util_avg_pct', 'gpu_util_avg_pct', 'avg_h2d_total_numel', 'avg_h2d_data_numel', 'avg_h2d_meta_numel', 'avg_concat_ms'
        ])
        for r in rows:
            writer.writerow([
                r.mode, r.arrival_rate, r.duration_s, r.total_requests, r.throughput_qps,
                r.avg_ms, r.p50_ms, r.p95_ms, r.p99_ms, r.avg_batch_size, r.max_batch_size,
                r.avg_queue_ms, r.avg_h2d_only_ms, r.avg_h2d_pure_ms, r.h2d_bubble_ms, r.h2d_bubble_ratio_pct,
                r.sm_util_avg_pct, r.gpu_util_avg_pct, r.avg_h2d_total_numel, r.avg_h2d_data_numel, r.avg_h2d_meta_numel, r.avg_concat_ms
            ])


def _write_report(path: str, p99_target_ms: float, sweep: List[RunResult], best: List[RunResult]):
    lines = []
    lines.append('# BatchScheduler 在线到达模拟报告')
    lines.append('')
    lines.append(f'- p99 约束: <= {p99_target_ms:.1f} ms')
    lines.append('')
    lines.append(f'## p99<={p99_target_ms:.1f}ms 下最大吞吐')
    lines.append('')
    lines.append('| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in best:
        if r is None:
            continue
        lines.append(
            f"| {r.mode} | {r.arrival_rate:.1f} | {r.throughput_qps:.1f} | {r.avg_ms:.3f} | {r.p99_ms:.3f} | "
            f"{r.avg_batch_size:.2f} | {r.avg_queue_ms:.3f} | {r.avg_h2d_only_ms:.4f} | {r.h2d_bubble_ms:.4f} | {r.h2d_bubble_ratio_pct:.2f}% | "
            f"{r.sm_util_avg_pct if r.sm_util_avg_pct is not None else 'N/A'} | {r.gpu_util_avg_pct if r.gpu_util_avg_pct is not None else 'N/A'} | "
            f"{r.avg_concat_ms:.3f} | {r.avg_h2d_total_numel:.0f} | {r.avg_h2d_data_numel:.0f} | {r.avg_h2d_meta_numel:.0f} |"
        )

    lines.append('')
    lines.append('## Sweep 详情')
    lines.append('')
    lines.append('| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in sweep:
        lines.append(
            f"| {r.mode} | {r.arrival_rate:.1f} | {r.throughput_qps:.1f} | {r.avg_ms:.3f} | {r.p99_ms:.3f} | "
            f"{r.avg_batch_size:.2f} | {r.avg_queue_ms:.3f} | {r.avg_h2d_only_ms:.4f} | {r.avg_h2d_pure_ms:.4f} | {r.h2d_bubble_ratio_pct:.2f}% | "
            f"{r.sm_util_avg_pct if r.sm_util_avg_pct is not None else 'N/A'} | {r.gpu_util_avg_pct if r.gpu_util_avg_pct is not None else 'N/A'} | "
            f"{r.avg_concat_ms:.3f} | {r.avg_h2d_total_numel:.0f} | {r.avg_h2d_data_numel:.0f} | {r.avg_h2d_meta_numel:.0f} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Online arrival batch scheduler simulation for baseline, multislice, emblayerV1/V2')
    parser.add_argument('--arrival_rates', type=str, default='2000,4000,8000,12000,16000')
    parser.add_argument('--p99_target_ms', type=float, default=100.0)
    parser.add_argument('--duration_s', type=float, default=30.0)
    parser.add_argument('--max_batch_size', type=int, default=1024)
    parser.add_argument('--max_wait_ms', type=float, default=2.0)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--avg_seq_len', type=float, default=8.0)
    parser.add_argument('--num_sparse', type=int, default=200)
    parser.add_argument('--num_seq', type=int, default=100)
    parser.add_argument('--user_sparse_count', type=int, default=100)
    parser.add_argument('--even_vocab_size', type=int, default=4096)
    parser.add_argument('--odd_vocab_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out_dir', type=str, default='./batchscheduler_outputs')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--modes', type=str, default='baseline,multislice,emblayerV1,emblayerV2',
                        help='Comma-separated modes to run; aliases: emblayer/emblayerV1/v1, emblayerV2/v2')
    args = parser.parse_args()

    if args.user_sparse_count < 0 or args.user_sparse_count > args.num_sparse:
        raise ValueError('--user_sparse_count must be in [0, num_sparse]')

    arrival_rates = _parse_float_list(args.arrival_rates)
    modes = _parse_modes(args.modes)

    device = 'cpu'
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda:0'
    if device == 'cpu':
        raise RuntimeError('This script requires CUDA for throughput simulation.')

    model, feature_columns = bench.build_din_model(
        device=device,
        max_seq_len=args.max_seq_len,
        num_sparse=args.num_sparse,
        num_seq=args.num_seq,
        even_vocab_size=args.even_vocab_size,
        odd_vocab_size=args.odd_vocab_size,
    )
    model.user_sparse_count_for_emblayer_v2 = int(args.user_sparse_count)

    model.feature_columns = feature_columns
    model.max_seq_len = args.max_seq_len
    model.avg_seq_len = args.avg_seq_len

    host_data_meta = bench.build_inputs(
        model=model,
        feature_columns=feature_columns,
        batch_size=min(args.max_batch_size, 32),
        max_seq_len=args.max_seq_len,
        avg_seq_len=args.avg_seq_len,
        seed=args.seed,
    )
    infer_models = _build_infer_models(model, host_data_meta, device, modes)

    results = []
    for mode in modes:
        host_cache: Dict[int, Tuple[Dict, float, Dict]] = {}
        h2d_cache: Dict[int, Tuple[float, float]] = {}
        for rate in arrival_rates:
            res = _simulate(
                mode=mode,
                model=model,
                infer_models=infer_models,
                host_data_cache=host_cache,
                device=device,
                arrival_rate=rate,
                duration_s=args.duration_s,
                max_batch_size=args.max_batch_size,
                max_wait_ms=args.max_wait_ms,
                seed=args.seed + int(rate),
                h2d_cache=h2d_cache,
            )
            results.append(res)
            print(
                f"mode={res.mode} rate={rate:.1f} qps={res.throughput_qps:.1f} p99={res.p99_ms:.3f} "
                f"avg_batch={res.avg_batch_size:.2f} sm={res.sm_util_avg_pct} gpu={res.gpu_util_avg_pct}")
            
            # 击穿点检测：当当前模式 p99 超过目标 3 倍时，提前停止该模式的后续更高负载 sweep.避免warm时误判
            if res.p99_ms > 10 * args.p99_target_ms and rate > min(arrival_rates):
                print(
                    f"[INFO] {res.mode} p99={res.p99_ms:.3f} exceeds target {args.p99_target_ms:.1f}ms, "
                    "stop this mode's sweep")
                break

    best_rows = []
    for mode in modes:
        display_mode = MODE_DISPLAY.get(mode, mode)
        mode_rows = [r for r in results if r.mode == display_mode]
        best = _select_best_under_p99(mode_rows, args.p99_target_ms)
        if best is not None:
            best_rows.append(best)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'batchscheduler_sweep.csv')
    report_path = os.path.join(out_dir, 'online_report.md')

    _write_csv(csv_path, results)
    _write_report(report_path, args.p99_target_ms, results, best_rows)

    print('saved:', csv_path)
    print('saved:', report_path)


if __name__ == '__main__':
    main()


# cd /root/DeepCTR-Torch/examples
# /root/autodl-tmp/myenv/bin/python batchscheduler.py \
#   --arrival_rates 2000,4000,8000,12000,16000 \
#   --p99_target_ms 100 \
#   --duration_s 30 \
#   --max_batch_size 1024 \
#   --max_wait_ms 2 \
#   --max_seq_len 128 --avg_seq_len 8 \
#   --num_sparse 200 --num_seq 100 \
#   --user_sparse_count 100 \
#   --even_vocab_size 4096 --odd_vocab_size 64 \
#   --modes baseline,multislice,emblayerV1,emblayerV2 \
#   --out_dir ./batchscheduler_outputs