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


def _parse_float_list(text: str) -> List[float]:
    values = []
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
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
              host_data_cache: Dict[int, Dict],
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
                host_data_cache[batch_size] = bench.build_inputs(
                    model=model,
                    feature_columns=model.feature_columns,
                    batch_size=batch_size,
                    max_seq_len=model.max_seq_len,
                    avg_seq_len=model.avg_seq_len,
                    seed=seed + batch_size,
                )
            host_data = host_data_cache[batch_size]

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

    return RunResult(
        mode='emblayerV2' if mode == 'joint_v2' else 'baseline',
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
            'sm_util_avg_pct', 'gpu_util_avg_pct'
        ])
        for r in rows:
            writer.writerow([
                r.mode, r.arrival_rate, r.duration_s, r.total_requests, r.throughput_qps,
                r.avg_ms, r.p50_ms, r.p95_ms, r.p99_ms, r.avg_batch_size, r.max_batch_size,
                r.avg_queue_ms, r.avg_h2d_only_ms, r.avg_h2d_pure_ms, r.h2d_bubble_ms, r.h2d_bubble_ratio_pct,
                r.sm_util_avg_pct, r.gpu_util_avg_pct
            ])


def _write_report(path: str, p99_target_ms: float, sweep: List[RunResult], best: List[RunResult]):
    lines = []
    lines.append('# BatchScheduler 在线到达模拟报告')
    lines.append('')
    lines.append(f'- p99 约束: <= {p99_target_ms:.1f} ms')
    lines.append('')
    lines.append('## p99<=100ms 下最大吞吐')
    lines.append('')
    lines.append('| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in best:
        if r is None:
            continue
        lines.append(
            f"| {r.mode} | {r.arrival_rate:.1f} | {r.throughput_qps:.1f} | {r.avg_ms:.3f} | {r.p99_ms:.3f} | "
            f"{r.avg_batch_size:.2f} | {r.avg_queue_ms:.3f} | {r.h2d_bubble_ms:.4f} | {r.h2d_bubble_ratio_pct:.2f}% | "
            f"{r.sm_util_avg_pct if r.sm_util_avg_pct is not None else 'N/A'} | {r.gpu_util_avg_pct if r.gpu_util_avg_pct is not None else 'N/A'} |"
        )

    lines.append('')
    lines.append('## Sweep 详情')
    lines.append('')
    lines.append('| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in sweep:
        lines.append(
            f"| {r.mode} | {r.arrival_rate:.1f} | {r.throughput_qps:.1f} | {r.avg_ms:.3f} | {r.p99_ms:.3f} | "
            f"{r.avg_batch_size:.2f} | {r.avg_queue_ms:.3f} | {r.avg_h2d_only_ms:.4f} | {r.avg_h2d_pure_ms:.4f} | {r.h2d_bubble_ratio_pct:.2f}% | "
            f"{r.sm_util_avg_pct if r.sm_util_avg_pct is not None else 'N/A'} | {r.gpu_util_avg_pct if r.gpu_util_avg_pct is not None else 'N/A'} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Online arrival batch scheduler simulation for baseline vs emblayerV2')
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
    args = parser.parse_args()

    if args.user_sparse_count < 0 or args.user_sparse_count > args.num_sparse:
        raise ValueError('--user_sparse_count must be in [0, num_sparse]')

    arrival_rates = _parse_float_list(args.arrival_rates)

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
    infer_models = _build_infer_models(model, host_data_meta, device)

    results = []
    for mode in ('baseline', 'joint_v2'):
        host_cache: Dict[int, Dict] = {}
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
            
            # 击穿点检测：如果 baseline 的 p99 已经超过目标的 3 倍，认为后续更高的负载下 baseline 不可能满足 p99 约束，提前停止 baseline 的 sweep
            if mode == 'baseline' and res.p99_ms > 3*args.p99_target_ms :
                print(
                    f"[INFO] baseline p99={res.p99_ms:.3f} exceeds target {args.p99_target_ms:.1f}ms, "
                    "stop baseline sweep and continue with experimental group")
                break

    best_rows = []
    for mode in ('baseline', 'emblayerV2'):
        mode_rows = [r for r in results if r.mode == mode]
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
#   --out_dir ./batchscheduler_outputs