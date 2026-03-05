# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2900.0 | 2444.0 | 577.140 | 670.552 | 222.18 | 467.630 | 1.2205 | 1.2200 | 0.04% | 15.5 | 39.0 | 0.222 | 17607293 | 17607293 | 0 |
| baseline | 3000.0 | 3026.0 | 330.350 | 445.602 | 216.14 | 224.268 | 1.1878 | 1.1876 | 0.02% | 12.0 | 17.5 | 0.216 | 16967947 | 16967947 | 0 |
