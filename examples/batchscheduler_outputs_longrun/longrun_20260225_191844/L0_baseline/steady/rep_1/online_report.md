# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2800.0 | 2595.0 | 166.221 | 229.945 | 216.25 | 80.276 | 1.1905 | 0.0031 | 0.26% | 0.0 | 0.0 | 0.216 | 16705433 | 16705433 | 0 |

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2500.0 | 2464.0 | 488.599 | 585.173 | 224.00 | 381.462 | 1.2329 | 1.2299 | 0.25% | 14.0 | 32.0 | 0.224 | 17551219 | 17551219 | 0 |
| baseline | 2600.0 | 2414.0 | 147.852 | 218.286 | 219.45 | 58.380 | 1.2072 | 1.2045 | 0.22% | 0.0 | 2.0 | 0.219 | 17999812 | 17999812 | 0 |
| baseline | 2700.0 | 2277.0 | 182.391 | 367.455 | 207.00 | 69.373 | 1.1415 | 1.1385 | 0.26% | 21.0 | 37.0 | 0.207 | 16317587 | 16317587 | 0 |
| baseline | 2800.0 | 2595.0 | 166.221 | 229.945 | 216.25 | 80.276 | 1.1905 | 1.1874 | 0.26% | 0.0 | 0.0 | 0.216 | 16705433 | 16705433 | 0 |
