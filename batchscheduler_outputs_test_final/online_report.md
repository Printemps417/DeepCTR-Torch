# BatchScheduler 在线到达模拟报告

- p99 约束: <= 100.0 ms

## p99<=100.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2500.0 | 2445.4 | 344.501 | 582.536 | 244.54 | 248.147 | 9.4476 | 1.3402 | 85.81% | 25.833333333333332 | 53.666666666666664 | 0.245 | 18854523 | 18854523 | 0 |
| multislice | 2500.0 | 2462.2 | 119.185 | 165.331 | 192.36 | 40.221 | 1.4666 | 1.0591 | 27.79% | 23.0 | 52.4 | 0.192 | 14831293 | 14831293 | 0 |
| emblayerV1 | 2500.0 | 2438.6 | 146.356 | 380.690 | 193.54 | 63.765 | 0.3512 | 0.1717 | 51.11% | 27.333333333333332 | 51.5 | 0.194 | 2025590 | 1908689 | 116901 |
| emblayerV2 | 2500.0 | 2433.0 | 123.576 | 215.268 | 196.21 | 41.294 | 1.3370 | 0.1967 | 85.29% | 27.6 | 52.8 | 0.196 | 1952651 | 1884892 | 67759 |
