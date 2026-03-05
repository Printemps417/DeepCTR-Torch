# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV1 | 2900.0 | 2904.0 | 543.920 | 630.477 | 223.38 | 443.200 | 0.1870 | 0.1826 | 2.34% | 15.0 | 28.5 | 0.223 | 2388423 | 2250894 | 137529 |
| emblayerV1 | 3000.0 | 2928.0 | 361.251 | 464.590 | 225.23 | 258.051 | 0.1884 | 0.1840 | 2.32% | 18.5 | 27.0 | 0.225 | 2375666 | 2238834 | 136832 |
