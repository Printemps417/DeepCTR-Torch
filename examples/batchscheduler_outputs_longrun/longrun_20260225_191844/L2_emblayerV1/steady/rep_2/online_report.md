# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV1 | 2800.0 | 2674.0 | 186.049 | 235.231 | 222.83 | 96.926 | 0.1882 | 0.0055 | 2.92% | 17.0 | 32.5 | 0.223 | 2356164 | 2220462 | 135702 |

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV1 | 2500.0 | 2450.0 | 494.553 | 573.663 | 222.73 | 384.984 | 0.1891 | 0.1831 | 3.19% | 12.5 | 25.5 | 0.223 | 2386587 | 2249092 | 137495 |
| emblayerV1 | 2600.0 | 2548.0 | 334.971 | 539.819 | 231.64 | 208.203 | 0.1950 | 0.1887 | 3.24% | 12.5 | 25.0 | 0.232 | 2447358 | 2306349 | 141010 |
| emblayerV1 | 2700.0 | 2643.0 | 165.593 | 235.662 | 220.25 | 75.505 | 0.1860 | 0.1805 | 2.95% | 22.0 | 35.5 | 0.220 | 2376913 | 2240003 | 136910 |
| emblayerV1 | 2800.0 | 2674.0 | 186.049 | 235.231 | 222.83 | 96.926 | 0.1882 | 0.1827 | 2.92% | 17.0 | 32.5 | 0.223 | 2356164 | 2220462 | 135702 |
