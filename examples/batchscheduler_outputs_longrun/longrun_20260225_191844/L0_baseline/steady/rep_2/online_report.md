# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2700.0 | 2672.0 | 134.184 | 178.829 | 222.67 | 45.628 | 1.2238 | 0.0030 | 0.24% | 0.0 | 9.0 | 0.223 | 18196072 | 18196072 | 0 |

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2500.0 | 2499.0 | 691.655 | 775.036 | 227.18 | 572.928 | 1.2489 | 1.2460 | 0.23% | 2.5 | 17.0 | 0.227 | 17551219 | 17551219 | 0 |
| baseline | 2600.0 | 2552.0 | 164.880 | 230.638 | 232.00 | 73.420 | 1.2750 | 1.2724 | 0.20% | 0.0 | 3.0 | 0.232 | 17999812 | 17999812 | 0 |
| baseline | 2700.0 | 2672.0 | 134.184 | 178.829 | 222.67 | 45.628 | 1.2238 | 1.2209 | 0.24% | 0.0 | 9.0 | 0.223 | 18196072 | 18196072 | 0 |
| baseline | 2800.0 | 2175.0 | 165.179 | 424.179 | 197.73 | 47.493 | 1.0879 | 1.0853 | 0.24% | 19.5 | 37.0 | 0.198 | 15981142 | 15981142 | 0 |
