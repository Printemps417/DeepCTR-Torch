# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multislice | 2800.0 | 2723.0 | 203.709 | 250.380 | 226.92 | 113.501 | 1.2467 | 0.0001 | 0.01% | 17.5 | 34.5 | 0.227 | 17733460 | 17733460 | 0 |

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multislice | 2500.0 | 2280.0 | 596.960 | 691.870 | 228.00 | 474.245 | 1.2532 | 1.2529 | 0.03% | 5.5 | 21.5 | 0.228 | 17825982 | 17825982 | 0 |
| multislice | 2600.0 | 2530.0 | 233.804 | 310.083 | 230.00 | 133.152 | 1.2633 | 1.2633 | 0.00% | 17.0 | 33.0 | 0.230 | 17971775 | 17971775 | 0 |
| multislice | 2700.0 | 2620.0 | 163.387 | 231.981 | 218.33 | 73.313 | 1.2026 | 1.2024 | 0.02% | 20.5 | 38.5 | 0.218 | 16962440 | 16962440 | 0 |
| multislice | 2800.0 | 2723.0 | 203.709 | 250.380 | 226.92 | 113.501 | 1.2467 | 1.2466 | 0.01% | 17.5 | 34.5 | 0.227 | 17733460 | 17733460 | 0 |
