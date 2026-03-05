# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multislice | 2900.0 | 2904.0 | 566.469 | 642.809 | 223.38 | 465.947 | 1.2772 | 1.2265 | 3.97% | 14.5 | 23.5 | 0.223 | 17555532 | 17555532 | 0 |
| multislice | 3000.0 | 2849.0 | 349.183 | 439.269 | 219.15 | 248.004 | 1.2028 | 1.2015 | 0.11% | 19.0 | 35.0 | 0.219 | 16897200 | 16897200 | 0 |
