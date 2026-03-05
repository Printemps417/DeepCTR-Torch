# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV2 | 2900.0 | 2928.0 | 546.826 | 628.802 | 225.23 | 447.173 | 0.1843 | 0.1706 | 7.43% | 10.5 | 24.5 | 0.225 | 2260751 | 2191954 | 68797 |
| emblayerV2 | 3000.0 | 2896.0 | 355.037 | 447.730 | 222.77 | 253.119 | 0.1828 | 0.1691 | 7.45% | 13.0 | 24.5 | 0.223 | 2248644 | 2180194 | 68450 |
