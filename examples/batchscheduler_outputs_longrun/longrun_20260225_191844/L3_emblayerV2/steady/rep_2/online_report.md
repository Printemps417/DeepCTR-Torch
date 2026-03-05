# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV2 | 2500.0 | 2409.0 | 472.300 | 562.363 | 219.00 | 367.966 | 0.3022 | 0.1666 | 44.86% | 15.5 | 30.0 | 0.219 | 2202010 | 2134981 | 67029 |
| emblayerV2 | 2600.0 | 2578.0 | 258.879 | 388.347 | 214.83 | 148.316 | 0.3037 | 0.1646 | 45.81% | 12.5 | 24.5 | 0.215 | 2137219 | 2072142 | 65077 |
| emblayerV2 | 2700.0 | 2604.0 | 203.431 | 354.458 | 217.00 | 96.949 | 0.3039 | 0.1649 | 45.74% | 17.0 | 33.0 | 0.217 | 2184394 | 2117912 | 66482 |
| emblayerV2 | 2800.0 | 2610.0 | 258.168 | 359.374 | 217.50 | 155.301 | 0.3038 | 0.1654 | 45.56% | 17.0 | 33.0 | 0.218 | 2177128 | 2110847 | 66281 |
