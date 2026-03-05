# BatchScheduler 在线到达模拟报告

- p99 约束: <= 1000.0 ms

## p99<=1000.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2600.0 | 2603.6 | 2509.650 | 4186.388 | 126.39 | 2426.591 | 5.3418 | 1.3837 | 74.10% | 33.3 | 64.8 |
| emblayerV2 | 2600.0 | 2612.4 | 2062.228 | 3851.794 | 126.82 | 1979.496 | 0.1689 | 0.0488 | 71.12% | 36.22222222222222 | 65.44444444444444 |
