# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV1 | 2500.0 | 2380.0 | 460.709 | 554.811 | 216.36 | 357.850 | 0.1823 | 0.1780 | 2.40% | 15.0 | 25.5 | 0.216 | 2326375 | 2192394 | 133981 |
| emblayerV1 | 2600.0 | 2542.0 | 243.748 | 365.358 | 231.09 | 135.726 | 0.1915 | 0.1878 | 1.96% | 12.5 | 23.0 | 0.231 | 2447358 | 2306349 | 141010 |
| emblayerV1 | 2700.0 | 2504.0 | 182.324 | 323.684 | 227.64 | 79.965 | 0.1899 | 0.1861 | 1.96% | 17.5 | 37.0 | 0.228 | 2455694 | 2314246 | 141449 |
| emblayerV1 | 2800.0 | 2782.0 | 246.977 | 345.271 | 231.83 | 144.811 | 0.1920 | 0.1884 | 1.87% | 20.5 | 38.0 | 0.232 | 2467065 | 2324921 | 142144 |
