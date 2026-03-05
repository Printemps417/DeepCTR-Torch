# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV2 | 2700.0 | 2549.0 | 218.514 | 293.804 | 231.73 | 122.428 | 0.1910 | 0.0147 | 7.70% | 17.5 | 33.0 | 0.232 | 2324369 | 2253618 | 70751 |

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emblayerV2 | 2500.0 | 2404.0 | 468.164 | 557.197 | 218.55 | 362.989 | 0.1818 | 0.1672 | 8.06% | 8.0 | 20.0 | 0.219 | 2202010 | 2134981 | 67029 |
| emblayerV2 | 2600.0 | 2440.0 | 322.351 | 436.581 | 221.82 | 208.625 | 0.1836 | 0.1692 | 7.85% | 12.5 | 23.5 | 0.222 | 2255317 | 2186646 | 68671 |
| emblayerV2 | 2700.0 | 2549.0 | 218.514 | 293.804 | 231.73 | 122.428 | 0.1910 | 0.1763 | 7.70% | 17.5 | 33.0 | 0.232 | 2324369 | 2253618 | 70751 |
| emblayerV2 | 2800.0 | 2631.0 | 313.897 | 521.075 | 219.25 | 194.226 | 0.1817 | 0.1673 | 7.90% | 17.5 | 36.5 | 0.219 | 2230195 | 2162308 | 67887 |
