# BatchScheduler 在线到达模拟报告

- p99 约束: <= 300.0 ms

## p99<=300.0ms 下最大吞吐

| Mode | Arrival Rate | Max Throughput (QPS) | Avg Latency (ms) | p99 (ms) | Avg Batch | Queue (ms) | Avg H2D-only (ms) | H2D Bubble (ms) | H2D Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multislice | 2800.0 | 2607.0 | 172.669 | 232.711 | 217.25 | 86.407 | 1.1936 | 0.0004 | 0.04% | 17.5 | 33.5 | 0.217 | 16911039 | 16911039 | 0 |

## Sweep 详情

| Mode | Arrival Rate | Throughput (QPS) | Avg (ms) | p99 (ms) | Avg Batch | Queue (ms) | H2D-only (ms) | Pure H2D (ms) | Bubble Ratio (%) | SM Util (%) | GPU Util (%) | Avg Concat (ms) | H2D Total Numel | H2D Data Numel | H2D Meta Numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multislice | 2500.0 | 2316.0 | 526.577 | 613.793 | 210.55 | 419.360 | 1.1583 | 1.1579 | 0.03% | 13.0 | 31.5 | 0.211 | 16261513 | 16261513 | 0 |
| multislice | 2600.0 | 2436.0 | 153.053 | 231.764 | 221.45 | 65.977 | 1.2168 | 1.2161 | 0.06% | 0.0 | 4.0 | 0.221 | 17999812 | 17999812 | 0 |
| multislice | 2700.0 | 2577.0 | 174.580 | 253.047 | 214.75 | 82.441 | 1.1795 | 1.1788 | 0.06% | 17.5 | 35.5 | 0.215 | 16962440 | 16962440 | 0 |
| multislice | 2800.0 | 2607.0 | 172.669 | 232.711 | 217.25 | 86.407 | 1.1936 | 1.1932 | 0.04% | 17.5 | 33.5 | 0.217 | 16911039 | 16911039 | 0 |
