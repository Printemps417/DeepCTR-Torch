# Batch Scheduler for Recommender Inference: 架构设计与优化报告

## 摘要

本文档给出 DeepCTR-Torch 推理侧 Batch Scheduler 的系统化设计说明，重点讨论在线到达场景下的动态批处理（dynamic batching）框架、触发策略、缓存预热、空闲调度、数据路径优化与工程可扩展性。文档面向技术文档与研发实现对齐，不涉及实验结果与横向对比，仅解释“为什么这样设计、如何实现、可扩展到哪里”。

该 Scheduler 的核心定位并非单纯追求吞吐，而是在到达抖动、序列长度波动、GPU 计算-传输异构成本并存的条件下，建立一个可配置、可解释、可观测、可迭代的在线调度框架。其关键设计思路是：

1. 以统一状态机驱动在线队列与批次触发；
2. 以三级触发规则统一吞吐、时延与设备利用率目标；
3. 以 warmup + 切片复用 + 线性插值构成低开销缓存体系；
4. 以分阶段指标（queue / h2d / concat / e2e）支撑问题定位与后续优化闭环。

## 关键词

Batch Scheduler；在线推理；推荐系统；动态批处理；GPU 推理；H2D 传输；缓存预热；空闲调度

---

## 1. 设计背景与问题定义

### 1.1 在线推荐推理的典型挑战

推荐系统在线推理与离线批处理有本质差异：

- 请求是随机到达而非固定批次输入；
- 用户行为序列长度存在长尾分布；
- GPU 计算链路与 Host→Device 传输链路耦合明显；
- 系统需要在 SLA（如 p99）约束下运行，而不是仅优化平均吞吐。

若直接采用固定 batch 或“到一个请求推一个请求”的极端策略，会分别导致两类问题：

- 固定大 batch：吞吐较好，但排队等待显著，尾时延恶化；
- 固定小 batch：时延较稳，但设备利用率偏低，吞吐上限受限。

因此，需要一个在线调度机制在不同压力区间内做自适应平衡。

### 1.2 调度器职责边界

本 Scheduler 的职责聚焦在“在线请求到达后的执行组织”，而非完整 Serving 框架。其边界包括：

- 输入：到达率配置、模型模式（baseline/multislice/emblayerV1/emblayerV2）、批处理上限、等待上限等；
- 输出：按 sweep 维度统计的时延/吞吐/队列/H2D 等指标；
- 不负责：上游网络接入、鉴权、请求路由、分布式副本一致性。

该边界划分使调度器可以作为中间件能力复用于多种推理后端。

---

## 2. 架构目标与非目标

### 2.1 设计目标

**目标 G1：统一调度抽象**  
不同推理路径共享同一套在线调度主循环，避免每种后端维护一套独立调度器。

**目标 G2：低扰动执行**  
调度器自身不能成为瓶颈，必须将缓存 miss、输入重建、参数探测等高开销操作前置或近似化。

**目标 G3：可解释性**  
每个请求时延可拆解为 queue + concat + execute（包含 H2D 与 compute），便于定位瓶颈。

**目标 G4：可配置性**  
关键行为由参数驱动（arrival_rate, max_batch_size, max_wait_ms, mode, use_int32 等），便于按业务场景调优。

**目标 G5：可扩展性**  
保留多模型、多设备、多队列等扩展空间，并在接口层预留稳定契约。

### 2.2 非目标

- 不在调度器内部实现完整分布式负载均衡；
- 不在本阶段引入复杂学习型调度策略（如 RL）；
- 不把所有优化下沉到自定义 CUDA 内核，优先保证策略层可演进。

---

## 3. 系统分层与核心模块

### 3.1 视图一：功能分层

Scheduler 可拆解为五层：

1. **配置解析层**：解析 arrival rates、mode 别名、批上限与等待阈值；
2. **模型构建层**：按 mode 组装推理模型实例；
3. **在线调度层**：维护队列、做触发决策、形成动态批并执行；
4. **缓存优化层**：预热 host buffer、H2D 统计缓存、插值与切片复用；
5. **观测与报告层**：聚合指标，输出 CSV 与报告。

### 3.2 视图二：关键函数职责映射

- `_build_infer_models`：多模式推理后端工厂；
- `_simulate`：在线到达 + 队列 + 触发 + 执行的主状态机；
- `_prewarm_caches`：战略批点预热；
- `_slice_host_data`：由大 batch 快速切片得到小 batch；
- `_get_interpolated_metrics`：对未测批大小进行线性插值估计；
- `_collect_dmon_util`：采集运行期间 GPU/SM 利用率；
- `_write_csv` / `_write_report`：结构化结果落盘。

### 3.3 数据契约

`RunResult` 作为统一记录结构，保证不同 mode 下输出列一致。其字段涵盖：

- 负载维度：arrival_rate, duration_s, total_requests；
- 时延维度：avg/p50/p95/p99；
- 调度维度：avg_batch_size, avg_queue_ms；
- 传输维度：avg_h2d_only_ms, avg_h2d_pure_ms, h2d_bubble_ratio_pct；
- 设备维度：sm_util_avg_pct, gpu_util_avg_pct；
- 输入规模维度：h2d total/data/meta numel。

统一契约的意义在于：调度策略、模型结构和数据形态可以演进，但观测协议稳定，便于持续集成与历史复盘。

---

## 4. 在线调度状态机设计

### 4.1 到达过程建模

请求到达间隔使用指数分布建模，等价于泊松到达过程：

$$
\Delta t \sim \mathrm{Exponential}(\lambda),\quad \lambda=\text{arrival\_rate}
$$

该建模适合近似真实线上“独立到达 + 局部突发”的统计特征，且便于参数化 sweep。

### 4.2 队列模型

队列元素结构为 `(arrival_timestamp, seq_len)`，其中 `seq_len` 用泊松采样并截断到 `[1, max_seq_len]`，使样本长度分布具备可控随机性。

队列语义为 FIFO，满足在线公平性与时延可解释性。执行前按当前可用请求形成批次，不做复杂重排（避免引入排序收益与公平性冲突）。

### 4.3 三级触发策略

每轮循环计算三类触发信号：

- **批满触发**：`len(queue) >= max_batch_size`；
- **超时触发**：最老请求等待时间超过 `max_wait_ms`；
- **空闲触发**：执行设备空闲且队列非空。

统一触发判定可写为：

$$
\text{trigger} = \mathbb{I}(Q\ge B_{max}) \lor \mathbb{I}(W_{oldest}\ge T_{max}) \lor \mathbb{I}(\text{idle} \land Q>0)
$$

其工程意义：

- 批满条件保障吞吐；
- 超时条件保障时延上界；
- 空闲条件避免“设备闲着但不发车”的保守等待。

### 4.4 Idle 空闲调度策略

当队列为空时，调度器进入短睡眠，不忙等：

$$
\Delta t_{sleep}=\min(0.5\text{ms},\; t_{next\_arrival}-t_{now})
$$

当队列非空但尚未触发时，采用更短轮询睡眠（例如 0.2ms）以提高响应性。这是典型“低 CPU 占用 + 可控调度粒度”折中策略。

该设计避免了两种极端：

- 纯忙等：CPU 开销高，且对时延收益有限；
- 粗粒度 sleep：响应迟滞，导致额外排队时延。

### 4.5 时延分解

单请求端到端时延定义：

$$
L = t_{done} - t_{arrival}
$$

队列等待：

$$
L_q = t_{exec\_start} - t_{arrival}
$$

可进一步理解为：

$$
L \approx L_q + L_{concat} + L_{h2d+compute}
$$

其中 `L_concat` 在当前实现中通过可配置近似或缓存值模拟，`L_{h2d+compute}` 由实际推理调用产生。

---

## 5. 数据路径与内存优化

### 5.1 Host 侧输入组织

输入结构由 `bench.build_inputs` 生成，包含 dense/sparse/sequence 及其前缀、偏移、索引等元信息。不同 mode 下会选择不同组织方式，但调度器对其采用“字典 + 统一缓存”抽象，不依赖具体算子实现细节。

### 5.2 pinned memory 与传输稳定性

Host 输入通过 `pin_host_data_tensors` 处理，目的是让 H2D 拷贝具备更稳定的 DMA 行为与更低抖动。对在线系统而言，降低抖动常常比降低均值更重要，因为尾时延对用户体验更敏感。

### 5.3 以切片替代重建

缓存 miss 时，调度器优先从已存在的大 batch（master batch）切片构造小 batch，而不是重新调用构建流程。该策略有三点优势：

1. 避免重复随机生成与张量构建；
2. 避免频繁触发 Python 层对象分配；
3. 使 miss 代价与批大小近似线性相关。

从复杂度上看，若重建代价记为 $C_{build}(B)$，切片代价记为 $C_{slice}(b)$，一般有：

$$
C_{slice}(b) \ll C_{build}(b),\quad b\le B
$$

### 5.4 H2D 指标分层

调度器区分两个指标：

- `h2d_only_ms`：包含输入准备路径在内的“可见 H2D 阶段时间”；
- `h2d_pure_ms`：仅测张量复制动作。

定义 H2D Bubble：

$$
\text{bubble} = \max(\overline{h2d\_only} - \overline{h2d\_pure}, 0)
$$

定义 Bubble Ratio：

$$
\text{bubble\_ratio}=\frac{\text{bubble}}{\overline{h2d\_only}}\times 100\%
$$

该分层使“传输本身慢”与“准备链路慢”可以被区分，从而避免误判优化方向。

---

## 6. Warmup 与缓存体系设计

### 6.1 预热目标

Warmup 的本质不是“让 GPU 跑热”，而是让在线阶段尽量避免首次路径开销，包括：

- 首次输入构建与内存分配；
- 首次 H2D 统计测量；
- 首次 batch 规模覆盖不足导致的动态探测。

### 6.2 战略批点选择

预热点由以下集合并集构成：

- 边界点：`1` 与 `max_batch_size`；
- 对数点：`2^k` 序列（直到上限）；
- 分位点：`0.25/0.5/0.75 * max_batch_size`。

该设计兼顾“覆盖度”与“预热成本”：

- 对数点提升跨数量级拟合能力；
- 分位点改善中区间插值精度；
- 边界点保障插值外推的稳定边界。

### 6.3 插值估计策略

对于未直接测量的 batch size，采用线性插值估计 `concat_ms` 和 `h2d` 相关开销：

$$
\hat v(b)=v(b_1)+\frac{v(b_2)-v(b_1)}{b_2-b_1}(b-b_1),\quad b_1<b<b_2
$$

该方法的工程价值是将在线“测量行为”前移为离线近似，减少主循环抖动与尾时延污染。

### 6.4 预热与在线主循环解耦

预热阶段在每个 mode 进入 sweep 前执行一次；在线循环只读缓存并按需补齐。这样可以保证：

- 主循环逻辑简单、可预测；
- 预热策略可独立演化（例如改成自适应采样）；
- 调度行为不受临时 benchmark 干扰。

---

## 7. 多模式推理后端统一接入

### 7.1 模式抽象

当前支持模式包括 baseline、multislice、emblayerV1、emblayerV2。调度器将其抽象为统一接口：

- 输入准备：`_prepare_inputs`；
- 前向执行：`_forward_mode`；
- 指标评估：`h2d_only_ms / pure_h2d_only_ms / h2d_numel_breakdown`。

### 7.2 工厂式构建

`_build_infer_models` 按 mode 需求按需加载扩展与模型对象。该工厂模式具备两点优势：

1. 避免无关后端在运行期占用资源；
2. 模式扩展时仅需增加分支与映射，不改调度主循环。

### 7.3 对调度器透明的算子差异

多模式在算子层差异较大（如多切片、序列/向量融合路径），但调度器只关心“每批可执行”与“指标可观测”这两个契约。该透明性是系统长期可维护的关键。

---

## 8. 观测体系与工程可解释性

### 8.1 指标分层原则

设计上遵循“每层至少一个可解释指标”：

- 队列层：`avg_queue_ms`；
- 批处理层：`avg_batch_size`；
- 传输层：`avg_h2d_only_ms / avg_h2d_pure_ms / bubble_ratio`；
- 执行层：`avg/p50/p95/p99`；
- 设备层：`sm_util_avg_pct / gpu_util_avg_pct`。

### 8.2 外部采样与主流程隔离

GPU 利用率通过独立线程读取 `nvidia-smi dmon`，运行与停止由包装函数控制。该方式不侵入推理算子路径，可用于线上前的容量评估与离线调优闭环。

### 8.3 报告输出协议

CSV 与 Markdown 报告由统一字段写出，适合后续接入可视化脚本与 CI 工具链。稳定输出协议使长期趋势分析可自动化。

---

## 9. 稳定性与故障处理设计

### 9.1 Fail-fast 原则

- 非 CUDA 环境直接报错退出，避免生成误导性吞吐数据；
- mode 解析失败、参数越界（如 `user_sparse_count > num_sparse`）直接阻断；
- 若无请求被处理，抛出显式异常提示调整负载参数。

### 9.2 有界运行与提前终止

当尾时延明显击穿目标阈值时，可提前终止该 mode 后续更高压力 sweep，防止无意义长时间运行。该机制降低离线评估成本，也避免对共享设备造成长时间干扰。

### 9.3 可重复性

通过固定随机种子、显式到达率配置与统一报告协议，保证同配置下复现实验与问题复盘的一致性。

---

## 10. 复杂度与性能工程视角

### 10.1 主循环复杂度

设总请求数为 $N$，每次触发形成批大小 $b_i$，则：

- 队列入队/出队总体为 $O(N)$；
- 批执行次数约为 $K\approx N/\bar b$；
- 若缓存命中良好，额外调度开销近似 $O(K)$。

因此，优化关键在于提高命中率、降低 miss 成本，而非在循环中增加复杂决策。

### 10.2 关键工程折中

- 更激进触发：时延优先但吞吐可能下降；
- 更保守触发：吞吐优先但 queue 增长更快；
- 更密预热点：插值误差更小但预热时间更长。

当前实现通过“有限战略点 + 线性插值 + 空闲触发”形成中庸且鲁棒的默认策略。

---

## 11. 可扩展路线

### 11.1 多队列与优先级

可在当前 FIFO 上扩展为多优先级队列（如交互流量优先于探索流量），触发函数可推广为分队列加权策略。

### 11.2 多 GPU / 多实例调度

在调度器上层增加路由器：

- 设备选择：按队列长度、历史时延或利用率分发；
- 批次归并：在实例内维持当前策略，跨实例只做负载均衡。

### 11.3 自适应阈值

可将 `max_wait_ms` 与 `max_batch_size` 从静态配置升级为反馈控制：

$$
T_{max}(t+1)=T_{max}(t)+\eta\cdot (\text{SLA}_{target}-\text{SLA}_{obs})
$$

即依据近期时延偏差动态调整触发阈值，实现更稳态的 SLA 控制。

### 11.4 在线学习型调度

在保持当前可解释策略为默认路径的前提下，可增加“学习器给建议、规则做兜底”的混合架构，兼顾创新与稳定性。

---

## 12. 伪代码（面向实现复核）

```text
Initialize model/infer backends by mode
Initialize host_data_cache, h2d_cache
Prewarm strategic batch points

for each arrival_rate:
    queue <- empty
    next_arrival <- sample_exponential(rate)
    last_gpu_finish <- now

    while simulation_not_end:
        now <- clock()
        enqueue all arrivals happened before now

        if queue is empty:
            sleep(min(0.5ms, time_to_next_arrival))
            continue

        timeout <- oldest_wait_exceeds(max_wait_ms)
        full    <- queue_size >= max_batch_size
        idle    <- now >= last_gpu_finish

        trigger <- full or timeout or (idle and queue_non_empty)
        if not trigger:
            sleep(0.2ms)
            continue

        batch <- pop min(queue_size, max_batch_size)
        host_data <- cache_hit_or_slice_or_build(batch_size)
        h2d_stat <- cache_hit_or_interpolate(batch_size)

        sleep(concat_ms_estimate)
        sync_cuda()
        t_start <- now
        run_inference(mode, host_data)
        sync_cuda()
        t_done <- now

        update latency/queue/h2d/numel metrics
        last_gpu_finish <- t_done

Aggregate metrics and write outputs
```

---

## 13. 落地建议（面向技术文档集成）

为便于并入正式技术文档，建议将本报告作为“推理系统架构”子章节，配套以下实践：

1. 在文档中固定调度器术语：触发、批次、队列、bubble、预热点；
2. 把参数表单独维护为配置附录，避免正文被实现细节淹没；
3. 在代码变更流程中要求同步更新“函数职责映射”与“伪代码”；
4. 将 CSV 字段协议视为稳定接口，新增字段采用向后兼容策略。

---

## 14. 结论

本 Batch Scheduler 构建了一个面向在线推荐推理的“策略可解释、执行低扰动、指标可闭环”的架构基线。其核心价值不在单一指标提升，而在于把动态批处理从“经验调参”提升为“可建模、可验证、可扩展”的工程系统：

- 通过三级触发策略统一吞吐-时延-利用率目标；
- 通过 warmup 与缓存体系显著降低调度器自开销；
- 通过分层观测模型支撑优化问题快速定位；
- 通过模式透明接口保障后端演进与调度策略解耦。

该设计可作为后续自适应调度、分布式多实例路由与学习型策略的稳定底座。