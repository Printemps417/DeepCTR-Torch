# 4 模块设计、实现与实验分析（DIN + Emblayer + BatchScheduler）

本章面向 DeepCTR-Torch 中 DIN 推理链路，围绕 Emblayer 自定义算子与 BatchScheduler 在线调度器的联合优化展开。内容覆盖模块级设计、关键实现细节、编译期图改写、在线调度状态机与缓存体系，以及基于已给定 long-run 实验结果的性能分析。章节中所述实现与数据对应以下代码与实验工件：

- `examples/benchmark_dinemblayer_infer.py`
- `examples/batchscheduler.py`
- `examples/emblayer_op/emblayer_seq.cpp`
- `examples/emblayer_op/emblayer_seq_cuda.cu`
- `examples/emblayer_op/emblayer_vec.cpp`
- `examples/emblayer_op/emblayer_vec_cuda.cu`
- `examples/onnx_emblayerseq_converter.py`
- `examples/onnx_emblayer_joint_converter.py`
- `examples/onnx_multislice_converter.py`
- `examples/batchscheduler_outputs_longrun/longrun_20260225_194030_nice/all_report.md`

---

## 4.0 推荐链路调用位置与CTR模型代码结构

为了避免把 Emblayer 与 BatchScheduler 误解为“独立的 benchmark 附件模块”，有必要先明确它们在推荐链路中的实际挂载位置。本节从在线推理链路与仓库代码组织两个维度给出定位。

### 4.0.1 在推荐链路中的调用位置

在典型 CTR 在线服务中，一次请求通常经历“特征组装 -> 输入张量化 -> 模型推理 -> 结果后处理”四步。本项目的改造并不改变模型业务语义，而是替换了“输入张量化后到模型前向之间”的执行实现。

可抽象为：

$$
	ext{Feature Service} \rightarrow \text{Tensor Builder} \rightarrow \text{(Emblayer/MultiSlice Rebuild)} \rightarrow \text{DIN Forward} \rightarrow \text{CTR Score}
$$

其中：

1. **特征组装阶段（CPU）**  
    请求进入后，离散特征、稠密特征、行为序列特征会被拼成 `X` 对应的列布局（由 `feature_index` 定义）。

2. **输入表达阶段（Host）**  
    `examples/benchmark_dinemblayer_infer.py` 中 `build_inputs()` 会同时生成多种输入视图：`full`（baseline）、`non_seq_full`、`non_seq_compact`、`vec_*`、`seq_*`。在真实链路中可理解为“调度器根据模式选择输入协议”。

3. **重建阶段（GPU）——本次改造核心位置**  
    - baseline：直接使用 `full` 进入 `DINBaselinePerInputSlice`；  
    - multislice：`full -> MultiSlice -> 回填`；  
    - joint/joint_v2：`vec 压缩 + seq 压缩 -> EmblayerVec/Seq kernel 重建 -> x`。  
    这一阶段由 `DINSeqOnlyInfer`、`DINJointInfer`、`DINJointInferV2` 在 `forward` 中完成。

4. **CTR 推理阶段（DIN 主干）**  
    重建后的 `x` 与 baseline 张量语义一致，继续进入 DIN 的 embedding lookup、attention pooling 与 DNN 计算，最终输出点击概率。

因此，改造的“调用插入点”位于**输入协议与 DIN 前向之间**，属于推理前处理路径的算子化与调度化，而不是改写 DIN 的损失函数或网络结构。

### 4.0.2 CTR模型代码结构（DeepCTR-Torch 内部映射）

从仓库实现看，CTR 模型代码是分层组织的，改造正好卡在层与层之间：

1. **特征抽象层：`deepctr_torch/inputs.py`**  
    - `SparseFeat`：离散 ID 特征定义；  
    - `DenseFeat`：连续值特征定义；  
    - `VarLenSparseFeat`：序列离散特征定义；  
    - `build_input_features()`：为每个特征分配列区间，形成 `feature_index`。  
    该层决定了输入列布局，是后续 Slice/Emblayer range 的根依据。

2. **模型基类层：`deepctr_torch/models/basemodel.py`**  
    - `BaseModel`：统一管理 `feature_index`、embedding_dict、linear 分支、预测头；  
    - `Linear`：处理线性项；  
    - 训练/评估公共流程在该层复用。  
    DIN 等模型都继承该层，因此输入布局与 embedding 机制在全模型族共享。

3. **具体 CTR 架构层：`deepctr_torch/models/din.py`**  
    - `DIN(BaseModel)` 实现兴趣建模：query-key attention、序列 pooling、DNN 输出；  
    - `forward(X)` 假设输入 `X` 已按 `feature_index` 排列好。  
    这也是 Emblayer 改造保持“输出张量对齐”的关键约束：无论压缩协议如何变化，进入 DIN 的 `X` 语义必须一致。

4. **实验/推理编排层：`examples/benchmark_dinemblayer_infer.py` 与 `examples/batchscheduler.py`**  
    - 前者负责模式构建、host view 构造、扩展算子调用；  
    - 后者负责在线到达模拟、动态批调度、缓存与指标统计。  
    该层是“工程化接线层”，把底层模型能力映射到在线场景。

5. **编译期图改写层：converter 脚本**  
    - `onnx_multislice_converter.py`、`onnx_emblayerseq_converter.py`、`onnx_emblayer_joint_converter.py`  
    在 ONNX 层把碎片化 Slice 替换为聚合自定义算子节点，保障部署图与运行时协议一致。

综合来看，CTR 推理链路中“业务语义层（DIN）”与“执行优化层（Emblayer + Scheduler）”边界清晰：前者定义**算什么**，后者优化**怎么更快地算**。这也是本次改造能够在不改动模型语义的前提下，直接提升在线性能的结构性原因。

---

## 4.1 模块整体设计

DIN 在高并发推理下的主要瓶颈并不只在模型计算本身，还包括：

1. **输入表达冗余**：序列特征以固定 `max_seq_len` 做 padding，导致 H2D 传输与显存写入中存在大量无效零值。  
2. **图内切分代价**：baseline 图中存在大量 axis=1 的 `Slice`，其 kernel launch 与中间张量访问开销会放大尾延迟。  
3. **在线调度不确定性**：请求到达过程具备泊松随机性，批大小、排队时间、CPU 侧拼接、H2D 与 GPU 执行之间存在动态耦合。

本项目采用“**离线图改写 + 在线调度仿真**”双层方案：

- 离线侧：通过 converter 将 ONNX 中可识别的 Slice 结构替换为 `MultiSlice` / `EmblayerSeq` / `EmblayerVec` 聚合节点，降低图内碎片化操作。
- 在线侧：通过 BatchScheduler 在可控到达率下模拟真实生产队列，统一观测吞吐、平均时延、尾时延、队列等待、H2D 气泡与 GPU 利用率。

从系统分层看，数据路径可抽象为：

$$
\text{Request Stream} \rightarrow \text{BatchScheduler} \rightarrow \text{Host Input View} \rightarrow \text{H2D} \rightarrow \text{Emblayer Rebuild} \rightarrow \text{DIN Forward}
$$

其中 Emblayer 负责“输入重建算子化”，BatchScheduler 负责“负载生成与时序驱动”。

### 4.1.1 Emblayer算子整体设计（模块作用与输入输出）

Emblayer 的核心定位是将“CPU 侧提前展开好的稠密输入”替换为“压缩表达 + GPU 侧重建”，即把输入预处理从高冗余搬运路径转移到可并行的 CUDA kernel 路径。该设计在模块中承担三类职责：

- **序列重建（EmblayerSeq）**：从 `seq_values + seq_prefix + seq_lengths (+ seq_offsets)` 恢复每个样本每个序列特征的 token 段，再写回 DIN 输入矩阵指定列区间。
- **向量重建（EmblayerVec）**：从 `vec_values + vec_prefix + vec_indices` 恢复非序列特征的稠密行向量。
- **联合重建（Joint / JointV2）**：在同一推理步中先完成 vec 重建，再完成 seq 回填，输出完整 `x` 后进入 DIN 主干。

从接口角度，Emblayer 的输入输出可归纳为：

- 输入（压缩态）
  - 序列：`seq_values`, `seq_prefix`, `seq_lengths`, `seq_offsets`
  - 向量：`vec_values`, `vec_prefix`, `vec_indices`
  - V2 分拆：`user_vec_values`, `user_vec_indices`, `item_vec_values`, `item_vec_indices_template`
- 输出（稠密态）
  - `x in R^{B x D}` 或中间 `R^{B x N_seq x L}` 张量（按算子模式不同）

在 `benchmark_dinemblayer_infer.py` 中，这一职责具体映射到：

- `DINSeqOnlyInfer.forward(...)` 调用 `emblayer_seq_fused_fast(...)`
- `DINJointInfer.forward(...)` 调用 `emblayer_vec_fast(...) + emblayer_seq_fused_fast(...)`
- `DINJointInferV2.forward(...)` 优先走 `emblayer_vec_v2_regular_fast(...)`

这意味着 Emblayer 在模块中的本质作用不是改变模型语义，而是改变“输入恢复的实现位置与并行方式”。

### 4.1.2 BatchScheduler调度器整体设计（模块作用与输入输出）

BatchScheduler 的定位是“在线到达模拟 + 批处理调度 + 指标采样”，不是业务 RPC 服务本身，而是用于接近线上行为的压力驱动器。它的核心贡献是让算子收益在**动态批与排队条件**下被观察到，而不是仅在固定 batch 的离线 benchmark 中观察。

在 `examples/batchscheduler.py` 中，调度器输入包括：

- 到达过程参数：`arrival_rate`, `duration_s`, `seed`
- 调度策略参数：`max_batch_size`, `max_wait_ms`
- 数据形态参数：`max_seq_len`, `avg_seq_len`, `num_sparse`, `num_seq`
- 执行模式参数：`modes`（baseline/multislice/joint/joint_v2）

输出为 `RunResult`，覆盖端到端和分项指标：

- 吞吐与时延：`throughput_qps`, `avg_ms`, `p50/p95/p99`
- 队列与批：`avg_batch_size`, `avg_queue_ms`
- H2D 分析：`avg_h2d_only_ms`, `avg_h2d_pure_ms`, `h2d_bubble_ms`, `h2d_bubble_ratio_pct`
- 设备利用率：`sm_util_avg_pct`, `gpu_util_avg_pct`
- 输入规模：`avg_h2d_total_numel`, `avg_h2d_data_numel`, `avg_h2d_meta_numel`

该调度器通过以下机制逼近线上：

1. 泊松到达（指数间隔）建模请求随机性；
2. “满批/超时/worker 空闲”三触发调度策略；
3. 模式化 host view + H2D cache + 推理模型分发；
4. `nvidia-smi dmon` 并行采样 SM/GPU 利用率。

### 4.1.3 模块收益分析

联合模块收益可拆为“搬运收益 + 图执行收益 + 调度收益”。

1) **搬运收益（输入压缩）**  
设基线输入字节为 $B_{base}$，压缩模式输入字节为 $B_{mode}$，压缩率为：

$$
R_{compress} = 1 - \frac{B_{mode}}{B_{base}}
$$

`build_inputs()` 中 `bytes_baseline / bytes_joint / bytes_joint_v2` 的统计正是该收益的静态度量。

2) **H2D 气泡收益（调度/搬运协同）**  
脚本定义：

$$
\text{Bubble} = \max(\text{H2D-only} - \text{Pure-H2D}, 0)
$$

$$
\text{BubbleRatio}(\%) = \frac{\text{Bubble}}{\text{H2D-only}} \times 100
$$

Bubble 越低，说明主机准备、元数据处理、拷贝调度越接近“纯搬运极限”。

3) **在线服务收益（尾时延约束下吞吐）**  
对于目标 $p99 \le T$，可行吞吐定义为：

$$
Q^*(T) = \max_{r \in \mathcal{R},\ p99(r) \le T} qps(r)
$$

该指标比单点 `avg_ms` 更接近线上 SLA 视角。

从给定 long-run 结果看，`emblayerV2` 在高负载区间（例如 3000/4000）具有更好的尾时延与均值时延，说明“压缩输入 + 更规整的 V2 向量重建”在重载下更稳定；但由于每个 rate 仅单次聚合（`n=1`），中低负载下达标点会受随机采样扰动，需多次重复实验确认统计显著性。

---

## 4.2 Emblayer算子实现

### 4.2.1 设计目标（压缩数据，减少冗余）

Emblayer 的实现目标可以概括为三条：

1. **压缩表达优先**：避免把 padding 后的完整 `full` 全量搬运到 GPU；
2. **GPU 侧恢复**：将“按列拼接/回填”从 CPU 操作转为 CUDA 并行写入；
3. **模式可扩展**：支持 baseline 对齐（功能正确性）、joint（统一 vec+seq）、joint_v2（user/item 分拆复用）。

在实现上，`emblayer_seq.cpp` 与 `emblayer_vec.cpp` 都提供 slow/fast 两类入口：

- slow path：允许 CPU metadata，内部做检查与转换；
- fast path：要求 metadata 已在 CUDA 且连续，减少调度开销。

这使 benchmark/online 场景可以稳定使用 fast path，最大化体现算子本体能力。

### 4.2.2 数据压缩格式设计

`build_inputs()` 明确构建了多套输入视图，核心压缩格式如下。

#### 1) Seq 压缩格式

- `seq_values`: 所有样本、所有序列特征的 token 串接（1D）
- `seq_prefix`: 每个样本在“序列条目维度”的起止（长度 `B+1`）
- `seq_lengths`: 每条序列真实长度（长度 `B * num_seq_per_sample`）
- `seq_offsets`: 对 `seq_lengths` 的前缀和，用于定位 `seq_values` 子段

逻辑上：

$$
\text{seq\_offsets}[k+1] = \text{seq\_offsets}[k] + \text{seq\_lengths}[k]
$$

第 $k$ 条序列对应 token 区间：

$$
[\text{seq\_offsets}[k],\ \text{seq\_offsets}[k+1])
$$

#### 2) Vec 压缩格式（V1）

- `vec_values`: 非序列特征值按行展开
- `vec_indices`: 每个值对应输出列下标
- `vec_prefix`: 每行起止位置（CSR 风格）

即第 $b$ 行的非零段为：

$$
i \in [\text{vec\_prefix}[b],\ \text{vec\_prefix}[b+1])
$$

并按 `vec_indices[i]` 写回输出列。

#### 3) Vec 压缩格式（V2：user/item 分拆）

- user 特征在 batch 内共享，构建 `user_vec_values + user_vec_indices`，一次广播到所有行；
- item 特征按行变化，构建 `item_vec_values`；列模板固定为 `item_vec_indices_template`；
- 每行 item 非零个数固定为 `item_nnz_per_sample`，可去掉逐行 prefix 查找。

V2 的关键点是把“共享特征”与“逐行特征”拆开，减少重复元数据和分支判断，提升 kernel 规则性。

### 4.2.3 Cuda Kernel实现细节

#### A. EmblayerSeq

`emblayer_seq_cuda.cu` 中 `emblayer_seq_fused_kernel` 采用一维网格映射到三维逻辑空间 `(batch, seq_slot, token_pos)`：

- 每个线程计算一个 `(b, s, t)` 位置；
- 根据 `prefix` 判定该样本有效序列数；
- 根据 `seq_lengths` 判定 token 是否越界；
- 通过 `seq_offsets` 定位源地址；
- 通过 `dst_column_starts` 写入输出 `x[b, dst_col]`。

该实现将“seq 重建 + 列回填”融合为一次 kernel，避免先生成中间 3D 张量再二次 scatter。

#### B. EmblayerVec（V1）

`emblayer_vec_scatter_kernel` 对每个压缩元素 `idx`：

1. 在 `prefix` 上二分查找其所属行 `row`；
2. 读取 `col = feat_indices[idx]`；
3. 执行 `output[row, col] = vec_values[idx]`。

这种实现简单通用，但二分查找在 nnz 很大时会带来额外指令与分支开销。

#### C. EmblayerVec（V2 Regular）

`emblayer_vec_item_scatter_regular_kernel` 假设每行 item nnz 固定：

- `row = idx / item_nnz_per_sample`
- `off = idx % item_nnz_per_sample`
- `col = item_feat_indices_template[off]`

不再需要 prefix 二分，配合 `user_broadcast_kernel` 先写共享列，可显著提升写入路径的规则性与吞吐稳定性。这也是 `joint_v2` 在高负载区间更稳的重要实现原因之一。

#### 4.2.3 伪代码（实现主流程）

```pseudo
function EmblayerSeqFusedForward(seq_values, seq_prefix, seq_lengths, seq_offsets,
                                 dst_starts, output, num_seq_features, max_seq_len):
    parallel for idx in [0, B * num_seq_features * max_seq_len):
        b, s, t = decode(idx)
        seq_begin = seq_prefix[b]
        seq_end   = seq_prefix[b + 1]
        if s >= (seq_end - seq_begin):
            continue
        seq_id = seq_begin + s
        if t >= seq_lengths[seq_id]:
            continue
        src = seq_offsets[seq_id] + t
        dst_col = dst_starts[s] + t
        output[b, dst_col] = seq_values[src]
```

```pseudo
function EmblayerVecV2RegularForward(user_values, user_idx,
                                     item_values, item_idx_template,
                                     item_nnz_per_sample, output_dim):
    output = fill(pad_value, [B, output_dim])

    parallel for (b, u) in [0, B * user_nnz):
        col = user_idx[u % user_nnz]
        output[b, col] = user_values[u % user_nnz]

    parallel for idx in [0, B * item_nnz_per_sample):
        row = idx / item_nnz_per_sample
        off = idx % item_nnz_per_sample
        col = item_idx_template[off]
        output[row, col] = item_values[idx]

    return output
```

### 4.2.4 编译阶段融合Converter实现

编译阶段融合的核心思想是：在 ONNX 图中识别 axis=1 的 Slice 模式，将多个细粒度 Slice 节点折叠为少量聚合节点，并把切片参数转化为节点属性。

#### A. `onnx_emblayerseq_converter.py`

- 扫描图中 `Slice` 节点；
- 仅匹配 `axes=[1], steps=[1]` 的一维列切片；
- 根据 `seq_ranges` 命中目标区间；
- 删除命中的 Slice，插入单个 `EmblayerSeq_aggregated`；
- 将 `seq_starts/seq_ends/max_seq_num/max_seq_len` 编码为属性。

#### B. `onnx_emblayer_joint_converter.py`

- 分别匹配 `vec_ranges` 与 `seq_ranges`；
- 对 vec 片段插入 `EmblayerVec_aggregated`，对 seq 片段插入 `EmblayerSeq_aggregated`；
- 自动补充图输入（压缩态输入张量）；
- 保留未命中子图，保证转换是局部且可控的。

#### C. `onnx_multislice_converter.py`

- 识别同一 data_input 下可分组的多个 Slice；
- 构造 `MultiSlice` 节点，属性中带 `starts/lengths`；
- 保持输出拓扑顺序，减少图中 Slice 节点数。

#### 4.2.4 伪代码（图改写）

```pseudo
function ConvertSliceToEmblayer(graph, vec_ranges, seq_ranges):
    const_map = collect_constant_tensors(graph)
    matched_vec = {}
    matched_seq = {}

    for node in graph.nodes:
        if not is_axis1_unit_slice(node, const_map):
            continue
        r = (start(node), end(node))
        if r in vec_ranges and r not in matched_vec:
            matched_vec[r] = node.output
            mark_remove(node)
        if r in seq_ranges and r not in matched_seq:
            matched_seq[r] = node.output
            mark_remove(node)

    if matched_vec:
        insert EmblayerVec_aggregated(inputs=[vec_values, vec_prefix, vec_indices],
                                      outputs=ordered_vec_outputs,
                                      attrs={vec_starts, vec_ends, output_dim})
    if matched_seq:
        insert EmblayerSeq_aggregated(inputs=[seq_values, seq_prefix, seq_lengths],
                                      outputs=ordered_seq_outputs,
                                      attrs={seq_starts, seq_ends, max_seq_num, max_seq_len})

    remove marked Slice nodes
    ensure custom domain opset exists
    return graph
```

---

## 4.3 BatchScheduler流式队列调度实现

### 4.3.1 设计目标（模拟真实线上环境）

BatchScheduler 的目标不是生成“理想吞吐上限”，而是复刻线上系统中的关键不确定性：

- 请求按泊松过程到达，批大小不恒定；
- 批触发受 `max_batch_size` 与 `max_wait_ms` 双约束；
- CPU 准备时间、H2D、GPU 执行串接形成端到端时延；
- 在不同输入模式下，H2D 元数据占比和重建开销会变化。

因此该调度器输出的 `p99` 与 `queue_ms` 对算子优化更敏感，能体现“离线快”到“在线稳”的差异。

### 4.3.2 在线调度状态机实现

调度循环可抽象为 4 态状态机：

1. **Arrival**：按指数间隔推进 `next_arrival`，新请求入队；
2. **Wait**：队列非空但未满足触发条件，短暂 sleep；
3. **Dispatch**：触发批构建（满批/超时/worker 空闲）；
4. **Execute**：concat 模拟 + H2D + infer + 同步，记录指标。

触发逻辑：

$$
\text{trigger} = \text{batch\_full} \lor \text{timeout\_reached} \lor (\text{worker\_idle} \land |Q| > 0)
$$

请求端到端时延分解为：

$$
L_{e2e} = L_{queue} + L_{concat} + L_{h2d} + L_{gpu}
$$

对应到实现中，`latencies_ms` 从 arrival 时间戳到 `t_exec_done` 直接测得，`queue_wait_ms` 由 `t_exec_start - arrival_t` 测得。

#### 4.3.2 伪代码（状态机）

```pseudo
while now < end_time or queue not empty:
    while now >= next_arrival and now < end_time:
        queue.push(new_request(sample_seq_len()))
        next_arrival += Exp(arrival_rate)

    if queue.empty():
        sleep_until_next_arrival()
        continue

    timeout_reached = (now - queue.front.arrival) >= max_wait
    batch_full = queue.size >= max_batch_size
    worker_idle = now >= last_gpu_finish

    if not (batch_full or timeout_reached or worker_idle):
        short_sleep()
        continue

    batch = pop_min(queue, max_batch_size)
    host_view = get_or_slice_host_cache(batch.size)
    h2d_est   = get_or_interpolate_h2d_cache(batch.size)

    sleep(concat_ms(host_view))
    t_exec_start = now()
    run_infer(mode, host_view)
    t_exec_done = now()

    last_gpu_finish = t_exec_done
    update_metrics(batch, t_exec_start, t_exec_done, h2d_est)
```

### 4.3.3 数据路径与内存优化

该实现的内存优化重点不在“单次最快”，而在“连续 sweep 下不抖动”：

1. **Host 侧多视图缓存**：`host_data_cache[batch_size]` 缓存 `(host_data, concat_ms, numel_breakdown)`，避免循环内重复构建。  
2. **大批切片复用**：若缺失某 `bs`，优先从更大 master batch 做 `_slice_host_data()`，减少频繁 `build_inputs()`。  
3. **Pinned Memory 路径**：借助 `pin_host_data_tensors()` 与 `_get_cached_host_tensor(..., pin=True)`，让 H2D 使用非阻塞拷贝。  
4. **元数据位宽控制**：`--use_int32` 下 prefix/indices/lengths 采用 int32，直接减少 H2D 元数据字节。  
5. **JointV2 元数据打包**：`_get_joint_v2_packed_meta()` 将多段 metadata 合并，降低 host 侧碎片化读取成本。

这一路径与 Emblayer 的压缩表达共同作用，使 H2D 侧从“全量 dense 输入”转向“数据 + 少量索引元信息”的结构。

### 4.3.4 Warmup 与缓存体系设计

`_prewarm_caches()` 在每个 mode 启动时预热多个战略批点（1、max、2^k、0.25/0.5/0.75 比例点），目标是避免在线循环首次命中时出现冷启动尖峰。其策略价值有三点：

1. **减少首次构建抖动**：提前构造 host view 与 H2D 基线；
2. **支持插值估计**：`_get_interpolated_metrics()` 对非战略批点线性插值；
3. **缩短 sweep 总时长**：避免每个 rate 都触发完整 benchmark 过程。

此外，模式 sweep 中有“击穿提前停止”机制：当 p99 远超目标上界时，停止该模式后续更高负载点，减少无效计算。

#### 4.3.4 伪代码（预热与缓存）

```pseudo
function Prewarm(mode, max_batch_size):
    strategic = {1, max_batch_size, powers_of_two, quarter_points}
    for bs in strategic:
        if bs not in host_cache:
            host_cache[bs] = build_and_pin_host_view(bs)
        if bs not in h2d_cache:
            h2d_cache[bs] = benchmark_h2d(mode, host_cache[bs])

function GetBatchArtifacts(bs):
    if bs in host_cache:
        host = host_cache[bs]
    else:
        master = nearest_larger_cached_bs(bs)
        host = slice_from_master(master, bs)
        host_cache[bs] = host

    if bs in h2d_cache:
        h2d = h2d_cache[bs]
    else:
        h2d = linear_interpolate(h2d_cache, bs)
        h2d_cache[bs] = h2d

    return host, h2d
```

---

## 4.4 实验与性能分析

### 4.4.1 实验环境

本节依据给定 long-run 汇总结果与仓库脚本上下文进行分析，关键环境要素如下：

- 平台：Linux + CUDA 可用环境（脚本强制 `device=cuda`）
- 模型：DIN（`num_sparse=200`, `num_seq=100`）
- 序列设置：`max_seq_len=128`, `avg_seq_len≈8`
- 调度器：`max_batch_size=1024`, `max_wait_ms` 小窗口（毫秒级）
- 评估维度：`qps_mean`, `avg_ms_mean`, `p99_ms_mean`, `pass_p99_ratio`
- p99 目标：300ms（来自汇总报告头部）

### 4.4.2 实验基线与实验配置选用

本实验采用四种模式对照：

- `baseline`：原始输入切分与回拼路径；
- `multislice`：向量切片聚合；
- `emblayerV1 (joint)`：vec+seq 压缩重建；
- `emblayerV2 (joint_v2)`：在 V1 基础上引入 user/item 分拆与 regular item scatter。

负载范围包括 steady 区间（2500~3000）与 stress 点（4000）。

对比原则：

1. 同 arrival_rate 下比较 `avg/p99`，看性能上限；
2. 同 SLA（p99<=300ms）下比较可达吞吐，看片上可用容量；
3. 比较 stress 点看退化斜率，评估过载韧性。

### 4.4.3 实验结果

#### A. 高负载（3000）与压力点（4000）对比

基于 `all_report.md` 聚合表可得（steady=3000, stress=4000）：

| mode | phase | qps_mean | avg_ms_mean | p99_ms_mean |
|---|---:|---:|---:|---:|
| baseline | 3000 | 3009.967 | 1269.373 | 2352.931 |
| emblayerV1 | 3000 | 3011.033 | 1372.692 | 2421.902 |
| emblayerV2 | 3000 | 3010.133 | 551.175 | 963.606 |
| multislice | 3000 | 3009.600 | 837.372 | 1495.756 |
| baseline | 4000 | 3984.000 | 6576.148 | 12232.047 |
| emblayerV1 | 4000 | 3979.433 | 6271.259 | 11728.643 |
| emblayerV2 | 4000 | 3982.100 | 5939.778 | 11052.325 |
| multislice | 4000 | 3987.667 | 6901.561 | 12657.522 |

结论：在高负载与压力点下，`emblayerV2` 的均值与尾延迟均为最优；`multislice` 次之（3000 点）但在 4000 点退化更快；`emblayerV1` 相比 baseline 在压力点有改善，但稳态高负载并不占优。

#### B. 稳态高负载区间（arrival_rate >= 2800）平均表现

按 steady 区间聚合：

- baseline：avg p99 = **1598.31ms**, avg latency = **854.31ms**
- emblayerV1：avg p99 = **1267.00ms**, avg latency = **733.17ms**
- emblayerV2：avg p99 = **476.31ms**, avg latency = **258.99ms**
- multislice：avg p99 = **713.78ms**, avg latency = **412.65ms**

该结果显示 `joint_v2` 在持续高压下有明显优势，符合其数据格式更规整、kernel 分支更少的设计预期。

#### C. p99<=300ms 达标能力（基于单次聚合）

在当前 long-run 聚合（每个 rate 的 `n=1`）下，steady 达标点数量如下：

- baseline：7 / 50（14.0%）
- emblayerV1：4 / 51（7.84%）
- emblayerV2：5 / 51（9.8%）
- multislice：6 / 51（11.76%）

需要强调：该统计受单次采样波动影响较大，不应直接解释为“模式绝对优劣排序”。更稳妥的方法是对关键 rate（如 2600/2700/2800）做多次重复并报告置信区间。

#### D. 解释与工程含义

1. **为何 V2 在高压更优**：user/item 分拆后，user 广播写入与 item regular scatter 的访存模式更稳定，减少 prefix 二分与不规则分支。  
2. **为何达标点会离散**：在线动态批调度下，排队时延与 batch 边界受随机到达扰动，单次运行中 p99 易出现“锯齿”。  
3. **对线上配置的启发**：建议将 `joint_v2 + int32 metadata + 预热缓存` 作为默认方案，再通过 `max_wait_ms` 与 `max_batch_size` 做 SLA-吞吐折中调优。

### 4.4.4 实验结果分析：对照理论收益解释 V2 显著延时收益

本节将 4.1~4.3 中的理论收益模型，与 4.4.3 的实测结果进行一一映射，解释为什么 `joint_v2` 能在高负载下显著降低时延。

#### A. 从端到端时延分解看 V2 的受益位置

在线链路可写为：

$$
L_{e2e}=L_{queue}+L_{concat}+L_{h2d}+L_{gpu}
$$

`joint_v2` 的优化并不是只作用于某一项，而是对 `L_{concat}`、`L_{h2d}`、`L_{gpu}` 同时施压：

1. `L_{concat}`：输入协议更规整（user/item 分拆、模板化索引）后，主机侧准备与切片逻辑更少；  
2. `L_{h2d}`：压缩输入减少数据搬运体积，`int32` 元数据进一步降低 meta 负担；  
3. `L_{gpu}`：V2 regular kernel 取消 prefix 二分路径，线程执行形态更稳定，长尾抖动更小。

这解释了为什么 V2 在平均时延与 p99 两个维度都优于 baseline/multislice/V1。

#### B. 与实测数据的定量对照

根据 4.4.3 表格：

- 在 `arrival_rate=3000`：  
    - avg 从 `1269.373ms` 降到 `551.175ms`，降幅约

$$
\frac{1269.373-551.175}{1269.373}\approx 56.6\%
$$

    - p99 从 `2352.931ms` 降到 `963.606ms`，降幅约

$$
\frac{2352.931-963.606}{2352.931}\approx 59.0\%
$$

- 在 stress `arrival_rate=4000`：  
    - avg 从 `6576.148ms` 降到 `5939.778ms`，降幅约 `9.7%`；  
    - p99 从 `12232.047ms` 降到 `11052.325ms`，降幅约 `9.6%`。

- 在 steady 高负载区间（`arrival_rate>=2800`）均值聚合：  
    - avg 从 `854.31ms` 降到 `258.99ms`，降幅约 `69.7%`；  
    - p99 从 `1598.31ms` 降到 `476.31ms`，降幅约 `70.2%`。

这些数字说明：V2 的优势在“接近饱和但尚未完全击穿”的高负载段最明显；进入极端过载后，排队项 `L_{queue}` 占比上升，架构优化仍有效但边际收益被稀释。

#### C. 为什么 V1 不如 V2 稳定：理论与实现的一致性

从实现看，V1 向量路径 `emblayer_vec_scatter_kernel` 需要对每个 nnz 元素在 prefix 上做行定位（二分查找），这会引入更多分支与指令；而 V2 regular 路径利用固定 `item_nnz_per_sample` 与 `item_feat_indices_template`，可直接通过整除/取模定位 `(row, col)`，执行流更规则。

因此在理论上，若定义向量重建开销近似为：

$$
C_{vec}^{V1}\sim O(\text{nnz}\cdot\log B),\quad C_{vec}^{V2}\sim O(\text{nnz})
$$

则当批量与并发提高时，V2 相对 V1 的开销优势会被放大；这与 4.4.3 中“高压区 V2 明显优于 V1”的现象一致。

#### D. 对理论收益模型的回扣

第 4.1.3 给出的收益框架可写为：

$$
\Delta L \approx \Delta L_{h2d} + \Delta L_{gpu} + \Delta L_{concat}
$$

对 `joint_v2` 而言：

- `\Delta L_{h2d}<0`：压缩输入 + int32 metadata；
- `\Delta L_{gpu}<0`：regular scatter 与广播写入降低不规则分支；
- `\Delta L_{concat}<0`：模板化元数据减少 host 侧准备成本。

三项同向叠加，最终体现为“在相近 QPS 下，avg 与 p99 同步下降”的结果。也因此，`joint_v2` 并非只优化单点 micro-kernel，而是形成了从输入协议到执行路径的系统性降延时闭环。

---

### 本章小结

本章给出了 Emblayer 与 BatchScheduler 的模块化设计与实现闭环：

- Emblayer 通过压缩输入表达与 CUDA 重建降低冗余搬运，并在 V2 中进一步提升路径规则性；
- BatchScheduler 通过在线状态机、缓存与预热机制，将算子收益映射为真实队列条件下的端到端收益；
- 在给定 long-run 数据上，`emblayerV2` 在高负载和压力点体现出更好的延迟韧性，是当前最具工程落地价值的模式。
