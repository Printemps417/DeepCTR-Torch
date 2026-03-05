# Emblayer 技术报告：数据压缩、CUDA 算子、Converter 融合与优化分析

日期：2026-02-25  
仓库：DeepCTR-Torch  
核心实现：
- `examples/emblayer_op/emblayer_vec.cpp`
- `examples/emblayer_op/emblayer_vec_cuda.cu`
- `examples/emblayer_op/emblayer_seq.cpp`
- `examples/emblayer_op/emblayer_seq_cuda.cu`
- `examples/onnx_emblayer_joint_converter.py`
- `examples/onnx_emblayerseq_converter.py`
- `examples/benchmark_dinemblayer_infer.py`
- `examples/eval_baseline_vs_emblayerv2_throughput.py`

---

## 摘要

本文系统介绍 Emblayer 在 DIN 推理路径中的工程化实现，重点覆盖四个方面：

1. **数据压缩格式设计**：将原始稠密输入重表示为“数据载荷 + 元信息”，降低 Host-to-Device（H2D）传输与 CPU 端拼接负担。  
2. **CUDA 算子实现**：基于定制 kernel 完成向量特征散射重建与序列特征回填，支持 fused 写回路径。  
3. **Embedding Input 层融合 Converter**：通过 ONNX 图变换将多 Slice 子图替换为聚合自定义节点（EmblayerVec / EmblayerSeq）。  
4. **优化点分析**：现有实现的 `int32` 元信息、`packed_meta`、算子融合策略，以及“半精度 meta”进一步优化的可行路线。

从已有离线与在线实验结果可见，EmblayerV2 在保持 p99 SLA 的情况下可维持与 baseline 同量级吞吐，并显著降低 H2D 输入元素规模，说明该设计在 IO 受限场景具有较高工程价值。

---

## 1. 问题背景与目标

在 DIN 等推荐模型中，输入特征包含大量稀疏 ID 与可变长历史序列。传统 `full[B, D]` 稠密输入路径的主要问题是：

- 无效搬运：padding 区域和未命中列仍占用带宽；
- CPU 侧拼接与切片开销高；
- ONNX 图中多 Slice/Concat 节点造成图碎片化与调度开销。

Emblayer 的目标是在**不改变模型语义**的前提下，将输入阶段重构为“压缩表示 + GPU 重建”，并配合图级替换减少运行时冗余操作。

---

## 2. 记号与输入组织

设：

- $B$：batch 大小；
- $D$：模型总输入列宽；
- $S$：每样本序列特征数（`num_seq_per_sample`）；
- $L_{max}$：序列最大长度；
- $nnz$：非零（或有效）元素数。

baseline 输入为 $X \in \mathbb{R}^{B \times D}$。Emblayer 将其拆分为：

- 向量特征压缩表示（vec）；
- 序列特征压缩表示（seq）；
- 必要索引元信息（prefix/indices/lengths/offsets）。

---

## 3. 数据压缩格式设计

### 3.1 向量特征（EmblayerVec）

在实现中，vec 路径使用 CSR-like 表示：

- `vec_values`：按行拼接的有效特征值；
- `vec_prefix`：长度为 $B+1$ 的前缀数组，`vec_prefix[i+1]-vec_prefix[i]` 为第 $i$ 行有效元素个数；
- `feat_indices`：与 `vec_values` 同长，记录每个值写回到 $D$ 维向量中的列号。

该结构由 `build_inputs()` 生成（`benchmark_dinemblayer_infer.py`），并在 C++ 接口层校验单调性、范围与维度合法性。

### 3.2 序列特征（EmblayerSeq）

seq 路径采用单 iobuffer 扁平格式：

- `seq_values`：所有 token 串接后的 1D 缓冲；
- `seq_prefix`：样本到序列组的前缀（通常等差，步长为 $S$）；
- `seq_lengths`：每条序列实际长度；
- `seq_offsets`：`seq_lengths` 的前缀和，用于从 `seq_values` 定位。

逻辑上可重建为 $[B, S, L_{max}]$（或 fused 直接写回 $[B, D]$）。

### 3.3 V2 结构：user/item 拆分

`joint_v2` 在 vec 侧进一步拆分：

- user 侧特征：批内共享，仅存储一份并按 batch 广播；
- item 侧特征：按样本变化，可使用模板列索引（`item_feat_indices_template`）复用。

该设计减少重复元信息，尤其在“用户特征稳定、物品特征变化”场景中对带宽更友好。

### 3.4 H2D 成本分解

项目中按 data/meta 分别统计 H2D 元素与字节数。总体可写为：

$$
\text{H2DBytes} = \sum_k \text{numel}(T_k) \cdot \text{sizeof}(T_k)
$$

并通过名称规则将 `prefix/indices/lengths/offsets` 归类为 meta。该指标用于解释吞吐变化的根因（而非仅看端到端时延）。

---

## 4. CUDA 算子实现细节

### 4.1 EmblayerVec：scatter kernel

`emblayer_vec_scatter_kernel` 以 `idx in [0, nnz)` 为并行粒度：

1. 对 `prefix` 做二分查找，确定样本行号 `row`；
2. 取 `col = feat_indices[idx]`；
3. 执行 `output[row * output_dim + col] = vec_values[idx]`。

时间复杂度约为 $O(nnz \log B)$（二分定位行号）。尽管二分存在分支，但在较大 $D$、较稀疏场景下可有效避免 full 输入搬运。

### 4.2 EmblayerSeq：展开重建 kernel

`emblayer_seq_kernel` 将线性线程索引映射到三元坐标：

- `batch_id`、`seq_slot`、`token_pos`。

通过 `seq_prefix` 确定序列范围，利用 `seq_lengths` 与 `seq_offsets` 判断越界并取源索引，最后写入 padded 输出。其并行映射简单，访存模式可预测，适合大批量吞吐。

### 4.3 Fused 路径：序列直接写入最终输入

`emblayer_seq_fused_kernel` 不再生成中间 `seq_padded`，而是根据 `dst_column_starts` 直接将 token 写入最终 `output[B, D]` 指定列。

优势：

- 减少一次中间 Tensor 分配；
- 减少一次显式回填循环；
- 降低 global memory 额外读写。

该路径在 Python 推理模块中通过 `emblayer_seq_fused_fast` 调用，属于关键融合优化。

### 4.4 接口层与 fast path 设计

C++ 封装区分两类入口：

- **普通路径**：允许 CPU 元信息输入，执行更多合法性检查与推导；
- **fast 路径**：要求元信息已在 CUDA（且连续、类型合法），直接进入 kernel。

该分层便于开发调试与生产性能兼顾。

---

## 5. Embedding Input 融合 Converter 算法

### 5.1 目标

将 ONNX 图中沿 axis=1 的多 `Slice` 节点替换为聚合自定义算子：

- `EmblayerVec`：处理非序列列范围；
- `EmblayerSeq`：处理序列列范围。

### 5.2 算法流程

以 `onnx_emblayer_joint_converter.py` 为例：

1. 读取 `initializer` 与 `Constant`，构建常量值映射；
2. 解析 `Slice(starts, ends, axes, steps)`，筛选 axis=1、step=1 的候选；
3. 根据用户配置的 `vec_ranges/seq_ranges` 进行命中匹配；
4. 删除命中的原 Slice 节点；
5. 插入聚合节点：
   - `EmblayerVec(inputs=[vec_values, vec_prefix, vec_indices], attrs={vec_starts, vec_ends,...})`
   - `EmblayerSeq(inputs=[seq_values, seq_prefix, seq_lengths], attrs={seq_starts, seq_ends,...})`
6. 补充自定义 domain opset（默认 `com.deepctr`）。

### 5.3 复杂度与工程收益

- 转图复杂度近似线性于节点数：$O(|V|+|E|)$；
- 运行时可减少多节点图开销，简化执行计划；
- 与运行时压缩输入协议一致，便于调度与推理接口统一。

---

## 6. 优化点分析

### 6.1 已落地优化 A：`int32` 元信息

代码支持使用 `int32` 存储 `prefix/indices/lengths/offsets`，通过 `--use_int32` 控制。对同等元素数，meta 字节数约减半（相较 int64），通常可带来更低 H2D 开销与更低 CPU 准备开销。

### 6.2 已落地优化 B：`packed_meta`

`joint_v2` 将多段元信息（如 `user_vec_indices`、`seq_prefix`、`seq_lengths`）先在 host 端拼接成单一连续 buffer，再执行一次 H2D 拷贝，device 端按偏移切片复用。

该策略减少了多小张量拷贝引入的 launch/调度开销，尤其在高 QPS 小批量情况下收益明显。

### 6.3 已落地优化 C：算子融合

两条关键融合路径：

1. `emblayer_seq_fused_fast`：seq 直接写入 `x[B, D]`；
2. `emblayer_vec_v2_regular_fast`：item 侧固定 nnz 时使用模板索引，避免每批构造完整重复索引。

本质上均是在降低中间态和重复元信息构造成本。

### 6.4 “半精度 meta”现状与可行路线

当前主干实现**未直接采用 fp16/half 存储索引元信息**。原因是索引属于离散整数语义，若直接用 IEEE 半精度表示，可能出现不可逆或越界风险。

可行的下一步方案是“**窄位宽整数化**”而非“浮点半精度化”：

- 方案 1：`uint16 + base` 分块编码（每块存 base，块内存 delta）；
- 方案 2：对可证明范围的索引字段采用 `int16/uint16`，kernel 内无损恢复为 `int32`；
- 方案 3：运行前自适应选择 `int16/int32/int64`（按范围判定），在吞吐与安全性间折中。

该方向可在不牺牲索引正确性的前提下继续压缩 meta 带宽。

---

## 7. 实验证据与结果解读

### 7.1 离线吞吐（p99 <= 100 ms）

基于 `examples/tmp_eval_quick_after/offline_report.md`：

- baseline：QPS = 18226.87，avg = 56.18 ms，p99 = 56.44 ms；
- emblayerV2：QPS = 18370.64，avg = 55.74 ms，p99 = 56.31 ms。

结论：在该配置下，EmblayerV2 与 baseline 吞吐同量级并略优，说明压缩/重建开销未抵消其输入侧收益。

### 7.2 在线到达模拟（BatchScheduler）

基于 `examples/tmp_scheduler_quick_after/online_report.md`，在到达率 2600、p99 约束 100ms 下：

- baseline 吞吐 2591 QPS，H2D Total Numel = 1,347,527；
- emblayerV2 吞吐 2596 QPS，H2D Total Numel = 120,484（Data=89,793，Meta=30,692）。

可见在吞吐基本持平时，输入元素规模显著下降，验证了压缩协议的有效性。

### 7.3 关于 bubble 指标的解释

部分场景下 EmblayerV2 的 H2D bubble ratio 可能高于 baseline。该现象并不直接代表方案退化，通常与以下因素叠加有关：

- 多张量搬运与调度粒度变化；
- 小批量下 kernel/拷贝 launch 固定开销占比提升；
- 队列调度策略（到达率、batch 上限、等待时间）导致的阶段性放大。

因此应联合观察 QPS、p99、H2D total/data/meta 与 SM/GPU 利用率，而非单看 bubble。

---

## 8. 正确性与鲁棒性保证

实现中通过多层校验降低 silent error 风险：

- 类型约束：索引元信息必须为 `int32/int64`；
- 结构约束：`prefix` 单调、边界一致、`sum(lengths)==values.numel()`；
- 维度约束：输出维度、最大序列长度、范围合法。

在 demo 与 benchmark 中均有 baseline 对齐检查（如输出最大绝对误差验证），支持功能正确性回归。

---

## 9. 局限性与后续工作

### 9.1 局限性

- vec kernel 使用二分找行，理论上存在分支与不规则访存；
- 不同模式下最优批量区间不同，在线调度需与模型/硬件联合调优；
- 当前“半精度 meta”尚未落地为主干方案。

### 9.2 后续方向

1. vec 行号定位优化（warp 协作或分块 prefix 索引）；  
2. meta 自适应位宽压缩（`int16/int32` 动态切换）；  
3. packed_meta 与异步流进一步融合，减少小拷贝间隙；  
4. 建立统一的 roofline/带宽模型，指导不同负载下的模式选择。

---

## 10. 结论

Emblayer 方案通过“输入压缩表示 + 自定义 CUDA 重建 + ONNX 图级融合”形成了完整闭环：

- 在实现层面，已具备向量/序列双路径重建能力与 fast/fused 接口；
- 在图层面，可将多 Slice 子图替换为聚合节点，降低图碎片化；
- 在性能层面，已有结果显示其可在满足延迟约束下维持或提升吞吐，并显著缩减 H2D 输入规模。

从工程实践角度，EmblayerV2 已具备在 IO 受限推理链路中推广的可行性；后续可通过窄位宽整数 meta 与更深度融合进一步释放性能潜力。

---

## 附录 A：术语对照

- baseline：原始稠密输入路径。  
- emblayerV1（`joint`）：`EmblayerVec + EmblayerSeq` 联合重建。  
- emblayerV2（`joint_v2`）：在 V1 基础上增加 user/item 拆分与模板化优化。  
- H2D bubble：`h2d_only - pure_h2d`，反映额外调度/准备开销。  

## 附录 B：复现建议

可使用以下脚本快速复现实验：

- 离线吞吐：`examples/eval_baseline_vs_emblayerv2_throughput.py`  
- 在线仿真：`examples/batchscheduler.py`  
- 一体化 benchmark：`examples/benchmark_dinemblayer_infer.py`
