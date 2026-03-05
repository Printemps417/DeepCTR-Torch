# 生成式推荐场景下 Emblayer 算子优化架构报告（基于 DeepCTR-Torch 代码实现）

## 摘要

本文面向生成式推荐（Generative Recommendation）在线推理链路，系统梳理 DeepCTR-Torch 中 `emblayer` 系列算子（`EmblayerVec`、`EmblayerSeq`）的工程实现与理论收益。报告严格基于代码实现展开，覆盖以下核心内容：

1. 面向推荐稀疏输入的压缩格式设计与内存布局；
2. CUDA kernel 级算法实现（scatter/broadcast/fused writeback）；
3. ONNX 图级 Embedding Input 融合 Converter 算法；
4. 端到端工程收益分解公式（H2D、元数据、拼接开销、吞吐）；
5. 深入优化方向（meta 半精度、算子融合、索引压缩策略）及其适用边界。

本文只讨论架构设计和理论收益，不包含实验结果与数值对比。

---

## 1. 问题定义与优化目标

### 1.1 生成式推荐中的输入瓶颈

在生成式推荐推理中，样本输入通常具有以下特征：

- 非序列稀疏特征数量大（用户、上下文、候选 item 特征）；
- 行为序列特征长度分布长尾，padding 导致显著无效传输；
- 在线服务下 batch size 波动，CPU 侧拼接和 metadata 构造频繁；
- GPU 推理并非总是纯计算瓶颈，常受 H2D（Host-to-Device）与数据编排影响。

传统输入构建方式是“全宽矩阵 + 大量 Slice/Concat + padded seq tensor”，会造成：

1. 传输字节放大；
2. 元数据冗余；
3. kernel 启动与图节点冗余；
4. CPU/GPU 协同开销高。

### 1.2 Emblayer 的优化目标

`Emblayer` 的核心目标是将输入处理从“按列切分”转为“按语义压缩+算子重建”，具体包括：

- **Vec 侧**：将非序列特征压缩成一维值流 + 索引流，GPU 端重建二维输入；
- **Seq 侧**：将变长序列压缩成单 IO buffer + prefix/length/offset，GPU 端按需回填；
- **融合侧**：在同一输出张量上完成 seq 回填，减少中间张量与额外 kernel；
- **图侧**：将多 Slice 节点融合为自定义 ONNX 节点，降低图复杂度与执行框架调度开销。

---

## 2. 代码实现全景与分层架构

### 2.1 关键模块

- CUDA/C++ 扩展：
  - `examples/emblayer_op/emblayer_vec_cuda.cu`
  - `examples/emblayer_op/emblayer_seq_cuda.cu`
  - `examples/emblayer_op/emblayer_vec.cpp`
  - `examples/emblayer_op/emblayer_seq.cpp`
- 输入压缩与模式编排：
  - `examples/benchmark_dinemblayer_infer.py`
- 图转换器（Embedding Input 融合 Converter）：
  - `examples/onnx_emblayer_joint_converter.py`
  - `examples/onnx_emblayerseq_converter.py`
- 在线调度与收益指标分解：
  - `examples/batchscheduler.py`

### 2.2 五类路径（代码中显式定义）

`benchmark_dinemblayer_infer.py` 中定义了 `baseline / multislice / multislice_seq / seq_only / joint / joint_v2` 六条路径。下面按“相对 baseline 的改动 + 优化点”展开说明。

1. `baseline`：该优化等级下沿用原始模型输入语义，直接传输全量 `full[B, D]`，并由原模型按 `feature_index` 做切片与后续处理。
   - 相对原始定义的改动：无结构性改动，仅作为对照路径。
   - 优化点：无额外优化，作用是提供字节、H2D、时延与吞吐的基线参考。

2. `multislice`：该优化等级下，对 base 模型中的**非序列特征 slice 进行聚合**，使非序列 input 可通过连续 iobuffer (`compact`) 一次性取出，再回填到原列区间。
   - 相对 baseline 的改动：
     - 新增 `starts/span_lengths` 元数据；
     - 用 `multislice_ext.multislice(full_input, starts, span_lengths)` 代替多次 Python 级 slice；
     - 通过 `compact_layout` 将聚合结果写回到模型预期列位置。
   - 优化点：
     - 降低非序列切片的调度与访存碎片；
     - 为非序列输入形成更连续的数据访问模式，改善 H2D 与后续回填效率；
     - 减少大量小切片操作带来的 CPU 开销。

3. `multislice_seq`：该优化等级下，非序列沿用 `multislice` 聚合，序列侧改为 EmblayerSeq 压缩输入并在 GPU 端重建回填。
   - 相对 baseline 的改动：
     - 非序列从原始分散切片改为 `multislice`；
     - 序列输入从固定宽度 padded 矩阵改为 `seq_values + seq_prefix + seq_lengths + seq_offsets`；
     - 通过 `emblayer_seq_fused_fast` 直接写入输出 `x[B, D]` 的序列列。
   - 优化点：
     - 非序列和序列同时压缩，减少总输入字节；
     - 减少序列 padding 搬运；
     - 将“序列重建 + 回填”融合，降低中间张量开销。

4. `seq_only`：该优化等级下，非序列不走 MultiSlice，而是直接使用预先拼接好的 `non_seq_compact`；序列侧继续使用 EmblayerSeq fused。
   - 相对 baseline 的改动：
     - 非序列从 `full` 改为 `non_seq_compact`（仅保留非序列有效列，连续存放）；
     - 初始化 `x[B, D]` 后按 `compact_layout` 回填非序列；
     - 序列通过 `emblayer_seq_fused_fast` 直接补齐序列列。
   - 优化点：
     - 进一步减少非序列无效传输（相较 full 输入）；
     - 避免 Python 侧逐列拼接，降低准备开销；
     - 序列仍保持 fused 路径，控制 kernel 与内存开销。

5. `joint`：该优化等级下，非序列采用 EmblayerVec，序列采用 EmblayerSeq，形成 Vec+Seq 双 Emblayer 联合重建。
   - 相对 baseline 的改动：
     - 非序列从 dense `full` 改为 `vec_values + vec_prefix + vec_indices`（CSR 风格）；
     - 先用 `emblayer_vec_fast` 构建/填充 `x[B, D]` 的非序列部分；
     - 再用 `emblayer_seq_fused_fast` 将序列列写入同一 `x`；
     - 模型前向保持不变，仅替换输入重建路径。
   - 优化点：
     - 将非序列与序列都压缩为“值流+元数据”协议，显著降低输入字节；
     - 统一在 GPU 端重建，减少 CPU 侧切片和拼接；
     - 具备更强的图融合与部署扩展性（可与 ONNX converter 对齐）。

6. `joint_v2`：该优化等级下，在 `joint` 基础上做 user/item 结构化分离，并引入 metadata 打包与规则模板索引。
   - 相对 baseline 的改动：
     - 非序列拆为 `user_vec_values/user_vec_indices`（batch 内共享）与 `item_vec_values/item_vec_indices_template`（候选相关）；
     - 用户侧由 `emblayer_vec_user_broadcast_kernel` 广播写入；
     - item 侧优先使用 `emblayer_vec_v2_regular_fast`（固定 `item_nnz_per_sample` 模板散射）；
     - 将 `user_vec_indices + seq_prefix + seq_lengths` 打包为 `packed_meta_indices`，再切片复用。
   - 优化点：
     - 利用“单用户多候选”结构先验，减少用户侧重复传输；
     - 模板化 item 索引降低 metadata 构建与 H2D 负担；
     - packed meta 减少小张量 copy/调度次数，抑制 H2D bubble；
     - 与 `--use_int32` 结合时，可进一步降低 metadata 字节。

总体上，这六条路径构成了从“仅做切片聚合”到“输入协议重构（Vec/Seq）”再到“业务结构先验驱动的 metadata 优化（V2）”的渐进式优化链路。

---

## 3. 数据压缩格式设计（推荐场景导向）

### 3.1 非序列特征压缩：EmblayerVec V1

#### 3.1.1 存储结构

代码中非序列压缩核心三元组：

- `vec_values`：按行拼接后的值流（float）；
- `vec_indices`：每个值对应的目标列索引（int32/int64）；
- `vec_prefix`：每行在值流中的起止偏移（CSR 风格前缀和）。

在 `build_inputs` 中，按样本行遍历 non-seq spans 构建上述三元组。

#### 3.1.2 推荐场景适配性

该结构适合“稀疏宽表 + 行稀疏分布变化”的推荐输入：

- 仅传输有效值，减少无效列搬运；
- 支持列索引任意布局，不要求物理连续；
- 能与固定维度 dense 特征统一编码（dense 也写入 `vec_values`）。

### 3.2 序列特征压缩：EmblayerSeq

#### 3.2.1 单缓冲序列格式

`build_inputs` 中序列压缩采用四元组：

- `seq_values`：所有样本、所有序列特征的 token 串联值流；
- `seq_prefix`：每个 batch 对应的序列槽起止（样本维度前缀）；
- `seq_lengths`：每个序列槽真实长度；
- `seq_offsets`：每个序列槽在 `seq_values` 内的偏移（token 级前缀和）。

该设计等价于“二级 ragged 编码”：先按样本定位序列槽，再按序列槽定位 token。

#### 3.2.2 推荐序列长尾问题的针对性

与固定 `max_seq_len` padding 相比，该格式只传输真实长度：

- 长尾短序列场景下，理论字节开销显著降低；
- 避免大规模 padding token 的 H2D 浪费；
- 兼容多序列特征并行（`num_seq_per_sample`）。

### 3.3 V2 用户/候选分离压缩（joint_v2）

`joint_v2` 在 Vec 路径上引入推荐系统典型结构假设：

- 用户侧特征在一个 batch 内常“共享/缓慢变化”；
- 候选 item 侧特征按样本变化。

因此构造：

- `user_vec_values + user_vec_indices`（仅一份，广播到 batch）；
- `item_vec_values + item_vec_indices_template`（每样本 item 值流，索引模板可复用）；
- metadata 打包：`packed_meta_indices = cat(user_vec_indices, seq_prefix, seq_lengths)`。

这种分离使 metadata 与数据带宽模型更接近线上“1 user 对 N candidate”的服务形态。

### 3.4 索引精度策略（int32/int64）

代码在多处支持 `--use_int32`：

- prefix / indices / lengths / offsets 可降为 int32；
- 通过 `_get_cached_host_tensor` 与 `_prepare_inputs` 进行统一转换与缓存；
- `joint_v2` 里通过 `_get_joint_v2_packed_meta` 做一次性打包，减少小张量 copy 次数。

若索引边界满足 $N<2^{31}$，int32 相比 int64 的 metadata 字节可近似减半。

---

## 4. CUDA 算法实现细节

## 4.1 EmblayerVec kernel 设计

### 4.1.1 算子定义（输入、输出、功能）

EmblayerVec 的目标是将“非序列压缩表示”重建为模型可直接消费的二维输入矩阵 `output[B, D]`。代码中包含三类执行形态：

1. **通用散射（`emblayer_vec_scatter_kernel`）**
  - 输入：`vec_values[nnz]`、`prefix[B+1]`、`feat_indices[nnz]`
  - 输出：`output[B, D]`
  - 功能：把每个非零值按 `(row, col)` 散射回目标列。

2. **用户侧广播（`emblayer_vec_user_broadcast_kernel`）**
  - 输入：`user_values[user_nnz]`、`user_feat_indices[user_nnz]`
  - 输出：`output[B, D]`
  - 功能：将 batch 内共享用户特征广播写入每一行。

3. **规则 item 散射（`emblayer_vec_item_scatter_regular_kernel`）**
  - 输入：`item_values[B * item_nnz_per_sample]`、`item_feat_indices_template[item_nnz_per_sample]`
  - 输出：`output[B, D]`
  - 功能：在“每行 item 特征个数固定”的场景下按模板位置写回。

对应接口关系：

- `emblayer_vec_fast`：调用通用散射；
- `emblayer_vec_v2_fast`：用户广播 + item 通用散射；
- `emblayer_vec_v2_regular_fast`：用户广播 + item 规则散射（优先路径）。

### 4.1.2 核心 kernel 实现与线程模型设计

1. **线程映射策略**
  - 通用散射：1 个线程处理 1 个 `idx in [0, nnz)`；
  - 用户广播：1 个线程处理 1 个 `(row, off)`，总线程数 `B * user_nnz`；
  - 规则散射：1 个线程处理 1 个 `(row, off)`，总线程数 `B * item_nnz_per_sample`。

2. **块配置**
  - 统一 `threads = 256`；
  - `blocks = ceil(total/threads)`；
  - 适合大规模元素级并行，调度简单且与输入规模线性扩展。

3. **索引恢复逻辑**
  - 通用散射使用 `prefix` 二分查找恢复 `row`，再由 `feat_indices[idx]` 得到 `col`；
  - 广播/规则散射通过除法与取模直接得到 `(row, off)`，省去二分；
  - V2 regular 路径通过模板索引避免逐元素存储 `item_feat_indices`。

4. **复杂度特征**
  - 通用散射约为 $O(nnz\log B)$（包含二分）；
  - 广播与规则散射约为 $O(B\cdot nnz_{row})$，索引恢复为常数开销；
  - 因此 V2 regular 在固定布局场景具有更稳定延迟。

### 4.1.3 fused 与冲突规避优化手段

1. **已实现的融合策略（V2 级联融合）**
  - 代码中并非把所有步骤塞进单一 kernel，而是采用“用户广播 kernel + item 散射 kernel”的两阶段流水；
  - 该设计在保持可维护性的同时，利用推荐业务先验（user 共享）减少传输冗余。

2. **metadata 优化协同**
  - `joint_v2` 配合 `item_feat_indices_template` 与 `packed_meta_indices`，减少小元数据张量与重复索引；
  - `--use_int32` 可直接降低索引带宽，进一步放大 Vec 路径收益。

3. **bank conflict 相关说明（基于当前实现）**
  - 当前 Vec kernel 不依赖 shared memory 做中间缓存，主要是寄存器 + 全局内存直接读写；
  - 因此不存在典型 shared-memory bank conflict 热点；
  - 主要瓶颈转为全局内存访问离散性与散射写回随机性，代码通过 regular 模板化路径来改善访问规律性。

4. **可继续优化方向（与现实现兼容）**
  - 使用 warp 级 prefix 分段缓存减少重复二分；
  - 对固定布局场景进行向量化加载/写回（如 `float2/float4`）；
  - 在更高层做 Vec+Seq 单核融合，减少一次输出张量往返。

## 4.2 EmblayerSeq kernel 设计

### 4.2.1 算子定义（输入、输出、功能）

EmblayerSeq 的目标是将 ragged 序列压缩表示映射回模型输入空间，支持“独立重建”和“融合回填”两种模式。

1. **基础重建（`emblayer_seq_kernel`）**
  - 输入：`seq_values`、`prefix`、`seq_lengths`、`seq_offsets`
  - 输出：`output[B, max_seq_num, max_seq_len]`
  - 功能：将变长序列恢复为 3D padded 张量。

2. **融合回填（`emblayer_seq_fused_kernel`）**
  - 输入：`seq_values`、`prefix`、`seq_lengths`、`seq_offsets`、`dst_column_starts`
  - 输出：`output[B, D]`（由调用方预分配）
  - 功能：直接把序列 token 写到最终输入矩阵指定列，跳过中间 3D 张量。

接口层对应：

- `emblayer_seq_fast`：调用基础重建；
- `emblayer_seq_fused_fast`：调用融合回填，是联合路径主用实现。

### 4.2.2 核心 kernel 实现与线程模型设计

1. **统一线性展开**
  - 两类 kernel 都采用 1D 网格：`idx = blockIdx.x * blockDim.x + threadIdx.x`；
  - `idx` 映射到 `(batch_id, seq_slot, token_pos)`，其中 `span_per_batch = num_seq_or_max_seq_num * max_seq_len`。

2. **有效性判定链**
  - 先由 `prefix` 得到当前 batch 的有效序列槽区间；
  - 判定 `seq_slot < seq_count`；
  - 判定 `token_pos < seq_len`；
  - 最终通过 `src_idx = seq_offsets[seq_idx] + token_pos` 读取源 token。

3. **写回策略差异**
  - 基础重建：写回 3D `output[idx]`；
  - 融合回填：计算 `dst_col = dst_column_starts[seq_slot] + token_pos`，写回 `output[batch_id, dst_col]`；
  - 融合模式附带边界检查，确保 `dst_col` 不越界。

4. **线程组织特点**
  - 每线程处理一个 token 位置，天然适配大规模序列并行；
  - 对空槽位与超长位置采用早退分支，避免无效写回；
  - 支持 int32/int64 前缀与偏移类型混合分发。

### 4.2.3 fused 与冲突规避优化手段

1. **已实现 fused 优化（核心收益点）**
  - `emblayer_seq_fused_kernel` 将“序列重建 + 列映射 + 回填”合并为一次 kernel；
  - 避免先生成 `seq_padded[B,S,L]` 再逐序列拷贝到 `x[B,D]` 的两段式流程；
  - 减少中间张量显存占用与额外内存带宽。

2. **与上层路径协同**
  - 在 `seq_only / multislice_seq / joint / joint_v2` 中，fused kernel 直接写入最终 `x`；
  - 与 Vec 重建结果在同一输出张量汇合，缩短数据路径。

3. **bank conflict 相关说明（基于当前实现）**
  - Seq kernel 同样未使用 shared memory 进行 tile 缓存，主要是寄存器 + 全局内存访问；
  - 因而没有典型 shared-memory bank conflict 问题；
  - 性能关键点更多在分支发散（不同 seq_len）与全局写回连续性，fused 写回通过固定列起点 `dst_column_starts` 改善了目标地址规律。

4. **可继续优化方向（工程可落地）**
  - 按 `seq_slot` 或长度分桶发射 kernel，降低 warp 分支发散；
  - 对常见短序列长度做专用 kernel 模板；
  - 在支持条件下引入 cp.async/shared-memory staging（需同步评估 bank 布局）。

## 4.3 dtype 分发与接口约束

两类扩展均采用 C++ 层 `fast` 入口进行轻量校验并直接进入 CUDA：

- 支持 int32/int64 索引分发；
- 支持输出与输入数据类型分发（`AT_DISPATCH_ALL_TYPES`）；
- fast path 假设 metadata 已在 CUDA 且 contiguous，减少 CPU 校验路径。

这与在线推理场景匹配：在上游保证输入规范，换取更低算子调用开销。

---

## 5. Embedding Input 融合 Converter 算法

## 5.1 ONNX Joint Converter（Vec+Seq）

`onnx_emblayer_joint_converter.py` 的核心步骤：

1. 扫描 graph initializer 与 Constant 节点，建立常量值映射；
2. 识别 axis=1、step=1 的 Slice 节点，并解析 `[start,end)`；
3. 依据用户给定 `vec_ranges` 与 `seq_ranges` 进行匹配分组；
4. 删除已匹配 Slice 节点；
5. 插入单个 `EmblayerVec` 节点与/或单个 `EmblayerSeq` 节点；
6. 将 custom domain (`com.deepctr`) 写入 opset。

这本质上是“同轴切片模式识别 + 子图替换”。

## 5.2 关键属性设计

### 5.2.1 EmblayerVec 节点属性

- `output_dim`：融合后输出宽度；
- `col_start/col_end`：覆盖列范围；
- `vec_starts/vec_ends`：各输出分片对应列区间；
- `pad_value`：缺省填充值。

### 5.2.2 EmblayerSeq 节点属性

- `max_seq_num`：序列特征槽数；
- `max_seq_len`：槽内最大长度；
- `seq_starts/seq_ends`：每个序列特征目标列区间；
- `output_2d=1`：输出直接对接二维输入回填语义。

## 5.3 Converter 的工程价值

- 大量 Slice 节点折叠为单节点，图规模减小；
- 减少执行引擎图调度与内存边界开销；
- 提供输入压缩协议与算子执行之间的稳定契约；
- 为后续 runtime 插件化（TensorRT/ORT 自定义 op）提供统一入口。

---

## 6. 工程收益计算公式（理论模型）

## 6.1 输入字节模型（代码同构）

在 `build_inputs` 中，已定义各模式输入字节：

- baseline:  
  $$\text{Bytes}_{base}=\text{nbytes}(full)$$
- seq_only:  
  $$\text{Bytes}_{seq}=\text{nbytes}(non\_seq\_compact)+\text{nbytes}(seq\_values)+\text{nbytes}(seq\_prefix)+\text{nbytes}(seq\_lengths)$$
- joint:  
  $$\text{Bytes}_{joint}=\text{nbytes}(vec\_values)+\text{nbytes}(vec\_indices)+\text{nbytes}(vec\_prefix)+\text{Bytes}_{seq\_part}$$
- joint_v2:  
  $$\text{Bytes}_{joint\_v2}=\text{nbytes}(user\_vec\_values)+\text{nbytes}(user\_vec\_indices)+\text{nbytes}(item\_vec\_values)+\text{nbytes}(item\_template)+\text{Bytes}_{seq\_part+offset}$$

其中 `seq_part+offset` 额外包含 `seq_offsets`。

定义理论压缩比：
$$R_{bytes}(mode)=\frac{\text{Bytes}_{base}}{\text{Bytes}_{mode}}$$

## 6.2 元数据占比模型

`h2d_numel_breakdown` 与 `h2d_bytes_breakdown` 将 `prefix/lengths/offsets/indices` 计为 meta。

定义：
$$\rho_{meta}=\frac{\text{MetaBytes}}{\text{TotalBytes}}$$

若索引从 int64 改为 int32（边界合法前提下）：
$$\text{MetaBytes}_{i32}\approx\frac{1}{2}\text{MetaBytes}_{i64}$$

则总字节改善近似：
$$\Delta_{total}\approx \rho_{meta}\cdot 50\%$$

即 meta 占比越高，int32 压缩收益越显著。

## 6.3 H2D 分解与 Bubble 公式

`batchscheduler.py` 中定义：

- 平均 H2D-only：$T_{h2d\_only}$
- 平均 Pure H2D：$T_{h2d\_pure}$
- H2D Bubble：
  $$T_{bubble}=\max(T_{h2d\_only}-T_{h2d\_pure},0)$$
- Bubble Ratio：
  $$\text{BubbleRatio}=\frac{T_{bubble}}{T_{h2d\_only}}\times 100\%$$

该分解将“纯拷贝成本”和“数据准备/拼接开销”显式区分，适合评估 Emblayer 对 CPU 侧准备开销的抑制效果。

## 6.4 端到端时延与吞吐模型

定义单批次总时延近似：
$$T_{e2e}=T_{queue}+T_{concat}+T_{h2d\_pure}+T_{compute}+T_{sched}$$

在批调度稳态下，吞吐近似：
$$QPS\approx\frac{\mathbb{E}[B]}{\mathbb{E}[T_{service}]}$$

其中 $T_{service}$ 可由上述分解项组成。`Emblayer` 主要优化的是：

1. 降低 $T_{concat}$（减少 CPU 切片拼接与 metadata 构造）；
2. 降低 $T_{h2d\_pure}$（减少传输字节）；
3. 部分降低 $T_{compute}$（融合后减少中间张量读写和 kernel 数量）。

## 6.5 理论 speedup 评估

若 baseline 与优化路径分别为 $T_{base},T_{opt}$，理论加速比：
$$S=\frac{T_{base}}{T_{opt}}$$

进一步用 Amdahl 视角拆分可优化比例 $p$：
$$S\le \frac{1}{(1-p)+\frac{p}{k}}$$

其中 $k$ 为可优化部分（H2D+concat+部分 compute）的缩减倍数。该公式可用于在架构评审阶段做上限估计。

---

## 7. 深入优化点分析

## 7.1 Meta 半精度（含边界约束）

### 7.1.1 现状

当前代码主要采用 `int32/int64` 作为 metadata 类型，`--use_int32` 已是第一层压缩。

### 7.1.2 “meta 半精度”的可行解释

在推荐输入语义中，严格意义上的“半精度”可分两类：

1. **值半精度**：将 `vec_values`（float）降为 fp16/bf16；
2. **索引半精度**：将索引压到 16bit（本质是 u16/i16 编码，不是浮点半精度）。

对 metadata 更可行的是“16bit 索引编码”：

- 对局部窗口索引（如相对列偏移、短前缀增量）使用 `uint16`；
- 结合分段 base-offset 恢复绝对索引；
- kernel 内做轻量解码。

### 7.1.3 理论收益

若 metadata 可从 32bit 压到 16bit：
$$\text{MetaBytes}_{16}\approx\frac{1}{2}\text{MetaBytes}_{32}$$

总收益：
$$\Delta_{total}\approx \rho_{meta}\cdot 50\%\quad(\rho_{meta}=\text{meta占比})$$

### 7.1.4 风险与约束

- 索引范围溢出风险（需分段编码）；
- 解码开销可能抵消部分收益；
- 需要 converter 与 runtime 共同升级协议。

结论：`meta 半精度` 在“高 metadata 占比 + 范围可控”的在线场景具备工程价值，建议作为 V3 协议演进点。

## 7.2 算子融合深化

当前已实现融合：`emblayer_seq_fused_fast` 直接写入最终 `x[B,D]`。可进一步演进：

1. **Vec+Seq 单 kernel 融合**：在同一 kernel 中先填非序列列，再填序列列，减少一次全量 `x` 读写；
2. **填充 + cast 融合**：在写入阶段完成 dtype cast，避免额外转换；
3. **prefix 解码共享**：对同 row 元素采用 warp/block 级缓存，降低重复二分开销。

理论上，融合可减少：

- kernel launch 次数；
- 中间张量带宽；
- L2/DRAM 往返次数。

## 7.3 Metadata 打包与小张量合并

`joint_v2` 已实现 `packed_meta_indices`（`user_vec_indices + seq_prefix + seq_lengths`）一次拷贝。

这类优化的本质是减少“多小张量传输/调度”开销：

- 降低 driver 与 runtime API 调度次数；
- 减少非连续拷贝导致的延迟抖动；
- 为异步流水（copy-compute overlap）提供更大粒度数据块。

## 7.4 用户侧共享特征广播机制

`emblayer_vec_user_broadcast_kernel` 将用户侧特征视作 batch 共享，避免逐样本重复传输。

在生成式推荐重排场景（单用户多候选）中，该机制理论收益随候选数 $N$ 增大而增加：

- 传输从 $O(N\cdot U)$ 降为 $O(U)+O(N\cdot I)$；
- 其中 $U$ 为 user 特征数，$I$ 为 item 特征数。

当 $U$ 较大且 N 增长时，收益显著。

---

## 8. 面向技术文档落地的架构结论

### 8.1 已实现能力（可直接复用）

1. **压缩输入协议**：Vec/Seq 双路径，支持 ragged 序列；
2. **高性能 CUDA 重建**：scatter/broadcast/fused 写回；
3. **ONNX 图级融合**：Slice 子图替换为 Emblayer 自定义节点；
4. **在线收益分解框架**：字节、meta、H2D bubble、吞吐全链路指标。

### 8.2 架构收益来源（理论）

- 减少输入字节（尤其序列 padding 去除）；
- 降低 metadata 与 CPU 拼接成本（int32、packed meta、模板索引）；
- 减少中间张量与节点调度（fused kernel + graph converter）；
- 利用推荐业务结构先验（user 共享广播）。

### 8.3 下一步演进建议（不改语义）

1. V3 元数据协议：分段 16bit 索引编码；
2. Vec+Seq 更深层 kernel 融合；
3. prefix/offset 解码缓存化与并行前缀预处理；
4. converter 增加自动 range 发现与静态合法性校验。

---

## 9. 参考实现索引（代码定位）

- EmblayerVec CUDA：`examples/emblayer_op/emblayer_vec_cuda.cu`
- EmblayerSeq CUDA：`examples/emblayer_op/emblayer_seq_cuda.cu`
- EmblayerVec C++ 接口：`examples/emblayer_op/emblayer_vec.cpp`
- EmblayerSeq C++ 接口：`examples/emblayer_op/emblayer_seq.cpp`
- 输入构建与模式调度：`examples/benchmark_dinemblayer_infer.py`
- 联合 Converter：`examples/onnx_emblayer_joint_converter.py`
- Seq Converter：`examples/onnx_emblayerseq_converter.py`
- 在线调度指标：`examples/batchscheduler.py`

---

## 附：符号表

- $B$：batch size
- $D$：模型输入总维度
- $nnz$：非零（有效）元素数
- $U$：user 侧非序列特征数
- $I$：item 侧非序列特征数
- $T_{h2d\_only}$：包含准备与拷贝的 H2D 时间
- $T_{h2d\_pure}$：纯拷贝时间
- $T_{bubble}$：H2D 准备气泡时间
- $\rho_{meta}$：metadata 字节占比
