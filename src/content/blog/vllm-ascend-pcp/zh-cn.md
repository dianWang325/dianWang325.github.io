---
title: 从源码看懂 vLLM-Ascend 的 PCP：长上下文 Prefill 如何并行
pubDate: 2026-10-15
description: 基于 vLLM-Ascend main 源码，拆解 PCP 的 DualChunkSwap 分片、rank-local batch、KV 同步与 hidden state 恢复。
category: AI 系统
draft: false
slugId: vllm-ascend-pcp
pinTop: 0
---

当一个请求的 prompt 很长时，单卡需要串行处理大量 token，首 token 延迟（TTFT）会明显上升。Prefill Context Parallel（PCP）的做法是：**沿序列维把一次 Prefill 的 token 分给多个 rank 并行计算，再通过通信恢复成全局一致的结果。**

本文基于 2026-10-15 的 `vllm-ascend` `main`，对应提交 [`908273b`](https://github.com/vllm-project/vllm-ascend/commit/908273b4c16b6c5fc758160f602c7e651c34b542)。交互动画中的对象名和阶段都映射到这一版代码。

> PCP 当前走 Model Runner V2（MRV2）路径，启动前需设置 `VLLM_USE_V2_MODEL_RUNNER=1`。它增加的是 Context Parallel rank，不是把现有 TP rank 再切一遍。

## 先用动画走一遍

下面以 `PCP=4`、一条 16-token 的 Prefill 请求为例。点击“下一步”，可以依次看到全局 batch、DualChunkSwap、rank-local 输入、并行前向、KV 同步以及 hidden states 恢复。

<iframe class="pcp-interactive-frame" src="/interactive/pcp-walkthrough.html" title="vLLM-Ascend PCP 交互式执行流程" loading="lazy"></iframe>

## 1. Global batch 仍然是事实来源

MRV2 的 Model Runner 先构造完整的 `AscendInputBatch`。PCP 并不会永久改写 scheduler 看到的全局 batch，而是由 `AscendPCPManager.partition_batch()` 为当前 rank 派生一个本步使用的局部视图。

这个区别很重要：请求状态、采样与最终输出仍然按全局语义推进；被切开的只是模型本次 forward 所消费的 token 和相关元数据。

## 2. DualChunkSwap 为什么是“首尾配对”

设 PCP world size 为 `P`。Prefill 序列先被切成 `2P` 个连续 chunk，rank `r` 取得：

```text
chunk r  +  chunk (2P - 1 - r)
```

当 `P=4` 时，分配关系是：

| Rank | 获得的 chunk | 16-token 示例 |
|---|---|---|
| 0 | C0 + C7 | 1, 2, 15, 16 |
| 1 | C1 + C6 | 3, 4, 13, 14 |
| 2 | C2 + C5 | 5, 6, 11, 12 |
| 3 | C3 + C4 | 7, 8, 9, 10 |

因果注意力中，越靠后的 query 能看到的历史越长，计算量通常越大。如果简单地让每个 rank 只拿一个连续大段，后半段 rank 会更重。DualChunkSwap 把“较轻的前段”和“较重的后段”配在一起，用静态映射改善负载均衡。

这个分配逻辑位于上游 vLLM 的 [`PCPManager._iter_rank_chunks()`](https://github.com/vllm-project/vllm/blob/429a3da6654facfa792995bbbc75ee64cf83be56/vllm/v1/worker/gpu/pcp_manager.py)，Ascend 的 PCP manager 复用了这套基础布局，并补充 NPU 侧元数据处理。

## 3. `partition_batch()` 不只切 `input_ids`

真正进入模型前，Ascend 的 [`partition_batch()`](https://github.com/vllm-project/vllm-ascend/blob/908273b4c16b6c5fc758160f602c7e651c34b542/vllm_ascend/worker/v2/pcp_manager.py) 会让多个结构保持一致：

- `input_ids` 与 `positions` 变成 rank-local；
- request row、query 长度与 token 数随之更新；
- block table、slot mapping 等 KV cache 元数据同步变换；
- padding、采样索引以及后续恢复所需的映射一并生成。

因此，PCP 不能理解成在模型入口简单做一次 tensor slice。它更像是把“这一步的 batch 执行视图”整体投影到某个 rank。

## 4. 每个 rank 只计算自己的 Q/K/V

完成分区后，各个 rank 并行执行模型 forward，只为本地 token 计算 hidden states 和 Q/K/V。到这里计算已经被拆开，但注意力仍要求各 rank 获得正确的上下文，所以接下来必须同步 KV。

以当前 SFA 路径为例，[`_fused_preprocess_with_pcp_kv()`](https://github.com/vllm-project/vllm-ascend/blob/908273b4c16b6c5fc758160f602c7e651c34b542/vllm_ascend/attention/context_parallel/sfa_cp.py) 会打包本地 K/V，在 PCP 组内执行 AllGather，再依据 global slot 写入 cache。其他 attention backend 有各自的实现细节，但目标相同：让局部计算产生的 KV 回到正确的全局位置。

在纯 PCP（`DCP=1`）场景里，每个 PCP rank 最终维护完整 KV 副本；如果叠加 DCP，KV 的所有权和通信组则由 DCP/PCP 拓扑共同决定，不能再套用“每张卡一份完整 KV”的结论。

## 5. AllGather 之后为什么还要 restore

模型 forward 产生的 hidden states 仍按各 rank 的局部 token 顺序排列。直接 AllGather 得到的是 rank-major 顺序：

```text
1, 2, 15, 16, 3, 4, 13, 14, 5, 6, 11, 12, 7, 8, 9, 10
```

而 logits、sampling 和请求后处理需要原始全局顺序：

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
```

所以基础 `PCPManager` 在构造 batch layout 时同时生成 `_hidden_restore_idx`。[`restore_hidden_states()`](https://github.com/vllm-project/vllm/blob/429a3da6654facfa792995bbbc75ee64cf83be56/vllm/v1/worker/gpu/pcp_manager.py) 先 AllGather，再用该索引恢复全局 token 顺序。Ascend manager 还负责对齐 NPU 执行所需的 padding 与输出形状。

## 6. Prefill 和 Decode 并不采用同一种切法

DualChunkSwap 描述的是 Prefill。纯 PCP 且 `DCP=1` 时，decode 请求可以按 round-robin 分散到不同 PCP rank；这时不是把单个 decode token 再做序列切分，而是做请求级 sharding。

一旦启用 DCP，decode query 需要在对应 DCP ranks 间共同参与计算，不能把 PCP-only 的 round-robin 行为直接外推到组合拓扑。

## PCP、DCP、TP 的边界

| 并行方式 | 主要切分维度 | 最直接的收益 | 典型通信 |
|---|---|---|---|
| PCP | Prefill 的 token/序列维 | 降低长 prompt 的 TTFT | PCP 组内 KV、hidden states 集合通信 |
| DCP | Decode attention 的 KV/序列上下文 | 扩展长上下文 decode、分摊 KV 工作 | DCP 组内 attention 协作 |
| TP | 模型权重与 hidden/head 维 | 降低单卡权重和算子计算压力 | 每层 TP collective |

三者解决的问题不同，也可以组合。当前 Ascend 文档给出的组合约束里，MLA/SFA 通常要求 `DCP=PCP` 或 `DCP=TP×PCP`；GQA 目前要求 `PCP=DCP`。实际部署前应按所用模型与 attention backend 再核对当前版本的 [Context Parallel 指南](https://github.com/vllm-project/vllm-ascend/blob/908273b4c16b6c5fc758160f602c7e651c34b542/docs/source/user_guide/feature_guide/context_parallel.md)。

## 启动示例

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve <model-path> \
  --tensor-parallel-size 1 \
  --prefill-context-parallel-size 4
```

启用后，总 world size 会随 PCP size 增长。当前 PCP 仍属于实验能力；Pipeline Parallel、encoder-decoder、多模态和 LoRA 等组合存在限制，生产部署前需要用目标模型、实际序列长度和并行拓扑做性能与正确性验证。

## 一句话总结

PCP 的核心不是“把 prompt 平均切四份”这么简单，而是：**用 DualChunkSwap 平衡因果注意力负载，把完整 batch 映射成 rank-local 执行视图，同步 KV，再通过 restore index 把并行结果无损还原到全局 token 顺序。**
