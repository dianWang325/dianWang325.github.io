---
title: 投机推理
pubDate: 2026-11-03
description: 从 vLLM-Ascend Model Runner V2 的 Worker 调用链，看开启与关闭投机解码时的执行差异。
category: AI 系统
draft: false
slugId: speculative-decoding
pinTop: 0
---

开启投机和不开启投机，都走同一个 **`NPUWorker → NPUModelRunner（V2）`**。投机逻辑位于 ModelRunner 内部，通过 `speculator` 生成草稿，再由主模型验证。

本文基于 2026 年 11 月 3 日查阅的 vLLM-Ascend main 与上游 vLLM Model Runner V2 实现。以下展示普通文本生成场景中，单个执行 rank 内的逻辑调用链；省略图执行包装、通信和缓存管理细节。

## 不开启投机

```text
Executor
  └─ NPUWorker
       └─ NPUModelRunner（V2）
            │
            ├─ execute_model(...)
            │    └─ 主模型 forward → 保存 hidden states 等本轮状态
            │
            └─ sample_tokens(...)
                 └─ sample(...)
                      ├─ 主模型 compute_logits(...)
                      └─ 普通 Sampler → 输出 token
```

`execute_model()` 执行主模型前向，保存本轮计算状态。随后 `sample_tokens()` 使用这些状态计算 logits，并通过普通 Sampler 选择下一个 token。在常规自回归 decode 中，每个请求每轮通常推进一个 token。

## 开启投机

```text
Executor
  └─ NPUWorker                         ← worker 层没有新增草稿 worker
       └─ NPUModelRunner（V2）
            │
            ├─ execute_model(...)
            │    └─ 主模型 forward
            │         └─ 处理包含上一轮草稿的输入，保存验证用 hidden states
            │
            └─ sample_tokens(...)
                 │
                 ├─ sample(...)
                 │    ├─ 主模型 compute_logits(...)
                 │    └─ RejectionSampler → 接受/纠正 token
                 │
                 ├─ postprocess_sampled(...)
                 │    └─ 按接受数、拒绝数更新请求状态
                 │
                 └─ speculator.propose(...)
                      └─ 生成下一轮草稿，写入请求状态
```

上图展示本轮有草稿待验证的路径。主模型一次前向计算草稿对应位置的分布，RejectionSampler 决定接受哪些草稿、在哪里输出纠正 token。随后更新请求状态，再由 speculator 生成下一轮候选。

**从单轮执行看，顺序是“验证上一轮草稿 → 更新状态 → 生成下一轮草稿”。** 新草稿写入 `req_states.draft_tokens`，供后续轮次使用。MTP、EAGLE 等方法通过草稿模型生成候选，n-gram 等方法则不需要神经网络草稿模型。

开启投机后，本轮如果没有草稿，仍然走普通采样。采样分支可以概括为：

```python
if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
    sampler_output = self.sampler(...)
else:
    sampler_output = self.rejection_sampler(...)
```

## 差异发生在哪里

| 位置 | 不开启投机 | 开启投机 |
| --- | --- | --- |
| Worker 入口 | NPUWorker | 同一个 NPUWorker 类 |
| Runner 初始化 | `speculator = None` | 创建与投机方法对应的 speculator |
| 主模型输入 | 常规 prefill / decode 输入 | 验证轮次中包含草稿 token |
| 本轮采样 | 普通 Sampler | 有草稿时使用 RejectionSampler |
| 请求状态更新后 | 完成输出处理 | 继续生成下一轮草稿 |

两种模式都会更新请求状态；普通模式图中为突出主路径省略了这一步。以上结构也不意味着整个服务只有一个 Worker：TP、PP 等并行方式仍然涉及多个 rank。在 PP 场景中，当前实现主要由最后一个 PP rank 持有 speculator，完成采样与草稿生成，其他阶段配合接收结果并更新状态。

## 源码入口

- [Ascend NPUWorker：execute_model 与 sample_tokens 转发](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/worker.py)
- [Ascend Model Runner V2：初始化与 NPU 执行适配](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/model_runner.py)
- [Ascend init_speculator：按投机方法选择实现](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/spec_decode/__init__.py)
- [上游 Model Runner V2：采样、请求状态更新和下一轮草稿生成](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py)

源码链接指向持续变化的 main 分支；本文描述的是上述查阅日期的执行结构。
