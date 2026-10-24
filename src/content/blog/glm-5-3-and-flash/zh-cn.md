---
title: "【Model】GLM 5.3 & GLM 5.3 Flash"
pubDate: 2026-10-24
description: 从公开配置拆解 GLM 5.3 与 Flash 的骨干、混合注意力、MoE 和 MTP 索引复用，并厘清它们与 mHC 的关系。
category: Model
draft: false
slugId: glm-5-3-and-flash
pinTop: 0
---

GLM 5.3 与 GLM 5.3 Flash 的差异贯穿整个骨干。官方模型卡说明，**GLM 5.3 沿用 GLM 5.2 的底座，主要增益来自后训练；Flash 则重新训练底座，使用混合注意力、mHC 和原生多模态设计。**

本文以 2026-10-24 核对的公开权重配置为准。模型名称分别对应 `zai-org/GLM-5.3` 和 `zai-org/GLM-5.3-Flash`；固定配置版本见文末。文章关注结构和执行语义，不把 API 定价或排行榜差异直接当作架构收益。

## 1. 先看两条主干

![GLM 5.3 与 Flash 主干对比：前者是 78 层文本 Decoder，后者是带视觉编码器和 mHC 的 45 层混合注意力 Decoder](/images/posts/glm-5-3-and-flash/architecture.svg)

图中的 Attention 和 FFN 是同一 Decoder 层内的两个子模块。为便于理解，图按模块职责分行展示，并不表示整个模型先执行完全部 Attention 再执行全部 FFN。

| 结构 | GLM 5.3 | GLM 5.3 Flash |
|---|---|---|
| 实现类型 | `glm_moe_dsa` | `glm5_next`，文本子配置为 `glm5_next_text` |
| 输入 | 文本 | 文本、图像、视频 |
| Decoder 层数 | 78 | 45 |
| 隐藏维度 | 6144 | 4096 |
| Attention | DSA 稀疏注意力 | 34 层线性注意力 + 11 层 DSA |
| FFN 布局 | 前 3 层 Dense，后 75 层 MoE | 前 3 层 Dense，后 42 层 MoE |
| 每个 MoE 层的路由专家 | 256，单 Token 选 8 个 | 288，单 Token 选 8 个 |
| 每个 MoE 层的共享专家 | 1 | 1 |
| 残差通路 | 常规残差连接 | 4 路 mHC |
| 最大位置配置 | 1,048,576 | 1,048,576 |

这些是配置口径。最大位置配置不等于所有服务都开放同样的上下文长度；模型总参数、激活参数、权重文件大小和运行显存也需要分开统计。Flash 模型卡披露的是 **320B 总参数、18B 激活参数**，不宜把激活参数当成部署时只需加载的权重规模。

## 2. DSA：先选历史位置，再算注意力

普通因果注意力让当前 Query 访问此前的全部有效 KV。DSA（DeepSeek Sparse Attention）增加一个轻量 Indexer：先给历史位置打分并筛选候选，再在选中的位置上执行主注意力。

```text
当前 Query
    ↓
Indexer：为历史位置打分 → Top-K 位置列表
    ↓
读取被选中位置的 KV → 计算注意力权重 → 加权得到输出
```

这里的“索引”是历史位置的编号，不是 MoE 专家编号。Indexer 解决“去哪里看”，主注意力解决“这些信息各占多大权重”。两者不能合并理解为一次 Top-K 就得到了注意力输出。

GLM 5.3 的配置继承了层间索引复用设计：78 个主干层的 `indexer_types` 中，21 个为 `full`，57 个为 `shared`。前者计算索引，后者复用已有选择。**`full` 在这里描述 Indexer 的执行方式，不表示把 DSA 改成全量稠密注意力。**

这属于沿深度方向的优化：不同层仍计算自己的 Query、注意力权重和输出，只共享历史位置的筛选结果。[IndexCache 论文](https://arxiv.org/abs/2603.12201)讨论了这种跨层复用；GLM 5.2 模型卡将其称为 IndexShare。具体分组以配置数组为准，不能简单地假定从第 1 层开始所有层都严格四层一组。

## 3. Flash：把多数层换成线性注意力

Flash 的主干 Attention 排列可以直接从 `layer_types` 读出：

```text
[线性 → 线性 → 线性 → DSA] × 11 → 线性
                  合计 45 层
```

对应实现中，线性路径使用 Kimi Delta Attention（KDA）相关计算。概念上，它通过递推状态汇总历史信息；DSA 路径则保留对选中历史 KV 的直接访问。前者减少持续访问历史 KV 的工作，后者提供对具体历史位置的访问能力。

需要分清两个统计维度：**34/11 描述 Attention 类型，3/42 描述 Dense/MoE FFN 类型。** 例如主干前 3 层同时具有线性 Attention 和 Dense FFN；后续的线性 Attention 层也可以搭配 MoE FFN。

由结构可以推断，Flash 的长上下文执行压力有机会下降，但整个模型并非完全固定大小状态：DSA 层仍需要相应历史缓存。真实显存和吞吐取决于长度、批量、量化格式、状态布局及 Kernel 实现，不能仅按线性层占比计算加速倍数。

## 4. “单层有 288 个专家”是什么意思

MoE 把 FFN 扩展为一组具有独立参数的专家。一个 Token 到达当前 MoE 层后，Router 根据隐藏向量选择专家：

```text
Token 的隐藏向量 ─→ Router ─→ 288 个专家中选 8 个 ─→ 加权合并 ─┐
         └──────────────→ 1 个共享专家 ────────────────────┴→ 输出
```

这里的“288”是**这一层的专家池大小**。下一 MoE 层有自己的参数；当前 Token 在下一层可以走另一组专家。每个 Token 不会执行全部 288 个路由专家，但每层还有始终参与的共享专家，以及路由、Attention 等计算。

专家分工由训练形成，通常没有预先指定“数学专家”“编程专家”等人工职责。Flash 的单层专家池比 GLM 5.3 大，并不意味着整个模型更大：主干层数、隐藏维度、专家内部维度和激活数量共同决定容量与计算。

## 5. 原生多模态与 mHC 各自改变什么

Flash 的视觉配置包含 24 层编码器，隐藏维度为 1024，投影输出维度为 4096，以便接入语言主干。图像和视频经过视觉处理后，与文本表示一起进入后续计算；模型输出仍是文本。原生多模态也涉及预训练配方，不能仅理解为给文本模型临时拼一个编码器。

mHC 则改变 Attention、FFN 周围的残差连接。Flash 用 4 路状态携带一个 Token 的内部表示，子层入口先组合这些状态，子层计算后再写回。它与选择哪几个专家、读取哪些历史位置分别属于不同机制。

张量形状、三种映射和双随机约束详见配套文章 [【Utils】mHC](/blog/mhc/)。

## 6. MTP 迭代间的索引复用

两者配置均启用了 `index_share_for_mtp_iteration`。这一项应结合执行后端理解：配置表达模型意图，真正是否启用、怎样管理缓冲区由框架决定。

以本文核对的 vLLM `DeepSeekMultiTokenPredictor.set_skip_topk()` 为例，一轮 MTP 草稿生成的第 0 步计算自己的 Top-K 索引，后续步骤复用该选择：

```text
草稿第 0 步：筛选历史位置 S → Attention → 候选 1
草稿第 1 步：复用位置 S     → Attention → 候选 2
草稿第 2 步：复用位置 S     → Attention → 候选 3
                                   ↓
                             主模型验证候选
```

**复用的是位置列表，每一步仍重新计算 Query、注意力权重和输出。** 新生成位置的可见性与尾部处理需看具体后端，不能凭开关推断完整掩码行为。

| 复用机制 | 复用方向 | 省去的主要工作 |
|---|---|---|
| 层间 IndexShare | 同一预测位置，不同层 | 部分层的重复索引筛选 |
| MTP 迭代间复用 | 同一轮草稿，不同预测步骤 | 后续草稿步骤的重复索引筛选 |

例如生成 4 步草稿，索引筛选可以从 4 次变成 1 次，但主注意力和其他子层仍执行。相邻位置可能依赖相似历史，这是复用的直觉；它不保证等价于逐步重新筛选。草稿质量、接受率以及主模型验证开销共同影响最终收益。

## 7. 从架构判断部署工作量

对推理适配而言，GLM 5.3 重点涉及 DSA/MLA、层间索引共享和 MoE；Flash 还需要 KDA 状态更新、线性/DSA 混合调度、mHC 混合映射及视觉路径。模型更轻不代表算子覆盖更简单。

一个实用的检查顺序是：先核对配置与权重版本，再确认各 Attention 路径和状态缓存均受支持，最后单独验证 MTP。主干前向跑通，并不能证明投机解码或所有多模态输入已经可用。

## 参考资料与版本

- [GLM 5.3 官方模型卡](https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c19c8738e2c57c71798c601d477a7bf844/README.md)与[配置](https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json)。
- [GLM 5.3 Flash 官方模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/README.md)与[配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json)。
- [GLM 5.2 模型卡：IndexShare](https://huggingface.co/zai-org/GLM-5.2)、[IndexCache 论文](https://arxiv.org/abs/2603.12201)。
- [Transformers：GLM5 Next 实现](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py)，对应 KDA、mHC 和视觉路径。
- [vLLM：MTP 索引复用](https://github.com/vllm-project/vllm/blob/7a64fb8fc9239cb65e0df9313fa759594dfb206a/vllm/model_executor/models/deepseek_mtp.py)，查找 `set_skip_topk`。
