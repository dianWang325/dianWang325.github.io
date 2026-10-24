---
title: "【Utils】mHC"
pubDate: 2026-10-24
description: 从残差连接出发，用张量形状、数值示例和参考代码理解 mHC 的读取、混合、写回及双随机约束。
category: Utils
draft: false
slugId: mhc
pinTop: 0
---

**mHC 扩展的是层间携带信息的残差通路。** 它让一个 Token 维护多路状态，并学习如何读取、混合和更新这些状态，同时约束残差混合的数值性质。

本文先交代来源与问题，再用 4 路张量推导一次子层更新。公式采用“每一行是一路状态”的约定，省略子层内部的归一化等细节；参考代码用于说明张量语义，不是生产 Kernel。

## 1. 谁提出的，为什么需要它

mHC 全称是 Manifold-Constrained Hyper-Connections，由 **DeepSeek-AI 团队**在同名论文中提出，arXiv 首版提交于 **2025-12-31**。论文通过 3B、9B 和 27B 规模实验研究这一连接机制；不能把它的提出归属于 GLM 5.3 Flash。Flash 是采用该机制的后续模型实例。

它的前身是 2024 年提出的 [Hyper-Connections（HC）](https://arxiv.org/abs/2409.19606)。HC 将单路残差扩展为多路，并学习路间连接；mHC 针对这种连接在大规模训练中暴露的两个问题进一步设计：

1. 多层无约束混合矩阵连续相乘，可能反复放大或衰减信号，导致训练不稳定。
2. 多路状态增加显存读写、激活存储和流水线通信，理论 FLOPs 较小也可能遇到实际吞吐瓶颈。

mHC 用受约束的残差映射改善第一点，并结合融合、重计算和通信调度缓解第二点。它的目标是保留多路连接的表达能力，让训练能够稳定扩展。[原论文](https://arxiv.org/html/2512.24880v2)

## 2. 从单路残差到四路状态

普通残差子层可写成：

$$
x_{l+1}=x_l+F_l(x_l),\qquad x_l\in\mathbb{R}^{C}.
$$

其中 $F_l$ 可以是 Attention 或 FFN。原状态通过旁路保留，子层输出作为增量相加。

引入 $n$ 路后，一个 Token 的状态变成矩阵：

$$
X_l=\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}
\in\mathbb{R}^{4\times C}.
$$

对完整批量，形状从 `[B, T, C]` 变为 `[B, T, 4, C]`。例如 GLM 5.3 Flash 的 $C=4096$，每个 Token 在主干内部携带 $4\times4096$ 个状态元素。四行都是同一个 Token 的表示，不对应四个 Token 或四个专家。

![mHC 张量流：四路状态读取为一路，经一次子层计算后写回四路，同时与受约束的残差混合结果相加](/images/posts/mhc/tensor-flow.svg)

## 3. 三种映射逐步计算

### 3.1 读取：四行合成一行

令读取权重为 $a=[a_1,a_2,a_3,a_4]$：

$$
z=\underbrace{H_{\mathrm{pre}}}_{1\times4}
\underbrace{X_l}_{4\times C}
=\sum_{j=1}^{4}a_jx_j
\in\mathbb{R}^{1\times C}.
$$

子层接收的仍是 $C$ 维输入，并输出同维度的更新：

$$u=F_l(z)\in\mathbb{R}^{1\times C}.$$

因此，残差状态变成四路，不意味着 Attention 或 FFN 被复制执行四次。

### 3.2 混合：旧状态在四路之间交换信息

残差映射 $R=H_{\mathrm{res}}\in\mathbb{R}^{4\times4}$ 作用在流维度：

$$
V=RX_l,\qquad V_i=\sum_{j=1}^{4}R_{ij}x_j.
$$

这个操作不沿序列维交换 Token，也不是把 4096 个通道全部做一次大矩阵投影。每个特征通道都按对应位置的 $4\times4$ 权重混合四路。

### 3.3 写回：子层更新分配到四路

令写回列向量为 $b=H_{\mathrm{post}}^\top\in\mathbb{R}^{4\times1}$：

$$
U=bu=\begin{bmatrix}b_1u\\b_2u\\b_3u\\b_4u\end{bmatrix}
\in\mathbb{R}^{4\times C}.
$$

最终逐元素相加：

$$
\boxed{X_{l+1}=RX_l+bF_l(H_{\mathrm{pre}}X_l)}.
$$

以第 1 路为例，$x'_1=\sum_jR_{1j}x_j+b_1u$：新状态同时包含旧状态的混合以及子层带来的新信息。

在动态连接中，读取、写回、混合权重由当前多路状态经过学习到的映射生成，因此通常随 Token 和子层变化；图中省略了这条系数生成分支。

## 4. “流形约束”落在哪个张量上

关键约束作用于残差矩阵 $R$：

$$
R_{ij}\geq0,\qquad \sum_jR_{ij}=1,\qquad \sum_iR_{ij}=1.
$$

即每个元素非负，每行、每列之和都为 1，称为**双随机矩阵**。下面是一个满足条件的手工示例，并非模型实际学到的权重：

$$
R=\begin{bmatrix}
0.7&0.1&0.1&0.1\\
0.1&0.7&0.1&0.1\\
0.1&0.1&0.7&0.1\\
0.1&0.1&0.1&0.7
\end{bmatrix}.
$$

把单个特征通道的四路数值设为 $[1,2,3,4]^\top$，混合后得到：

$$
R\begin{bmatrix}1\\2\\3\\4\end{bmatrix}
=\begin{bmatrix}1.6\\2.2\\2.8\\3.4\end{bmatrix}.
$$

每个输出是输入的凸组合，没有超出原来的数值范围；四路平均值仍为 2.5。行和条件负责前者，列和条件负责后者。更重要的是，多个双随机矩阵相乘仍然双随机，约束可以沿深度传播。

这个性质只针对 $RX$ 这条混合分支。加入 $bu$ 后，完整输出的均值当然可以变化；双随机也不代表每个向量都被原样保留，路间差异仍可能被混合减弱。不能据此声称整个网络永远不发生梯度消失或信息损失。

## 5. Sinkhorn 如何产生这种矩阵

从非负矩阵开始，交替将每行、每列除以各自的和，可以逐步逼近双随机矩阵，这就是这里使用的 Sinkhorn-Knopp 归一化思路。

GLM 5.3 Flash 的公开配置为：

```json
{
  "mhc": true,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "hc_eps": 1e-6
}
```

有限次迭代、浮点误差以及分母加入的 epsilon 会使行列和与 1 存在偏差。实现中应监测误差，而不能要求所有输入都得到数学上精确的双随机矩阵。

Flash 的 Transformers 实现将读取权重参数化为 `sigmoid(...) + eps`，写回权重为 `2 * sigmoid(...)`；这两个向量不要求元素和为 1。**不能把读取、写回、残差混合三种权重都当成归一化平均。**

## 6. 一段可运行的张量参考代码

下面固定采用 $R_{ij}$ 表示“输入第 $j$ 路对输出第 $i$ 路的贡献”。生产代码的矩阵存储方向可能转置，应以实际 `einsum` 下标为准。

```python
import torch

torch.manual_seed(0)
B, T, N, C = 1, 2, 4, 8
X = torch.randn(B, T, N, C)

# 演示动态系数生成：每个 Token 的 N*C 个状态预测 2N+N*N 个数。
# 省略生产实现的归一化、独立缩放、偏置初始化和精度策略。
projection = torch.nn.Linear(N * C, 2 * N + N * N)
coeff = projection(X.flatten(-2))
a_raw, b_raw, r_raw = torch.split(coeff, [N, N, N * N], dim=-1)
a = torch.sigmoid(a_raw) + 1e-6           # [B, T, N]
b = 2 * torch.sigmoid(b_raw)              # [B, T, N]

R = torch.softmax(r_raw.reshape(B, T, N, N).float(), dim=-1)
R = R + 1e-6
R = R / (R.sum(dim=-2, keepdim=True) + 1e-6)
for _ in range(19):
    R = R / (R.sum(dim=-1, keepdim=True) + 1e-6)
    R = R / (R.sum(dim=-2, keepdim=True) + 1e-6)

z = torch.einsum("btn,btnc->btc", a, X)   # [B, T, C]
F = torch.nn.Sequential(torch.nn.Linear(C, C), torch.nn.SiLU())
u = F(z)                                  # 子层只执行一次
mixed = torch.einsum("btij,btjc->btic", R, X)
written = b.unsqueeze(-1) * u.unsqueeze(-2)
Y = mixed + written                       # [B, T, N, C]

print("X, z, Y:", X.shape, z.shape, Y.shape)
print("row error:", (R.sum(-1) - 1).abs().max().item())
print("col error:", (R.sum(-2) - 1).abs().max().item())
```

这里选 $C=8$ 方便运行和检查，换成 4096 不改变张量关系。示例中的 $F$ 只是一个小网络；真实模型在相应位置使用自己的 Attention 或 Dense/MoE FFN。

## 7. 在 Flash 中如何连接起来

Flash 的 Decoder 分别在 Attention 和 FFN 子层周围使用 mHC，最终通过一个收拢步骤回到单路表示；本文核对的 Transformers 实现中，最终收拢采用流维度上的无权重平均。初始化和最终收拢也属于模型设计，不能只写中间更新公式就认为定义了完整网络。

| 模块 | 它决定什么 |
|---|---|
| Attention / DSA | 当前 Token 如何读取其他位置的信息 |
| MoE Router | 当前子层选择哪些专家计算 |
| mHC | 子层如何读取并更新多路残差状态 |

四路状态不会让主计算模块自动变成四倍 FLOPs，但会增加状态读写和连接系数计算。吞吐取决于实现能否融合这些操作，以及具体并行布局。不能把论文在特定实验中的训练开销直接当作 Flash 的推理开销。

结合模型结构阅读，可参见 [【Model】GLM 5.3 & GLM 5.3 Flash](/blog/glm-5-3-and-flash/)。

## 参考资料

- [DeepSeek-AI：mHC 原论文 v2](https://arxiv.org/html/2512.24880v2)，来源、训练问题和双随机约束；[提交历史](https://arxiv.org/abs/2512.24880)。
- [Hyper-Connections 原论文](https://arxiv.org/abs/2409.19606)，多路连接的前身。
- [Flash 固定版本配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/config.json)。
- [Transformers 固定版本实现](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/glm5_next/modeling_glm5_next.py)，查找 `Glm5NextTextHyperConnection`、`Glm5NextTextDecoderLayer` 和 `Glm5NextTextHyperHead`。
