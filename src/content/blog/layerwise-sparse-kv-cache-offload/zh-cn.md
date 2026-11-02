---
title: Layerwise 与 Sparse KV Cache Offload：内存池与执行流程
pubDate: 2026-11-02
category: AI 系统
draft: false
slugId: layerwise-sparse-kv-cache-offload
pinTop: 0
---

## Prefill 层级 Buffer 复用

Prefill 不再给每个模型层永久分配一份 NPU KV Buffer，而是让多个逻辑层复用少量物理 Buffer。

每层执行过程大致是：

1. 必要时从 Memcache 加载该层前缀 KV。
2. 等待目标共享 Buffer 可以安全覆盖。
3. 执行 Attention 并更新 KV。
4. 将完整层 KV 保存到主机。
5. 在计算当前层时预取后续层。

共享 Buffer 只有在以下操作都完成后才能复用：

- KV 已保存到 Memcache；
- Decode 已完成对该 Buffer 的远程读取。

## Prefill 与 Decode 的 KV 数据流

```text
Prefill NPU：少量可复用层 Buffer
    ├── 保存完整层 KV → Prefill Memcache 主机池
    └── 远程传输 → Decode 主机完整 KV 池

Decode 主机完整 KV 池
    └── 只将 Top-K miss 搬回 → Decode NPU Hot Buffer
```

## 两个主机内存池

这里实际上存在两个不同的主机内存池：

- Prefill 侧：Memcache 管理的 KV 池。
- Decode 侧：MemFabric 注册的连续主机内存池。

## 单步 Decode 流程

每个 Decode Step 包含：

1. 生成新 Token 的 K/V，但不写入完整的 NPU Paged KV Cache。
2. 将新 K/V 写入主机池中的逻辑位置。
3. Indexer 选择逻辑 Top-K Token。
4. 查询每层 LRU 驻留表：
   - 哪些是 hit；
   - 哪些是 miss；
   - miss 应分配到哪个 Hot Buffer 槽；
   - 必要时淘汰哪个旧槽。
5. 只把 miss 行从主机搬到 NPU。
6. 将逻辑 Top-K 位置转换成 Hot Buffer 物理位置。
7. 稀疏注意力读取 Hot Buffer 完成计算。
8. 保留驻留状态，供下一步复用。
