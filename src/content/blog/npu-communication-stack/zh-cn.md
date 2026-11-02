---
title: "【AI Infra】从 HBM 到 HCCS：理清昇腾 NPU 的内存与通信栈"
pubDate: 2026-11-02
description: 从存储位置、硬件互连、数据搬运和软件接口四个维度，理解 PCIe、HCCS、DMA、RDMA、URMA、HCCL 与 MemFabric，并串起 KV Cache Offload 的三条数据路径。
category: AI Infra
draft: false
slugId: npu-communication-stack
pinTop: 0
---

阅读 vLLM-Ascend 的 KV Cache Offload 实现时，会接连遇到 HBM、DRAM、HCCS、PCIe、SDMA、RDMA、URMA、HCCL、MemFabric 和 Memcache。这些名字经常出现在同一条数据路径上，但描述的是不同问题。

理解它们，可以先问四句话：**数据存在哪里？沿什么链路传输？由什么机制搬运？哪个软件组织这次操作？** 把这四件事分开，就能解释为什么 PCIe 不一定需要主机内存中转，也能理解为什么 AscendStoreConnector 和 MemFabric 不属于同一层。

本文按 2026-11-02 查阅的官方文档整理；硬件路径取决于具体产品和拓扑，KV Offload 部分对应当前文档所描述的 P/D 分离方案。

## 1. 先建立整体架构

![昇腾 NPU 通信栈：vLLM、Connector、HCCL、MemFabric、底层传输后端、硬件互连和存储位置的层次关系](/images/posts/npu-communication-stack/architecture.svg)

图中箭头表示调用或依赖关系，不表示每次通信都必须经过所有模块。HCCL 和 MemFabric 是不同用途的软件路径：前者组织多卡集合通信，后者提供内存寻址与复制能力。Memcache 是缓存管理服务，可以利用下层内存传输能力。

| 维度 | 主要概念 | 回答的问题 |
|---|---|---|
| 存储位置 | Host DRAM、NPU HBM | 数据在哪块内存中？ |
| 硬件互连 | PCIe、HCCS、UB、以太网 | 数据通过哪条通路？ |
| 搬运机制与访问接口 | DMA、SDMA、RDMA、URMA | 如何执行本地或远程读写？ |
| 通信与内存软件 | HCCL、MemFabric | 如何组织通信或统一访问内存？ |
| 缓存与框架集成 | Memcache、KV Connector、Offload Manager | 保存哪些 KV，什么时候加载、卸载？ |

## 2. 数据在哪里：Host DRAM 与 NPU HBM

Host DRAM 是主机的主内存，通常就是平常所说的服务器内存。CPU 内部的 L1/L2/L3 Cache 是另一类容量更小、速度更快的缓存，不能和主内存混为一谈。

HBM 可以先理解为 NPU 的显存，用来保存模型权重、KV Cache、中间激活和工作区。它也不是专门为 KV 准备的缓存区。HBM 本身属于 DRAM 技术，文档把 Host DRAM 和 HBM 并列，是为了区分主机内存和设备显存的位置。

```text
CPU：片上缓存 ↔ Host DRAM
NPU：片上存储 ↔ HBM
```

KV 是模型生成的 Key、Value 数据；KV Cache 是把历史 KV 保存起来供后续 Attention 复用的缓存。它可以放在 HBM，也可以卸载到 Host DRAM。**“Cache”描述用途，“HBM/DRAM”描述存储介质与位置。**

## 3. 数据走哪条路：PCIe、HCCS 与交换机

### 3.1 PCIe：可以连接主机与设备，也可以支持设备间 P2P

PCIe 是通用高速 I/O 互连。CPU 与 NPU 之间可以通过 PCIe 交换数据；在设备、驱动和拓扑支持时，设备之间也可以通过 PCIe Peer-to-Peer（P2P）直接访问对方内存。

```text
设备 P2P：
NPU A HBM → PCIe 路径 → NPU B HBM

主机内存中转：
NPU A HBM → Host DRAM → NPU B HBM
```

这是两条不同的数据路径。**走 PCIe 不等于必须先写入 Host DRAM。** P2P 事务可能经过 PCIe Switch，或者在支持的拓扑中经过 Root Complex/Host Bridge；经过 CPU 所在芯片的 I/O 路由，也不等于 CPU 执行了逐字节复制。

P2P 是否可用需要检查硬件拓扑、地址映射、ACS 等限制，不能仅凭“两个 NPU 在同一台机器里”判断。[Linux 内核的 PCI P2P DMA 文档](https://docs.kernel.org/driver-api/pci/p2pdma.html)专门说明了这些路由边界。

### 3.2 HCCS：昇腾高速互连，不限于单机

HCCS 是昇腾体系中的高速互连，可承载 NPU 间的数据交换。在支持的超节点系统中，它也能连接不同服务器中的 NPU。昇腾的[超节点 P2P 带宽测试文档](https://www.hiascend.com/doc_center/source/zh/mindcluster/2610/toolbox/toolboxug/toolboxug_0215.html)明确包含节点间 HCCS 链路测试。

“服务器节点”和“超节点”需要区分：一个超节点可以由多台服务器组成，内部通过高性能互连协同工作。因此，“同一超节点内通信”不等于“同一机箱内通信”。

### 3.3 是否经过交换机，取决于拓扑

同机 NPU 可以通过直接链路互连，也可以通过交换芯片转发；跨服务器通信同样可能经过专用互连交换设备或网络交换机。

```text
直接链路：设备 A ──────────── 设备 B
交换路径：设备 A ── 交换芯片 ── 设备 B
```

PCIe Switch、HCCS/UB 互连交换设备、以太网交换机分别服务于对应的互连体系。看到“交换机”时，应继续问：它交换的是什么协议、位于哪条路径上？

UB（UnifiedBus，灵衢）是另一项需要单独识别的互连体系。不要把 UB、HCCS 和 PCIe 当作可以随意替换的同义词；实际组合应以产品架构与软件支持矩阵为准。

## 4. 谁负责搬：DMA、SDMA、RDMA、URMA

这四个名词并不严格处于同一抽象层。把它们区分为“硬件搬运”“远程访问能力”“软件接口”，比统一称为通信协议更准确。

### 4.1 DMA：让专用硬件搬数据

DMA（Direct Memory Access，直接内存访问）让设备或专用引擎执行数据传输，减少 CPU 对数据复制的参与。

```text
软件准备源地址、目标地址、长度
              ↓
硬件执行搬运
              ↓
查询完成状态或接收完成通知
```

CPU 仍可能负责内存注册、提交任务、处理完成事件。DMA 并不意味着软件完全不参与，也不决定传输走 PCIe 还是其他互连。

### 4.2 SDMA：具体的硬件搬运引擎

在这里的昇腾语境中，SDMA 是设备侧的 DMA 搬运引擎。MemFabric 可以通过它访问映射好的内存。在支持统一映射的超节点环境里，可访问的地址范围也可能包含其他节点的内存。

因此，不能因为名字里没有“Remote”，就推断 SDMA 只能在本机搬运。访问范围由地址映射、互连和平台能力共同决定。

### 4.3 RDMA：直接访问远端注册内存

RDMA（Remote Direct Memory Access）提供远程内存访问能力，传输硬件能够按建立好的连接和权限读写对端注册内存，减少接收端 CPU 对数据搬运的介入。RoCE 是在以太网上实现 RDMA 的一种方式；RDMA 和 RoCE 也不是同一个层次的词。

“远端”是相对于发起访问的一侧而言的另一端。例如 Decode 节点从另一台服务器上的 Prefill NPU 拉取 KV 时，Prefill HBM 就是 Decode 视角下的远端内存。它可以位于同一机柜、同一超节点，不涉及公网。

`device_rdma` 中的 `device` 用来区分设备侧传输路径和 Host RDMA 路径；是否需要主机中转，应继续检查实现与硬件支持。

### 4.4 URMA：面向 UB 的远程内存访问接口

URMA（Unified Remote Memory Access）是 UB 通信基础库，向上提供远端读写、收发和原子操作等接口，并屏蔽部分底层驱动差异。它通过下层驱动和硬件实现通信，本身不是一条物理链路。[openEuler 的定义](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/urma/URMA%20User%20Guide.ch.html)明确区分了这两个层次。

在 MemFabric 配置中，`sdma`、`device_rdma`、`device_urma` 可以作为并列的后端选项。这种并列来自软件封装，并不意味着 SDMA 引擎、RDMA 技术和 URMA 软件库在概念上完全同级。

## 5. 软件怎样组织通信：HCCL、MemFabric 与 Memcache

### 5.1 HCCL：组织多卡通信操作

HCCL 是昇腾的集合通信库，例如 AllReduce、AllGather、Broadcast 等。应用提交通信操作后，库根据拓扑和配置选择算法及传输路径。

```text
AllReduce 等通信语义
        ↓
HCCL 组织算法与调度
        ↓
设备互连与传输资源
```

HCCL 是软件库，HCCS 是互连，两者名字相似但职责不同。HCCL 也不只对应 HCCS 一种路径。[CANN 通信配置文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hixlug/hixldev_001.html)中分别列出了机内 PCIe 和超节点 HCCS 的配置项。

### 5.2 MemFabric：统一编址与内存复制

MemFabric 面向跨设备、跨节点的 DRAM/HBM 访问，提供 GVA（Global Virtual Address，全局虚拟地址）和接近 `memcpy` 的复制接口。它再把复制请求交给 SDMA、Device RDMA、Device URMA 等后端完成。[项目架构说明](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/README.md)

GVA 是逻辑上的统一编址，便于软件描述“要访问哪块内存”。它不意味着所有物理内存已经融合成一块，也不保证任意路径具有同样的带宽、延迟或缓存一致性。

### 5.3 Memcache：管理 KV 缓存存储

本文的 Memcache 指昇腾 KV 池方案使用的缓存服务，不是泛指 Memcached。它管理缓存数据、存储空间及相应服务，底层可以使用 MemFabric 的内存访问能力。

`AscendStoreConnector` 则处于 vLLM 框架接入层：把模型执行时的 KV 保存、加载请求对接到 Ascend Store 后端。在本文的 Prefill Layerwise 方案中，该后端是 Memcache。

| 组件 | 主要职责 |
|---|---|
| AscendStoreConnector | 把 vLLM 的 KV 生命周期对接到存储后端 |
| Memcache | 管理 KV 缓存及其存储服务 |
| MemFabric | 统一寻址并执行内存复制 |

## 6. 用 KV Offload 串起完整数据路径

vLLM-Ascend 的 Layerwise/Sparse Offload 组合中，Prefill 和 Decode 使用两份不同的 Host KV 池。以下路径来自[设计文档](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.html#2-system-overview)。

### 6.1 Prefill 按层加载、保存

```text
Prefill NPU HBM 的可复用层 Buffer
            ↕ AscendStoreConnector 对接
Prefill Host KV 池（Memcache 管理）
```

Prefill 按层计算，让多个逻辑层复用少量设备 Buffer。需要前缀 KV 时从主机加载，计算后把更新结果保存回主机，并尝试让传输与计算重叠。

### 6.2 Prefill 向 Decode 交付 KV

```text
Prefill NPU HBM
      │ SfaRemoteD2HConnector + MemFabric
      ├── Main KV ───────→ Decode Host KV 池
      └── Indexer 数据 ──→ Decode NPU HBM
```

这条联合路径由 Decode 拉取 Prefill NPU 中的数据。它没有先从 Prefill Memcache 池读取，再转存到 Decode。

Prefill 的共享 Buffer 必须等 Memcache 保存和 Decode 读取都完成后才能覆盖，否则后续层可能破坏尚未传完的数据。

### 6.3 Decode 追加新 KV，并回迁 Top-K miss

```text
Decode NPU 生成当前 Token 的 KV
              ↓ D2H
Decode Host 完整 Main KV 池
              ↓ 仅对 Top-K miss 执行 H2D
Decode NPU 当前层 Hot Buffer → Sparse Attention
```

这时 Host 与 Device 可以位于同一台 Decode 服务器上：D2H/H2D 描述的是设备和主机之间的方向，不能据此认定是跨机网络通信。

Decode 仍然按层执行，当前层 Indexer 确定 Top-K 后，才知道这一层需要回迁哪些 KV。每层 Hot Buffer 保留已加载的热点条目，用于后续 Step 复用。

## 7. 阅读通信代码时的检查顺序

遇到一个“复制 KV”的接口，可以按以下顺序定位：

1. **找源和目标**：Host DRAM 还是 NPU HBM？本地还是远端？
2. **看粒度**：连续大块、KV Block，还是离散 Token 行？
3. **看组织者**：HCCL 集合通信、Connector 交付，还是 Offload Manager 稀疏回迁？
4. **看执行后端**：MemFabric 选择了 SDMA、Device RDMA，还是 Device URMA？
5. **核对物理路径**：PCIe、HCCS、UB 或 RoCE，是否允许设备 P2P？
6. **确认完成条件**：提交成功是否等于传输完成？源 Buffer 何时可以复用？

例如，看到 `sparse_copy` 只能判断这是某种稀疏复制请求；还需要检查注册地址、传输后端、同步方式和硬件拓扑，才能知道它实际怎样完成。接口名、数据方向和物理链路需要分别辨认。

## 参考资料

- [vLLM-Ascend：Layerwise and Sparse KV Cache Offloading Design](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.html)
- [MemFabric Hybrid：项目架构与后端说明](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/README.md)
- [Linux Kernel：PCI Peer-to-Peer DMA Support](https://docs.kernel.org/driver-api/pci/p2pdma.html)
- [openEuler：URMA User Guide](https://docs.openeuler.org/zh/docs/24.03_LTS_SP3/unifiedbus/unifiedbus/urma/URMA%20User%20Guide.ch.html)
- [昇腾：超节点 P2P 带宽测试](https://www.hiascend.com/doc_center/source/zh/mindcluster/2610/toolbox/toolboxug/toolboxug_0215.html)
- [CANN：HIXL 建链与通信路径配置](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hixlug/hixldev_001.html)
