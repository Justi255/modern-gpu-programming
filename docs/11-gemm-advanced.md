(gemm-advanced)=
# 11. 调度、驻留与高级 GEMM 的适用边界

一个 CTA 内部的流水线足够快以后，整体性能仍可能受到网格调度、缓存工作集合和输出尾波影响。本章把观察范围从一个块扩大到全部 SM。持久化调度、warp specialization 和 CTA cluster 都试图改变工作如何分配，但它们解决的问题不同，也有不同的架构前提。RTX 5060 Laptop 的实测资源与 B200 相差很大；本书把这些机制拆成可迁移的思想和必须重新验证的实现。

## 11.1 普通网格与持久化网格

普通分块 GEMM 为每个输出块启动一个 program。若共有

$$
T=\left\lceil\frac{M}{B_M}\right\rceil
\left\lceil\frac{N}{B_N}\right\rceil
$$

个输出块，则启动 $T$ 个独立工作。GPU 调度器按资源条件把它们放到 SM 上，最后不足一波的工作称为尾波。若一次只能运行 $R$ 个 CTA，则理想波数为 $\lceil T/R\rceil$。最后一波只有少数 CTA 时，许多 SM 会空闲；这对中等规模矩阵尤其明显。

持久化变体启动一个有界的 program 池。编号为 $p$ 的 program 依次处理 $p,p+P,p+2P,\ldots$，其中 $P$ 是池大小。每处理一个输出块，必须重新初始化累加器、重新计算行列地址，并完整走过 K 循环。持久化减少了 program 调度次数，也使软件可以更直接地控制遍历顺序；它并不自动增加数学并行度。若总块数本来就比池小，仍然无法凭空填满 GPU。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/fundamentals.py
:language: python
:pyobject: _matmul_persistent_kernel
:linenos:
:caption: 固定工作池与跨输出块的静态步进
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _matmul_persistent_kernel
:linenos:
:caption: 固定工作池与跨输出块的静态步进
```

::::
:::::

```{figure} static/kernels-persistent.svg
:alt: 普通网格把每个输出块作为独立任务，持久化网格由固定工作池循环获取块
:width: 93%

持久化改变任务所有权的时间跨度，输出块之间仍然相互独立。
```

## 11.2 池大小不是 SM 数量的同义词

「一个 SM 一个 program」是便于说明的起点，不是普遍最优解。若一个 program 消耗大量寄存器与共享内存，单个 SM 可能只能容纳一个；若它经常等待访存，多放一个 program 可能隐藏延迟。反过来，过大的池会削弱持久化的调度优势。池大小应由资源预算和测量决定，设备探针中的 SM 数只提供硬件规模。

SM 数也不应硬编码为所有同名 GPU 的共同属性。本项目围绕当前 RTX 5060 Laptop 的 26 个 SM 进行实验，但环境 JSON 才是每次结果的证据。不同功耗模式、驱动、机器散热状态与后台负载，都可能改变可观察性能。持久化内核在资源较小的设备上尤其需要警惕单个 program 过度占用资源，造成难以填补的空闲。

## 11.3 调整遍历顺序以利用 L2

按行扫描输出块，会让相邻 program 共享某些 A 块；按列扫描则倾向于共享 B 块。分组扫描把若干 M 块与若干 N 块组织成局部工作集合，在两个方向上折中。一个合理问题是：在下一次需要同一个输入块之前，L2 是否可能已被其他流量覆盖？答案取决于块尺寸、总并发和缓存容量，而不是索引顺序本身看起来是否整齐。

可以估算一个局部分组需要的输入容量，再与探针和 profiler 信息结合。这个估算不能精确预测 cache hit，因为 GPU 的实际调度并不保证严格遵循 program 编号，多个 SM 也会同时读取数据。应把分组大小作为待测参数，保留跨形状的退化记录；只展示最佳方阵会掩盖长条矩阵上增加地址计算的代价。

## 11.4 warp specialization 的概念与成本

warp specialization 让不同 warp 或 warpgroup 长期承担不同角色，例如加载、矩阵计算和输出处理。与所有 warp 轮流执行相同阶段相比，专职角色可以更连续地使用不同硬件单元。但它需要明确的数据交接、缓冲区寿命与同步协议，也可能让部分 warp 在某些阶段闲置。

上游高级 GEMM 在 SM100 上把 TMA、`tcgen05`、TMEM 和专职 warp 连接起来，并进一步使用两个 CTA 协作。本项目的 `persistent` 变体用于学习输出块调度，不能据此宣称实现了同一套 warp specialization 或 cluster 算法。SM120 的 Tensor Core 编程路径与 SM100 不同；看到同属 Blackwell，不能推断 `tcgen05` 与 TMEM 可用。编译器是否支持某个实验特性，也必须用目标设备与生成指令确认。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

每条跨角色边都应回答“谁产生、谁消费、完成信号是什么、何时允许覆盖”。这个方法适用于 TileLang 的 `T.copy`/`T.gemm` 流水、split-KV 的两阶段全局交接和普通 CUDA 队列。平台不同会改变 barrier 指令，但不会取消依赖图。

::::

::::{tab-item} Triton
:sync: triton

理解上游方案仍然有价值：每条跨角色边都应回答「谁产生、谁消费、完成信号是什么、何时允许覆盖」。这个四问方法同样适用于 Triton 编译器生成的流水线、后面的 split-KV 两阶段归约，以及普通 CUDA 中的生产消费队列。架构相关的 barrier 操作不可照抄，依赖图的检查方法则可以保留。

::::
:::::

## 11.5 运行同一数学问题的不同调度

下面的代码使用较不规则的输出形状，避免只验证内部完整块。每个变体返回相同形状和类型；调度改变不应成为修改数值合同的理由。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import matmul

torch.manual_seed(11)
a = torch.randn((769, 256), device="cuda", dtype=torch.float16)
b = torch.randn((256, 513), device="cuda", dtype=torch.float16)
reference = (a.float() @ b.float()).half()
for variant in ("tiled", "pipelined", "persistent"):
    c = matmul(a, b, variant=variant)
    torch.testing.assert_close(c, reference, rtol=2e-2, atol=2e-2)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060.fundamentals import matmul

torch.manual_seed(11)
a = torch.randn((769, 256), device="cuda", dtype=torch.float16)
b = torch.randn((256, 513), device="cuda", dtype=torch.float16)
reference = (a.float() @ b.float()).half()
for variant in ("tiled", "pipelined", "persistent"):
    c = matmul(a, b, variant=variant)
    torch.testing.assert_close(c, reference, rtol=2e-2, atol=2e-2)
```

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m fa5060.experiments gemm --backend tilelang --output results/local/tilelang/gemm.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m fa5060.experiments gemm --backend triton --output results/local/triton/gemm.json
```

::::
:::::

阅读结果时先确认三个变体均通过正确性检查，再比较相同形状的时间。若持久化仅在个别形状改善，应把它解释为一个有条件的调度选择。生产实现经常需要按形状 dispatch，教学变体的价值在于暴露这种条件，而不是制造一个始终获胜的标签。

## 11.6 向注意力过渡

GEMM 的输出块相互独立，而注意力在一个查询块内部还维护在线 softmax 状态；后者不能简单把不同 K 段的输出相加。两者相同的是：都要在复用、并发和状态容量之间权衡。接下来会先推导如何合并指数归一化状态，再讨论 prefill 与 decode 为什么需要不同的工作划分。

## 本机实验观察

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

本章的 TileLang 对应实验由 `--backend tilelang` 生成；读者结果保存在 `results/local/tilelang/`，本次发布证据位于 `results/tilelang/`。当前发布的实际状态和可下载证据见{doc}`15-results`；没有对应文件或成功记录时，保留“待测”，不引用历史 Triton 时间。

::::

::::{tab-item} Triton
:sync: triton

**历史 Triton 实验（2026-11-02，原环境）**

本机 512³ 的 persistent 普通提交中位时间为 FP16 38.528 μs、BF16 38.656 μs，均慢于对应流水线版本 30.400 与 29.888 μs，见 `results/gemm.json`。本实现按 SM 数设置静态持久化 grid；在这组规模上，软件循环与有限并发的代价没有被调度复用收益抵消。这个结果是保留教学变体的理由，而不是删掉不利数据的理由。

::::
:::::

## 练习

1. 假设总输出块数为 60，设备每波可同时执行 26 个 CTA。计算最后一波的活跃比例，并说明这个理想模型遗漏了哪些现实因素。
2. 给出持久化循环中忘记重置累加器会产生的错误模式。为什么第一个块可能正确，后续块却逐渐偏离？
3. 画出一个三角色流水线的生产消费图，为每条边标注数据准备与缓冲复用条件。再指出哪些条件是通用概念，哪些必须依赖具体指令集。

本章对应上游 `chapter_gemm_advanced` 的调度与角色分工主题，平台差异见[来源与迁移表](appendix/sources.md)。
