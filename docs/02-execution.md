(ch-execution)=
# 02　执行模型：从一个元素到一张 grid

把 GPU 想成“同时执行很多 Python 循环”会掩盖最重要的问题：谁拥有数据、谁能通信，以及谁必须一起到达一条指令。GPU 的并行层次为这些问题划出了边界。本章从一维加法开始，再把同样的层次用于矩阵 tile 与归约。

## 2.1 thread、warp、CTA、grid

CUDA thread 有自己的寄存器状态和地址计算。32 个 thread 组成一个 warp，warp 内的 thread 位置叫 lane，编号为 0 到 31。硬件以 warp 为主要指令发射单位；一条向量化的指令由活跃 lane 共同参与，每个 lane 处理自己的操作数。

CTA，也就是 CUDA thread block，是共享内存和块级同步的主要合作范围。一个 CTA 的 thread 在同一个 SM 上执行；不同 CTA 不应依赖特定调度先后。grid 是一次 kernel launch 的全部 CTA。GPU 把可运行 CTA 分发给 SM，直到 grid 的工作完成。

```{figure} static/foundations-execution.svg
:name: fig-execution
:alt: grid 中的程序实例分配到多个 SM，每个 CTA 包含四个 warp，warp 含三十二个 lane

软件的合作范围与硬件的执行位置。图中每 CTA 四个 warp 只是一个配置例子。
```

本机有 26 个 SM，但这不意味着应该恰好启动 26 个 CTA。CTA 可能因访存等待、资源占用和工作不均而需要更多可调度工作。与此同时，极小的工作量也可能根本不足以填满 26 个 SM。并行度是输入形状和分块共同产生的结果。

Triton 中，一个 `program` 通常对应一个 CTA。`num_warps=4` 意味着后端为这个程序实例使用四个 warp，即 128 个 thread；它不意味着一个 program 只能处理 128 个元素。一个 1024 元素的逻辑向量可以分布在这些 thread 的多个寄存器槽中。

## 2.2 用下标证明覆盖完整且没有重写

设数组长 $N$，每个程序实例处理 $B$ 个元素。第 $p$ 个程序实例的逻辑下标为

$$
i(p,r)=pB+r,\qquad 0\le r<B,
$$

grid 大小为 $\lceil N/B\rceil$，访问掩码为 $i<N$。这同时解决两个问题：每个有效元素恰好归属于一个 $p=\lfloor i/B\rfloor$，尾部程序又不会越界访问。

例如 $N=1000,B=256$，grid 有四个 program。前三个分别覆盖 0–255、256–511、512–767；最后一个只允许 768–999，共 232 个有效位置。为尾部屏蔽的 24 个位置仍属于编译时逻辑 tile，但不能执行实际越界 load/store。

```python
import torch
from fa5060.fundamentals import vector_add

x = torch.arange(1000, device="cuda", dtype=torch.float32)
y = torch.full_like(x, 0.25)
z = vector_add(x, y)
torch.testing.assert_close(z, x + y)
```

这里的证明与块大小无关，只依赖覆盖与掩码。改变块大小以后，数值行为应不变，但每 CTA 的寄存器、CTA 数量和调度开销可能改变。因此正确性模型与性能模型可以围绕同一套下标分别建立。

## 2.3 SIMT、分支与收敛

warp 中的 lane 可以有不同谓词。例如 `i<N` 对尾部 warp 的某些 lane 为真、另一些为假。掩码让这些 lane 不参与特定操作，而不是启动一个更短的物理 warp。用 `tl.where` 选择数值，也不保证未选择的一侧不会被计算；保护内存安全应使用 `tl.load`/`tl.store` 的 `mask`。

如果同一 warp 的 lane 走入不同控制流分支，硬件可能分批执行活跃集合。两个分支都很重时，就会损失本来可以共同执行的指令吞吐。应优先把整个 program 都相同的条件变为程序级分支，例如一个完整 tile 是否在因果边界之前；再对边界 tile 使用逐元素掩码。

现代 GPU 的独立线程调度也不意味着可以省略通信同步。“同属一个 warp”不是任意共享内存读写都天然有序的理由。低层 warp 通信要遵守对应指令的参与掩码和同步契约；在常规 Triton 中，应让编译器从归约、`tl.dot` 与数据依赖生成所需通信，而不是自行假定指令间的隐式顺序。

## 2.4 资源决定能驻留多少工作

一个 SM 能同时驻留的 CTA 数量受 thread 数、warp 数、寄存器、共享内存和架构上限共同约束。设每个 CTA 用 $T$ 个 thread、每 thread 用 $r$ 个寄存器、每 CTA 用 $s$ 字节共享内存，简化上界为

$$
n_{\rm resident}\le
\min\left(
\left\lfloor\frac{R_{\rm SM}}{Tr}\right\rfloor,
\left\lfloor\frac{S_{\rm SM}}{s}\right\rfloor,
\left\lfloor\frac{T_{\rm SM}}{T}\right\rfloor,
n_{\rm CTA,max}
\right).
$$

实际分配有粒度和保留开销，不能只靠这个式子预测精确值。它最有用的地方是解释趋势：增大 tile 可能减少重复访存，却增加累加器寄存器；增加流水级数可能隐藏延迟，却占用更多共享内存。资源的阶梯变化会让性能出现拐点。

occupancy 是活跃 warp 相对硬件上限的比例，不是计算单元利用率，更不是速度的直接指标。少量 warp 只要有足够独立指令，也可能很好地使用 Tensor Core；很多 warp 如果都在等待同一依赖，仍然不能提高吞吐。调优时应把它与访存、指令和依赖一起看。

## 2.5 同步范围必须覆盖真实依赖

寄存器默认属于各个 thread，跨 lane 的求和需要 shuffle 等通信；CTA 共享内存让不同 warp 交换数据，但生产者写完与消费者读之前需要适当同步。不同 CTA 之间通常通过独立 kernel launch、原子操作或专门的协作机制通信。本书基础 kernel 不使用跨 CTA 自旋等待。

这种限制带来一个实用设计方法：先让每个 CTA 独立完成一个输出 tile。需要跨 CTA 汇总时，明确写出“第一阶段临时结果 → 第二阶段归约”的全局边界。后面的 split-K 和 split-KV 正是这个思想；不能把一个 CTA 内的 barrier 扩大解释为整个 grid 的 barrier。


## 本机实验观察

本机 `results/basics.json` 中，长度 100003 的向量加法与 PyTorch 逐元素一致，普通提交 event 中位时间为 23.376 μs。这个不整除 block 的输入把尾部 mask 变成真实检查；该时间还可能包含主机提交空隙，不能用来反推 warp 指令延迟。

## 练习

1. 对 $N=4097,B=512$ 写出 grid 大小、最后一个 program 的有效元素数，证明不存在两个 program 写同一位置。
2. 一个程序实例用四个 warp 处理 2048 个元素，平均每个 thread 对应多少逻辑元素？为什么这个平均数不足以确定编译器的真实寄存器布局？
3. 一个 kernel 从每 CTA 32 KiB 共享内存增加到 64 KiB 后变慢。列出两种可检验的解释，并说明要从编译结果或分析器中读取什么证据。
4. 判断“CTA 0 写完标志后，其余 CTA 自旋等待，所以等价于 grid barrier”为何可能死锁。
