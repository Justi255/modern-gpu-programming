(ch-async)=
# 06　异步执行与流水：每块缓冲区都有生命周期

异步的价值是重叠独立工作，代价是程序必须准确表达依赖。把 load 提前并不自动安全；在没有确认消费者结束前复用缓冲区，通常比越界更难调试。本章分别看 host、kernel 内流水和硬件传输的异步，最后给出一套可以检查每个 stage 的方法。

## 6.1 host 异步不等于 kernel 内重叠

CPU 调用一个 CUDA kernel 通常只是把工作提交到 stream，随即继续执行。相同 stream 中的操作按依赖顺序执行，不同 stream 的工作有机会重叠，但是否实际重叠还取决于资源和同步。使用 event 或 `torch.cuda.synchronize()` 可以在 host 上等待完成。

kernel 内部则是另一件事。一个 CTA 处理 K 方向上的多个 tile，当前 tile 的矩阵乘加依赖其数据到达，下一 tile 的搬运却可能提前开始。让“下一块 load”与“当前块计算”重叠，才是本章讨论的软件流水。

同一 stream 保序也不能解决 kernel 内消费者提前读取的问题。反之，CTA 内 barrier 也不会让 CPU 自动等待整个 kernel 完成。理解异步时，应明确写出两个动作分别由谁执行、依赖发生在哪个范围。

## 6.2 单缓冲先建立正确性

把每个 K tile 的步骤写成：读取 A/B → 确认到达 → MMA → 确认不再读取 → 复用缓冲区。单缓冲中它们依次发生，不会重叠。设一次 load 时长 $L$，计算时长 $C$，共 $n$ 个 tile，忽略尾部输出时总时间约为

$$
T_{\rm serial}=n(L+C).
$$

这里 $L,C$ 是用于建立模型的变量，不是测得常数。在实际 kernel 中，地址计算、MMA 依赖链和存储层次还会进一步影响时间。

双缓冲给当前 tile 与下一 tile 不同的物理槽。理想情况下可以近似达到

$$
T_{\rm pipeline}\approx L+(n-1)\max(L,C)+C.
$$

当 $n$ 很大时，吞吐由较慢的阶段限制；当 $n$ 很小时，填充和排空开销仍然重要。若 $L=C$，理想长期收益接近两倍；不能因为增加两个 stage 就声称必然获得两倍性能。

```{figure} static/foundations-pipeline.svg
:name: fig-pipeline
:alt: 单缓冲串行执行 load compute，双缓冲让下一 tile 的 load 与当前 tile 的 compute 重叠

流水的收益来自阶段重叠，安全性来自缓冲区的生产与消费依赖。
```

## 6.3 环形 stage 的两个方向依赖

设缓冲槽数为 $S$，第 $k$ 个 tile 使用槽 $k\bmod S$。每个槽至少有两种状态转换：

```text
空闲 → 正在写入 → 数据已就绪 → 正在消费 → 空闲
```

生产者在写入前必须确认旧消费者结束，这是避免覆盖的依赖；消费者在读取前必须确认新生产者完成，这是避免读半成品的依赖。只有第二种依赖的流水看起来经常能运行，直到某次线程调度变化才发生错误。

循环多次复用同一个同步对象时，还要区分“这一次完成”和“上一轮已经完成”。低层 barrier 协议通常使用 phase 或 parity 实现世代区分。教学伪代码可以写 `wait_ready(slot, generation)`，真实实现必须遵守相应硬件与 API 的内存顺序。这个伪函数不是本项目公共接口。

## 6.4 共享内存预算限制 stage 数

若每个 stage 保存一个 FP16 的 A tile 和 B tile，忽略 padding 与辅助区时，共享内存需求为

$$
S_{\rm bytes}=S\cdot2B_K(B_M+B_N).
$$

取 $B_M=64,B_N=128,B_K=32$，每 stage 为 $2\times32\times192=12288$ 字节，即 12 KiB。四个 stage 需要 48 KiB，八个 stage 需要 96 KiB。96 KiB 看起来低于本设备每 CTA 的 99 KiB 上限，却只剩很少空间留给 padding、额外缓冲与编译器开销，而且几乎肯定限制同 SM 的多 CTA 驻留。$B_K$ 改为 64 后，每 stage 变为 24 KiB，五个 stage 已经超过每 CTA 上限。

这个预算不是编译器最终分配值，因为后端可能采用不同 staging 策略。它足以排除明显不可能的候选配置，再用编译元数据核对实际字节数。`num_stages` 增大以后变慢，可能是资源、无效预取或填排成本，不能只归因于“GPU 太弱”。

## 6.5 常规 load、异步 copy 与 TMA

从全局内存到计算 fragment 可以走不同路径。常规 load 把数据带到 thread 寄存器；一些异步 copy 指令支持全局到共享内存传输；TMA 用描述符表达多维张量的地址、stride、box 等信息，由硬件执行搬运。它们对应的对齐、布局、参与范围与同步协议不相同。

SM120 支持 TMA，但本书不把所有 `tl.load` 或带 `num_stages` 的循环称作 TMA。是否走到描述符/TMA 路径，需要从 Triton API、目标支持和生成 PTX 验证。一个 TMA 传输可以由很少的 thread 发起，而消费数据的可能是多个 warp，因此消费者还需要正确的完成通知。

第十三章的 `attention_tma.py` 给出实际 TensorDescriptor 版本：它在本机生成 `cp.async.bulk.tensor` 加载和 warp 级 `mma.sync`，并与默认 pointer 版本共享在线 softmax 语义。先学习本章的完成依赖，再运行该章的对照实验，可以把「硬件支持」「编译器确实采用」与「当前形状更快」分成三个可核验的问题。

在使用 `mbarrier` 追踪传输的低层路径里，“参与者到达”和“预期字节全部到达”是不同条件。把到达计数设对却没有登记传输字节，或者登记字节与实际传输不一致，都可能造成提前消费或永久等待。本书默认让编译器管理普通流水，不要求读者在第一版 GEMM 中手写 barrier。

数据中心 SM100 的 warp specialization、`tcgen05.commit`、TMEM 读取和 Cluster Launch Control 有各自的支持范围。本机的 TMA 能力不能推出这些机制也可直接使用。研究低层优化时，应逐项检查 SM120 与当前 Triton 后端的支持，先保存一个无这些机制的正确基线。

## 6.6 在 Triton 中观察流水

Triton 的 launch `num_stages` 和 `tl.range(..., num_stages=...)` 表达相关的编译配置，但它们不等于运行时创建了相同数量的 Python 对象。编译器根据循环、load、dot 关系选择可流水化的操作；数据依赖太紧或模式不支持时，源码参数与实际重叠可能并不一致。

本项目提供三条 GEMM 路线作为对比，实际实现见后续章节：

```python
import torch
from fa5060.fundamentals import matmul

a = torch.randn((512, 1024), device="cuda", dtype=torch.float16)
b = torch.randn((1024, 512), device="cuda", dtype=torch.float16)
reference = a.float() @ b.float()
for variant in ("tiled", "pipelined", "persistent"):
    c = matmul(a, b, variant=variant)
    torch.testing.assert_close(c.float(), reference, rtol=2e-2, atol=2e-2)
```

先确认同一 shape 的语义相同，再用 `python -m fa5060.experiments gemm --output results/gemm.json` 测量。只有看到编译资源和稳态时间，才能判断多 stage 是否有效。persistent 表示软件复用 CTA 处理多个输出 tile，不自动代表用了硬件工作窃取或 CLC。

## 练习

1. 若假设 $L=3,C=5,n=8$，计算串行模型与理想双缓冲模型时间，解释为什么收益小于两倍。
2. 对 $B_M=B_N=64,B_K=64$ 的 FP16 双路输入，求每 stage 字节数和三 stage 的总量。再考虑一个 4 KiB 辅助区。
3. 为环形槽画出第 0、1、2、3 个 tile 的读写顺序，假设槽数为 2。标出防止读早与写早的两类依赖。
4. 在编译结果中分别寻找异步 copy、TMA 与 MMA 证据；若只找到了 MMA，为什么仍不能断言 load 与计算完全串行？
