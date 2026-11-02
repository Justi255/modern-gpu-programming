(gemm-basics)=
# 9. 从矩阵公式到分块 GEMM

矩阵乘法的困难并不在于写出三个循环，而在于让每次搬来的元素参与足够多的运算，同时为 GPU 提供足够多的独立工作。本章把这两个问题分开：先确定一个 Triton program 负责的输出区域，再沿归约轴累计结果。这里的 GEMM 是教学用计算积木，也为注意力中的两次矩阵乘法建立直觉；它不承诺替代 cuBLAS 的全部功能。

## 9.1 固定数学约定

本项目使用通常的行列约定：$A\in\mathbb R^{M\times K}$、$B\in\mathbb R^{K\times N}$，输出 $C\in\mathbb R^{M\times N}$，且

$$
C_{mn}=\sum_{k=0}^{K-1}A_{mk}B_{kn}.
$$

上游教材为了线性层权重布局，使用了形状为 $N\times K$ 的右操作数。本项目的公共函数是 `matmul(a, b)`，因此不要把上游的 `B[n,k]` 地址公式原样复制过来。矩阵形状相同并不意味着存储约定相同；一次不必要的转置足以让后面的带宽分析失去意义。

乘加计为两次浮点运算，总工作量为 $2MNK$。若执行时间为 $t$ 秒，吞吐率为 $2MNK/(10^{12}t)$ TFLOP/s。这个公式只计数学上有用的运算，不把边界 padding 的额外计算计入分子，也不把输入创建和 JIT 编译算成 GPU 运算。

## 9.2 一个 program 拥有一个输出块

设输出块为 $B_M\times B_N$，归约步长为 $B_K$。二维逻辑网格的坐标 $(p_m,p_n)$ 决定当前块左上角 $(p_mB_M,p_nB_N)$。源码把网格压成一维：若 N 方向有 $T_N$ 个块，则编号 p 对应 $p_m=\lfloor p/T_N\rfloor,p_n=p\bmod T_N$。块内部的行、列索引分别来自 `tl.arange`；它们表示逻辑张量中的元素集合，并不是让 Python 逐个创建线程。

每轮从 $A$ 读取 $B_M\times B_K$，从 $B$ 读取 $B_K\times B_N$，通过 `tl.dot` 更新同一个 fp32 累加器。循环结束后，把结果转换为输出类型并写回。对 SM120，本书追踪的是 Triton 编译器生成的寄存器、共享内存和 warp 级矩阵指令路径。累加器在程序的寄存器状态中，不需要也不能仿照 SM100 教程显式分配 TMEM。

```{figure} static/kernels-gemm-tiles.svg
:alt: 输出块在 M N 平面分配，每个块沿 K 轴连续累计
:width: 92%

空间分块决定谁写输出；归约分块决定何时加载下一段输入。
```

地址计算值得单独推一次。一般二维 stride 下，$A_{mk}$ 的元素偏移是 $m s_{A0}+k s_{A1}$，$B_{kn}$ 的偏移是 $k s_{B0}+n s_{B1}$。本项目 GEMM 要求最后一维连续，即 $s_{A1}=s_{B1}=1$，外层行 stride 可以不同于逻辑宽度；源码将这两个外层 stride 作为参数传入。不要为了图省事把元素偏移乘成字节偏移：Triton 张量指针加整数时按元素类型前进。

以下是直接参与执行的分块内核。所有源码节选都从工程文件读取，构建教材时不会另外维护一份可能过时的实现：

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _matmul_tiled_kernel
:linenos:
:caption: 单个输出块与串行 K 块循环
```

## 9.3 边界是算法的一部分

真实形状很少恰好整除块大小。读取 $A$ 时的条件为 $m<M\land k<K$，读取 $B$ 时为 $k<K\land n<N$；无效位置填零，便可继续使用相同的乘加表达式。写回只检查 $m<M\land n<N$。输入零填充保证归约不变，输出 mask 则保证相邻 program 不会越界写入。

这也是为什么只测试方阵不够。取 $M=127,N=193,K=65$，三个维度的尾部会同时出现；再测试一个非连续输入，就能检查形状与 stride 是否被混为一谈。失败时先观察错误是否只发生在最后一行、最后一列或最后一个 K 块。这样的空间分布通常比单个最大误差更能指出问题。

## 9.4 分块如何增加数据复用

一次 K 步的有用工作量是 $2B_MB_NB_K$，fp16 输入搬运的理想字节数是 $2(B_MB_K+B_KB_N)$，因此仅考虑这一步输入时，算术强度为

$$
I_{\rm tile}=\frac{B_MB_N}{B_M+B_N}\quad\text{FLOP/byte}.
$$

当 $B_M=B_N=B$ 时得到 $B/2$。增大输出块能提高输入复用，但累加器元素数量随 $B_MB_N$ 增长，寄存器压力也随之上升。假设一个 $64\times64$ 的 fp32 累加器，其逻辑容量就是 16 KiB；实际每线程寄存器还包括地址、输入片段和临时值，不能只用这个数字预测 occupancy。大块不是免费的带宽优化。

同样的复用还存在于 program 之间。不同输出块可能读取同一个 A 或 B 区域，L2 缓存可以减少外存流量，但这取决于调度顺序和同时活跃的工作集合。不要把「理论上可复用」直接写成「一定只读一次」。后续实验需要同时检查实际时间与缓存行为。

## 9.5 运行与核对

在仓库根目录完成安装后，下面的例子验证一个带三个尾块的 fp16 GEMM。参考结果使用 fp32 输入相乘后转回输出类型，从而把存储精度和归约精度分开。示例容差服务于该输入分布，不能自动成为所有模型的误差合同。

```python
import torch
from fa5060.fundamentals import matmul

torch.manual_seed(9)
a = torch.randn((127, 65), device="cuda", dtype=torch.float16)
b = torch.randn((65, 193), device="cuda", dtype=torch.float16)
c = matmul(a, b, variant="tiled")
expected = (a.float() @ b.float()).to(c.dtype)
torch.testing.assert_close(c, expected, rtol=2e-2, atol=2e-2)
print(tuple(c.shape))
```

运行本项目的 GEMM 实验，统一收集正确性与时间数据：

```bash
python -m fa5060.experiments gemm --output results/gemm.json
```

第一次调用会触发编译。不要把终端中第一次等待的总时间当作内核延迟。下一章会进一步区分 JIT、预热、稳定执行以及计时边界。

## 9.6 阅读源码时的四个不变量

首先，每个有效输出元素恰好被一个 program 写入，不能有写竞争。其次，所有 K 块更新同一个逻辑累加器，循环中不能意外清零。第三，所有越界输入必须在乘加前变成零，而不是在运算后才丢弃输出。最后，输入为低精度并不表示累计也使用低精度：源码和生成代码都应核对累计类型。这四条不变量比某个固定块尺寸更值得记住。

## 练习

1. 对 $M=129,N=257,K=63$ 和 $B_M=B_N=64,B_K=32$，计算网格大小、K 循环轮数，以及边界 program 中有用输出的比例。
2. 用行切片（如 `storage[::2, :]`）构造外层非连续、最后一维连续的二维输入。先写出两组 stride，再验证 `matmul` 的结果；不要用 `.contiguous()` 隐藏地址问题。
3. 推导当 $B_M$ 固定时，增大 $B_N$ 对算术强度与累加器容量的影响。提出一个可通过计时和寄存器报告证伪的预测。

本章与上游 `chapter_gemm_basics` 的概念对应关系、布局约定差异及固定版本链接见[来源与迁移表](appendix/sources.md)。
