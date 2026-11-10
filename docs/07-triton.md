(ch-triton)=
# 07　TileLang 与 Triton 入门：用一个程序描述一个 tile

<span id="triton-tile"></span>

本章同时讲解 **TileLang 与 Triton 的 tile 编程入门**。以下标签页覆盖 factory、launch、掩码、归约和二维下标；选择任一后端都能顺序完成全部例子。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 让程序员在 tile 层明确写出计算区域、存储 scope 与协作操作：`T.Kernel` 声明执行网格，`T.Parallel` 声明并行元素，`T.alloc_shared` 与 `T.alloc_fragment` 声明状态，`T.copy`、`T.gemm` 与归约描述计算。Python 工厂构造程序；它本身不逐元素执行 GPU 计算。

::::

::::{tab-item} Triton
:sync: triton

Triton 让程序员用张量块描述 kernel，而不是逐个写出 CUDA thread 的行为。这能减少样板代码，但不会取消 GPU 的执行和存储层次。理解一个 kernel 的办法，是同时追踪 Python wrapper、程序实例的下标、逻辑 tile 的类型与 shape，以及它最终产生的内存访问。

::::
:::::

## 7.1 host wrapper 与 device kernel

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

本项目的 `vector_add` 是普通 Python 包装：验证输入元数据、分配输出、按当前 shape/dtype 特化 `_vector_add_kernel` 并调用。factory 用内部 `@T.prim_func` 构造设备函数，再交给 `tilelang.compile`，并缓存编译后的调用器；真正的加法写在 `T.Kernel` 与 `T.Parallel` 中，没有用 `torch.add` 充当计算实现。

::::

::::{tab-item} Triton
:sync: triton

本项目的 `vector_add` 是普通 Python 函数：检查元数据、分配输出、选择 grid，然后 launch。`_vector_add_kernel` 由 `@triton.jit` 修饰，函数体在 GPU 上执行。wrapper 中不能把 `torch.add` 的计算伪装为手写 kernel；device kernel 中也不能随意调用普通 Python 库函数。

::::
:::::

下面直接包含仓库中的真实 device kernel，修改源码后重建本书会同步更新：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/fundamentals.py
:language: python
:pyobject: _vector_add_kernel
:linenos:
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _vector_add_kernel
:linenos:
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

逐行读 TileLang factory：参数先确定 N、dtype、逻辑 BLOCK 和线程配置；内部 `T.Tensor` 描述输入输出 buffer；`T.Kernel` 产生块编号；`T.Parallel` 遍历当前块的逻辑位置，边界条件保护真实读写。`.astype("float32")` 明确计算精度，写回时再转换到输出 dtype。

::::

::::{tab-item} Triton
:sync: triton

逐行读这段代码：`tl.program_id(0)` 是 grid 第 0 维上的程序编号；`tl.arange(0,BLOCK)` 产生一个编译时已知长度的逻辑向量；两者相加得到本 program 负责的全局下标。`X+offsets` 是一个指针张量，不是立即读取的数值；`tl.load` 才执行读取，`tl.store` 执行写入。

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang factory 的 Python 整数、dtype 字符串与配置参数先参与程序构造，因而 shape 或配置改变可能得到不同特化。GPU 中读取的张量内容则属于运行时数据，例如注意力的长度与页表。不要在 factory 中把它们 `.item()` 到 CPU 后当作编译常量；那会破坏变长语义和 CUDA Graph 使用方式。

::::

::::{tab-item} Triton
:sync: triton

`BLOCK: tl.constexpr` 表示编译期参数，可用于定义向量 shape。`N` 则作为运行参数参与边界判断。编译器还可能对某些参数的值或对齐特征做特化；不能把 JIT 缓存简单理解为“同一个 Python 函数永远只编译一次”。dtype、设备、常量与 launch 配置都可能改变缓存键。

::::
:::::

## 7.2 从零写一个完整的 launch

以下代码可以保存为单独的 Python 文件运行。它刻意不包含通用的输入校验，用于展示 host/device 边界；项目公开函数应继续使用带验证的版本。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])
def add_example(n, block=256):
    @T.prim_func
    def main(X: T.Tensor((n,), "float32"),
             Y: T.Tensor((n,), "float32"),
             Z: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=128) as bx:
            for i in T.Parallel(block):
                index = bx * block + i
                if index < n:
                    Z[index] = X[index] + Y[index]
    return main

n = 1003
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)
kernel = add_example(n, 256)
z = kernel(x, y)
torch.testing.assert_close(z, x + y)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_example(X, Y, Z, N, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < N
    x = tl.load(X + indices, mask=valid, other=0)
    y = tl.load(Y + indices, mask=valid, other=0)
    tl.store(Z + indices, x + y, mask=valid)

n = 1003
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)
z = torch.empty_like(x)
block = 256
add_example[(triton.cdiv(n, block),)](x, y, z, n, BLOCK=block, num_warps=4)
torch.testing.assert_close(z, x + y)
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

这里先调用 `add_example(n, 256)` 得到编译后的调用器，再以 `kernel(x, y)` 执行。`out_idx=[2]` 指定第三个 buffer 是返回的输出；这是本节独立示例的便利接口，项目公开包装仍明确分配和校验输出。`T.Kernel` 的网格有四个 CTA，尾部 21 个逻辑位置不读写。

::::

::::{tab-item} Triton
:sync: triton

方括号提供 grid，圆括号提供参数。grid 为四个 program，每个逻辑处理 256 个值，尾部 21 个位置被掩码屏蔽。`num_warps=4` 是编译与执行配置，不是需要定义在 kernel 参数列表里的普通输入。

::::
:::::

不要在 Python 中用 `for` 为每个小 tile 启动一次 kernel 来替代 grid，这会把大量 host launch 开销引入本应并行的工作。应尽量让一个 launch 的 grid 描述所有独立 tile。

## 7.3 掩码的代数意义

尾部读取的 `other` 必须匹配接下来的运算。求和的无效项可以是 0，求最大值的无效项应是 $-\infty$，求最小值则常用 $+\infty$。对乘积归约，单位元又是 1。选择一个“看起来无害”的 0，并不总是正确。

例如一行有效值都是负数，最大值归约时把尾部补零，会让最大值变成不存在的 0。某些 softmax 情况下最终数值可能仍近似正确，但稳定性与中间尺度已经改变；对直接求最大值的函数就是明确错误。使用正确的归约单位元，能让证明覆盖所有输入。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的 `T.if_then_else` 是数值选择；内存安全必须落实到真正的 buffer 访问上。最容易审查的标量写法是在 `if idx < N:` 内完成 load 与 store。使用自动边界 copy 时，也应核对源区域、目标区域、零填充与生成谓词，不能只在最终结果上清零。

::::

::::{tab-item} Triton
:sync: triton

`tl.where(valid, tl.load(ptr), 0)` 不能代替 masked load：未选中的 load 仍可能执行。正确写法是 `tl.load(ptr, mask=valid, other=0)`。同样，输出指针即使计算到了逻辑边界之外，也必须在 store 上使用有效掩码。

::::
:::::

## 7.4 行 softmax：归约与广播

一行 $x_j$ 的稳定 softmax 写为

$$
m=\max_jx_j,\qquad
p_j=\frac{\exp(x_j-m)}{\sum_k\exp(x_k-m)}.
$$

减去最大值不会改变精确算术下的结果，却把最大指数限制为 1，避免有限大正数直接取指数导致溢出。对普通有限 logits，至少有一个有效位置的指数为 1，分母为正。

项目真实 softmax 核心如下。宽行采用分块在线归一化与第二次扫描；先读 `COLS<=BLOCK` 的分支即可理解基础归约。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/fundamentals.py
:language: python
:pyobject: _row_softmax_kernel
:linenos:
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _row_softmax_kernel
:linenos:
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

`T.reduce_max(values, maximum, dim=...)` 把 tile 沿指定维归约到较小 fragment；`T.reduce_sum` 计算指数和，随后用 `T.Parallel` 将归约结果广播回每个逻辑元素。fragment 属于一组协作 thread，不能把一个逻辑标量误认为仅在一个 thread 内存在。具体 shuffle 或共享内存通信由 lowering 决定。

::::

::::{tab-item} Triton
:sync: triton

`tl.max(values,axis=0)` 把一维 tile 归约为一个逻辑标量；随后 `values-maximum` 将这个值广播到各元素。广播是逻辑语义，不代表所有数据都原本位于同一 thread 中。跨 lane 的归约通信由编译器负责，后续章节会观察它如何降低。

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import row_softmax

x = torch.randn((17, 781), device="cuda", dtype=torch.float32) * 4
y = row_softmax(x)
torch.testing.assert_close(y, torch.softmax(x, dim=-1), rtol=2e-5, atol=2e-6)
torch.testing.assert_close(y.sum(dim=-1), torch.ones(17, device="cuda"),
                           rtol=2e-5, atol=2e-6)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060.fundamentals import row_softmax

x = torch.randn((17, 781), device="cuda", dtype=torch.float32) * 4
y = row_softmax(x)
torch.testing.assert_close(y, torch.softmax(x, dim=-1), rtol=2e-5, atol=2e-6)
torch.testing.assert_close(y.sum(dim=-1), torch.ones(17, device="cuda"),
                           rtol=2e-5, atol=2e-6)
```

::::
:::::

781 列通常补到一个 2 的幂逻辑宽度，但只有前 781 列被写回。`row_softmax` 对空轴返回空结果；对于 NaN、Inf 或所有元素均为 $-\infty$ 的行，普通最大值/指数算术可能传播 NaN，本函数不承诺注意力的“全掩码行返回零”语义。后续注意力代码必须单独处理空有效 KV，不能把这两个契约混为一谈。

## 7.5 二维广播只写必要的下标

构造矩阵 tile 的地址时，常用两个一维下标：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
# TileLang device 端：每个逻辑坐标的地址在 buffer 下标中表达。
# 下面片段放在 T.Kernel 内，M/N/STRIDE/BLOCK_M/BLOCK_N 为工厂参数。
for i, j in T.Parallel(BLOCK_M, BLOCK_N):
    row = block_m * BLOCK_M + i
    col = block_n * BLOCK_N + j
    if row < M and col < N:
        Output[row, col] = Input[row, col]
```

::::

::::{tab-item} Triton
:sync: triton

```python
# 下面是 device 端的形状示意，需放在 @triton.jit 函数内。
rows = tl.arange(0, BLOCK_M)       # [BLOCK_M]
cols = tl.arange(0, BLOCK_N)       # [BLOCK_N]
offsets = rows[:, None] * stride + cols[None, :]  # [BLOCK_M, BLOCK_N]
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 在 `T.Parallel(BLOCK_M, BLOCK_N)` 中通过 i/j 给出二维逻辑坐标；`T.Tensor` 或 `T.StridedTensor` 的访问把坐标与物理 stride 结合。检查整个访问区域时，先写每个轴的范围，再检查读写保护；不能把逻辑循环轴直接等同于 CUDA thread 的 x/y。

::::

::::{tab-item} Triton
:sync: triton

`[:,None]` 把 shape 变为 $[B_M,1]$，`[None,:]` 变为 $[1,B_N]$，相加按广播规则产生二维指针 tile。可先在纸上写出 shape，再检查每一项能否广播；这比遇到编译错误后随机修改维度更有效。

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

二维边界由行有效与列有效的交集决定，两个条件都为真才能读写对应元素。结构化 copy 的区域边界与显式 `if` 应保持同样的有效域。页面两种示例采用不同表达，覆盖证明完全相同。

::::

::::{tab-item} Triton
:sync: triton

对掩码也做同样分析：行有效条件 shape 为 $[B_M,1]$，列有效条件为 $[1,B_N]$，二者用 `&` 组合得到二维有效区域。Python 的 `and` 不表达这种逐元素逻辑。

::::
:::::

### 从 CUDA 基础迁移到 tile 层

| 共同的问题 | TileLang 的表达 | Triton 的表达 |
|---|---|---|
| 哪些 CTA 承担独立输出 | `T.Kernel` 的网格与块坐标 | launch grid 与 `tl.program_id` |
| CTA 内的逻辑元素范围 | `T.Parallel` 的迭代空间 | `tl.arange` 与广播后的 tile |
| 状态存在哪里 | global Tensor、shared、fragment scope | 值张量与编译器选择的寄存器/shared 布局 |
| CTA 内协作计算 | `T.copy`、`T.reduce_*`、`T.gemm` | `tl.load`、`tl.sum/max`、`tl.dot` |
| 可流水化的 K 迭代 | `T.Pipelined` | `tl.range(..., num_stages=...)` |

这些是职责对照，不是逐函数机械替换表。同样的逻辑元素集合可能采用不同的 lane 分配、共享布局、归约次序和编译策略。公开包装负责元数据与输出生命周期，数学参考独立于两套 device 实现。

## 7.6 正确的学习循环

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

先固定小输入和一组 TileLang 配置，验证 shape、边界、dtype、存储 scope 与输出后再调优。自动布局推导和配置搜索不会修复错误的下标；参数超出共享内存预算也不会变成合法程序。缓存键与搜索键必须包含改变生成代码的 shape、stride、dtype、目标及调度信息。

::::

::::{tab-item} Triton
:sync: triton

先固定一个小输入与一组配置，验证边界、dtype 和输出 shape，再研究生成代码，最后计时。`triton.autotune` 可以搜索配置，但不能修复错误的索引，也不能使超出共享内存限制的配置合法。调优键还必须包含影响性能和安全的形状信息，避免为一个输入选出的配置被不恰当地复用。

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

调试 TileLang 时，先在 factory 的 Python 层打印形状和配置，再保存构造的 IR 与生成设备源码。把编译错误、目标不支持、资源超限和运行数值失败分别记录。调试输出或同步模式都会影响性能，正式测量前应移除；完整缩小流程见{doc}`appendix/debugging`。

::::

::::{tab-item} Triton
:sync: triton

需要调试时，可使用 `tl.static_assert` 检查编译期条件、`tl.static_print` 输出编译信息，或在小输入下使用 `tl.device_print`。这些工具会改变编译或运行行为，不能带着大量 device 打印去测稳态性能。更多错误定位方法见{doc}`appendix/debugging`。

::::
:::::

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

本机 `results/basics.json` 的两组向量长度是 100003 与 1048576，结果都与 PyTorch 完全一致。第一组专门检查尾部，第二组扩大工作量。普通提交 event 中位时间分别为 23.376 与 23.888 μs；时间接近提示提交和固定开销值得单独测量，并不表示工作量没有增加。

::::
:::::

## 练习

1. 把独立加法例子的 `n` 改成 0、1、255、256、257，决定 host 端在零长度时是否应跳过 launch，并实现这个选择。
2. 写出 `rows[:,None]`、`cols[None,:]`、二维 mask 的 shape，手算右下角 tile 的四个位置哪些有效。
3. 把 softmax 输入整体加上 100，解释稳定公式为什么仍然适用。再测试 NaN 与全 $-\infty$，记录契约之外的行为。
4. 对一个求最小值的 kernel，说明 masked load 的 `other` 应取什么，并给出用 0 补齐时会失败的具体输入。
