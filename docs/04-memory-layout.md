(ch-memory-layout)=
# 04　布局：同一个张量，三种地址问题

矩阵在数学上只是一组带下标的数，在 GPU 上却有三种不同的“在哪里”：它在全局内存的哪个字节、它在共享内存的哪个 bank，以及它由哪个 lane 的哪个寄存器持有。把这三种问题混为一谈，容易产生“转置之后就应该连续”或“tile 形状相同所以不需要通信”的误解。本章先建立地址函数，再分别看这些存储层次。

## 4.1 shape 不等于存储顺序

一个 rank-$d$ 张量可用基地址 $p$、元素字节数 $s$、各维 stride $u_k$ 描述：

$$
\operatorname{addr}(i_0,\ldots,i_{d-1})
=p+s\sum_{k=0}^{d-1}i_ku_k.
$$

PyTorch 的 stride 以元素计，字节偏移还要乘 $s$。连续行主序 $A[M,N]$ 的 stride 为 $(N,1)$；转置视图 $A^T[N,M]$ 的 stride 为 $(1,N)$。转置视图修改的是下标解释，通常不会立即复制数据。

```python
import torch

a = torch.arange(12).reshape(3, 4)
at = a.T
print(a.shape, a.stride())       # (3, 4), (4, 1)
print(at.shape, at.stride())     # (4, 3), (1, 4)
print(a.data_ptr() == at.data_ptr())
print(at.contiguous().stride()) # (3, 1)，这次真的重排
```

编写 kernel 接口时必须决定接受哪些布局。如果允许任意非负 stride，地址公式就必须使用传入 stride；如果只支持连续张量，应在 wrapper 中检查并给出明确错误。默默把非连续张量按连续布局读取，会得到形状正确、内容错误的结果，尤其危险。

## 4.2 合并访问看的是同一条指令

warp 内的 lane 同时执行一次内存指令时，硬件尝试把地址合并为较少的事务。以 32 字节 sector 作简化模型：32 个 lane 各读取一个相邻 FP32 元素，总共覆盖 128 字节；地址恰好对齐时覆盖四个 sector。若起点偏移四字节，则可能覆盖五个 sector。缓存、请求粒度和后端指令形式还会影响实际观测，但“同一条指令覆盖多少地址范围”是正确的第一问。

如果 lane $\ell$ 读取 $A[\ell,j]$，行长为 1024，则相邻 lane 地址间距 4096 字节。即使总共只需要 128 字节，地址也散在很远的范围，无法像连续访问那样有效合并。若改为读取 $A[i,\ell]$，相邻地址仅差四字节，就容易得到连续事务。

```{figure} static/foundations-layout.svg
:name: fig-memory-layout
:alt: 连续 lane 对应连续全局地址，转置通过共享内存重排，随后连续写出

转置需要解决“读连续”和“写连续”之间的冲突。中间布局是算法的一部分。
```

判断访问是否合并时，不要只看源码中的数组是否声明为连续。要写出 lane 对下标的映射，并代入地址公式。一个连续存储的矩阵完全可能被跨行访问；一个转置视图也可能恰好配合当前 lane 映射形成连续访问。

## 4.3 转置是最小的布局实验

对 $A[M,N]$ 产生连续输出 $B[N,M]$，语义为 $B[j,i]=A[i,j]$。简单实现让 lane 沿 A 的列读取，读取很好合并；同样的 lane 沿 B 的行写出，却变成大跨度。改变 lane 对下标的映射可以改善写出，却又破坏读取。

一种常见方案是先以连续方式读入 tile，经共享内存或寄存器交换改变所有权，再以另一种连续方式写出。在传统 CUDA 中，程序员显式写共享内存和 barrier；在 Triton 中，编译器可能为从输入布局到输出布局的转换插入通信。逻辑上的 `tl.trans` 是元素坐标变换，是否真的生成共享内存交换，需要看上下文与编译结果。

```python
import torch
from fa5060.fundamentals import transpose

x = torch.arange(37 * 53, device="cuda", dtype=torch.float32).reshape(37, 53)
y = transpose(x)
assert y.shape == (53, 37)
torch.testing.assert_close(y, x.T)
assert y.is_contiguous()
```

37 和 53 故意不是 tile 的整数倍。这个例子同时检查矩阵转置语义、边界掩码和输出物理布局。只对方阵测试容易漏掉把 M/N 或 stride 写反的问题。

## 4.4 共享内存的 bank 模型

以常见的 32 个 bank、每个 bank 按 4 字节字组织的模型为例，标量 FP32 地址对应的 bank 可近似写为

$$
\operatorname{bank}(a)=\left\lfloor\frac{a}{4}\right\rfloor\bmod32.
$$

共享数组 `tile[32][32]` 行主序存放时，lane $\ell$ 读取 `tile[ell][j]`，字偏移为 $32\ell+j$，所有 lane 都落在 bank $j$，且读取不同地址，形成冲突。把每行物理长度改成 33，bank 变为 $(\ell+j)\bmod32$，lane 分散到不同 bank。32×33 的物理数组仍然承载 32×32 的逻辑数据，多出的列只是 padding。

广播同一个地址与读取同 bank 的不同地址要区别对待。前者可能由硬件广播服务，不能看到“bank 一样”就一律断言 32 路冲突。FP16、向量化 load、矩阵 load 与不同硬件路径也不能机械地套用单个 FP32 字模型；应结合实际指令的访问分组分析。

另一种方案是 swizzle。举一个教学用的 32×32 FP32 置换：物理列 $c'=c\mathbin{\oplus}r$，其中 $\oplus$ 是按位异或。每一行中 $c\mapsto c'$ 是双射；固定逻辑列 $c$ 时，行 $r$ 对应的 bank 也遍历 0 到 31。这个例子说明 XOR 能同时保留元素与分散 bank，但它不等于 Tensor Core 或 TMA 任意模式下的合法 swizzle。真实硬件布局还受到元素位宽、向量组和对齐约束。

## 4.5 从逻辑 tile 到所有权

即使两个中间张量的 shape 都为 $16\times16$，也可能分别由不同 lane 持有。将第一个交给第二个消费者时，编译器可能需要 shuffle、共享内存暂存与 barrier。一个完整布局应能回答

$$
(i,j)\longmapsto(\mathrm{warp},\mathrm{lane},\mathrm{register\ slot}),
$$

以及是否存在复制。归约通常把多个 lane 的值合并；广播可能把同一个标量复制到很多 thread；矩阵乘法则需要符合 MMA fragment 的特殊映射。这些都是逻辑张量算子背后的真实工作。

本书先使用 Triton 的自动布局，在{doc}`08-triton-layouts`学习如何从编译结果验证所有权转换。不要为了“显式控制”提前加入重排；先确定数据的生产与消费关系，再判断转换代价是否值得。

## 练习

1. 对 stride 为 $(1,7)$、shape 为 $(7,5)$ 的 FP16 张量，计算元素 $(3,2)$ 相对基地址的字节偏移。它可能是哪种视图？
2. 分析 32 个 lane 读取 FP32 数组 `x[2*lane]` 时覆盖多少个 32 字节 sector，假设基址对齐。与读取 `x[lane]` 比较。
3. 在 32 bank 模型下证明行长 33 的列访问没有 bank 冲突，再解释为什么这并不能证明任意 FP16 矩阵 load 都无冲突。
4. 修改上面的转置输入为 1×53、37×1 和 0 行张量。先读 wrapper 的约束，再预测每种情况应返回还是明确拒绝。
