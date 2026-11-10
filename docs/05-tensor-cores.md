(ch-tensor-cores)=
# 05　Tensor Core：把矩阵交给正确的指令

Tensor Core 的价值来自高吞吐矩阵乘加，但它不是任意 Python 乘法的自动加速器。程序需要把数据准备为合法类型、尺寸和 fragment 布局，还要保证参与 thread 满足指令契约。在 SM120 上理解这些条件，比记忆某个数据中心芯片的峰值数字更有用。

## 5.1 从标量 FMA 到矩阵 MMA

标量融合乘加计算 $d=ab+c$，一次记为两次浮点运算。矩阵乘加计算

$$
D_{m,n}=C_{m,n}+\sum_{k=0}^{K-1}A_{m,k}B_{k,n}.
$$

一个 $M\times N\times K$ 的 MMA 对应 $2MNK$ FLOP。warp 级 MMA 把 A、B 与累加器 fragment 分布在 32 个 lane 的寄存器中；一整个 warp 合作产生一个小矩阵结果。不是每个 lane 各自拥有一个完整矩阵，也不是只让一个 lane 发出这条 warp 指令即可。

对常见的 FP16/BF16 输入、FP32 累加，SM120 可以采用形如下面的 PTX 路径：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

这是指令形状的说明，不是可以独立粘贴执行的完整 PTX 程序。$m16n8k16$ 表示输出 16×8，归约长度 16，一次完成 $2\times16\times8\times16=4096$ FLOP。输出有 128 个 FP32 元素，典型 fragment 分配使每 lane 持有四个累加结果；A/B 的实际寄存器打包与元素所有权应以 PTX 规范和生成代码为准。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

`mma.sync` 的 `sync` 表示 warp 参与语义，不是 host 同步或 CTA barrier；`.aligned` 也要求参与一致。TileLang 的 `T.gemm` 在目标支持范围内组织 MMA fragment 和执行参与者。用户仍须让 tile、dtype、thread 配置和所有权满足编译器要求，不能只让一个 lane 执行整条 warp MMA。

::::

::::{tab-item} Triton
:sync: triton

`sync` 表示 warp 参与和同步语义，不能理解为整个 kernel 与 CPU 同步，也不能当作 CTA barrier。`.aligned` 带有参与一致性要求；让同一 warp 的部分活跃 lane 随意跳过 MMA 是不正确的低层写法。Triton 将矩阵 tile 降低为 MMA 时负责组织这种参与关系。

::::
:::::

## 5.2 本设备的累加器在寄存器中

```{figure} static/foundations-mma.svg
:name: fig-mma
:alt: SM120 的 A B 从全局内存进入共享内存或寄存器，再由 warp mma.sync 更新寄存器累加器

本书使用的 SM120 数据路径。共享内存是否参与、由哪种 load 完成，由具体内核与编译结果决定。
```

数据中心 SM100 常见的 `tcgen05.mma` 把累加器放在 TMEM，需要分配、提交、等待与读回协议。**SM120 不能把这套指令和 TMEM 管理代码直接拿来用。** 本书的低精度 GEMM 使用 warp 级 MMA，长期累加值占据寄存器，因此 tile 大小时要直接计算累加器压力。

例如一个 CTA 的输出 tile 为 64×128，共有 8192 个 FP32 累加元素；如果由四个 warp 的 128 个 thread 大致均摊，仅累加器平均就需要 64 个 32 位寄存器。地址、A/B fragment、掩码与临时值还在这之外。换成 128×128 时，平均累加器预算升为 128 个。这个估算解释了为什么更大的输出 tile 可能降低驻留 CTA 数，甚至产生 spill。

spill 将部分本应驻留寄存器的状态放到 local memory 路径。local 指线程私有地址空间，并不意味着片上低时延存储；实际访问可能经过缓存与外部显存。应从编译元数据或分析器找证据，而不是仅凭某个 tile 慢就断言 spill。

## 5.3 输入精度、累加精度与输出精度

FP16 与 BF16 都占两字节，但数值范围不同。FP16 有更多尾数精度、较窄指数范围；BF16 的指数位与 FP32 相同、尾数更短。把输入转成低精度后，FP32 累加只能减少后续求和误差，不能恢复已经舍入掉的输入信息。

记真实输入为 $A,B$，低精度表示为 $\hat A=A+\Delta A,\hat B=B+\Delta B$，即使累加完全精确，也有

$$
\hat A\hat B-AB
=\Delta A B+A\Delta B+\Delta A\Delta B.
$$

实际还会出现累加舍入与输出转换。验证时应把“相同低精度输入、FP32 计算的参考值”作为主要对照，并注明最终输出 dtype。对照全 FP32 原始输入可能是在同时测量量化误差和 kernel 误差，需要明确区分。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

`T.gemm(A_shared, B_shared, accumulator)` 在 tile 层表达矩阵乘加。A/B 保存低精度输入，`T.alloc_fragment(..., "float32")` 保存累加器，调用之前必须用 `T.clear` 初始化。数学类型不自动保证具体指令，FP32/TF32 路径要另行选择和检查；本书主线固定 FP16/BF16 输入与 FP32 累加。

::::

::::{tab-item} Triton
:sync: triton

`tl.dot(a,b,acc)` 是矩阵乘加语义，输入类型与编译目标共同决定具体实现。FP32 输入可能使用 TF32 等路径；要控制它，应查看 `input_precision` 选项与后端支持。本书主线采用 FP16/BF16 输入、FP32 累加，避免把 TF32 截断混入基础数值解释。

::::
:::::

## 5.4 合法 tile 与有效 tile 是两件事

真实矩阵的 M/N/K 不必恰好等于指令形状。编译器可以用较大的逻辑 tile、掩码和补零覆盖尾部。假设 K=30、内部 K tile 为 32，最后两个位置需要被安全加载为 0，从而保持乘法语义。M/N 尾部对应的结果可以算出但不写入有效输出之外。

不规则尺寸会增加无效计算。一个实际 17×17 的输出被 32×32 tile 覆盖时，有效输出比例约为 $289/1024$。这不意味着不能使用 Tensor Core，而是说明针对小矩阵需要考虑 tile 利用率、CTA 数与发射开销，不能只按峰值 FLOP/s 预测收益。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import matmul

torch.manual_seed(0)
a = torch.randn((65, 96), device="cuda", dtype=torch.float16)
b = torch.randn((96, 79), device="cuda", dtype=torch.float16)
c = matmul(a, b, variant="tiled")
reference = a.float() @ b.float()
error = (c.float() - reference).abs()
print("max abs error:", error.max().item())
torch.testing.assert_close(c.float(), reference, rtol=2e-2, atol=2e-2)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060.fundamentals import matmul

torch.manual_seed(0)
a = torch.randn((65, 96), device="cuda", dtype=torch.float16)
b = torch.randn((96, 79), device="cuda", dtype=torch.float16)
c = matmul(a, b, variant="tiled")
reference = a.float() @ b.float()
error = (c.float() - reference).abs()
print("max abs error:", error.max().item())
torch.testing.assert_close(c.float(), reference, rtol=2e-2, atol=2e-2)
```

::::
:::::

容差需要结合输入尺度、K 长度与输出类型解释。相对误差在接近零的位置会很大，绝对误差又可能掩盖大数的相对偏差。一个稳健的结果表至少检查最大绝对误差、相对或归一化误差，以及 NaN/Inf。这里的容差适合启动小规模实验，不构成对任意分布的误差保证。

### 从实际 tile 读出精度

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/fundamentals.py
:language: python
:pyobject: _matmul_tiled_kernel
:linenos:
:caption: FP16/BF16 输入与 FP32 累加器
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _matmul_tiled_kernel
:linenos:
:caption: FP16/BF16 输入与 FP32 累加器
```

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

先找 `T.alloc_shared`、`T.alloc_fragment`、`T.clear` 和 `T.gemm`。把输入 scope、累计类型、清零位置和最终转换圈出来；同一输出块的 K 循环中不得重复清零。后续第九章展开全部索引证明。

::::

::::{tab-item} Triton
:sync: triton

先找 A/B 的 masked load、FP32 accumulator 与 `tl.dot`。圈出每轮累加器更新和最后一次 store；输入低精度与累计 FP32 是两项独立决定。第九章展开全部索引证明。

::::
:::::

## 5.5 编译证据比函数名可靠

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

使用 `T.gemm` 后，仍要检查实际设备源码、PTX 或 cubin 是否包含当前目标所需的 `mma.sync`。函数名和 IR 的矩阵语义只能说明意图；生成指令与 profiler 分别回答“生成了什么”和“运行瓶颈在哪里”。TileLang 的自动 lowering 也需要这两层证据。

::::

::::{tab-item} Triton
:sync: triton

使用了 `tl.dot`，还应查看生成 PTX 是否确实含有适合目标设备的 MMA；使用了某个名为 `tensorcore` 的函数，也不能证明计算已经走到该路径。反汇编能回答“生成什么”，profiler 能回答“运行时瓶颈是什么”。两种证据结合，才能判断 Tensor Core 空闲究竟是因为指令没有生成、输入供应不足，还是并行工作不够。

::::
:::::

PTX 的架构和指令要求见 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。阅读时检查 `Target ISA Notes`，不要只根据“Blackwell”章节标题作推断。

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

`results/gemm.json` 中，FP16 的 512³ 分块 GEMM 最大绝对误差为 0.03125，累计后的结果还要存回 FP16。普通提交中位时间为 31.760 μs，而同模式 PyTorch 为 18.512 μs。使用 Tensor Core 是必要的实现事实，仍不足以保证战胜库函数。

::::
:::::

## 练习

1. 计算 32×64×32 一次 tile 更新的 FLOP 数。若最终 K=256，需要几次这样的 K tile 更新？
2. 对四个 warp 的 CTA，估算 32×64、64×64、64×128 输出 tile 的平均 FP32 累加器寄存器数。为什么真实 `n_regs` 应该更大？
3. 构造包含大幅值的 FP16 输入和 BF16 输入，先在 CPU 上理解表示范围，再预测哪一种更容易出现 Inf。不得把 Inf 输出当作普通浮点误差。
4. 在后面章节的 GEMM 编译结果中找出 MMA 指令，记录目标架构、操作数类型与累加类型。若未找到，先列出待检查条件，再修改代码。
