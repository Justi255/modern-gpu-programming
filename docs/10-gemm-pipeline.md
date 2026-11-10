(gemm-pipeline)=
# 10. 流水线：让搬运与计算重叠

分块 GEMM 已经减少了重复搬运，但它仍可能不断经历「等待数据、执行乘加、再等待数据」的周期。流水线试图在当前 K 块计算时准备未来的 K 块。这是一种关于依赖关系与资源寿命的安排，不是给循环加一个参数就保证提速。本章用本项目的 `pipelined` 变体学习这种安排，并说明在 SM120 上应如何验证编译器实际做了什么。

## 10.1 先写出时间下界

设每个 K 块的搬运时间为 $L$，计算时间为 $C$，共有 $T$ 个块。完全串行的理想时间为 $T(L+C)$。若独立硬件路径允许理想重叠，启动后每轮只需等待两者中较慢的一个，总时间近似为

$$
t_{\rm pipe}\approx L+C+(T-1)\max(L,C).
$$

这个模型说明两个事实。第一，只有足够多的循环轮次才能摊薄流水线填充与排空成本；短 K GEMM 可能没有可隐藏的稳定阶段。第二，即使无限增加缓冲区，也不能让时间低于主要资源的服务时间。若 Tensor Core 已经连续饱和，增加预取距离只会占用更多空间。

```{figure} static/kernels-pipeline.svg
:alt: 串行加载计算和双缓冲加载计算的时间线，缓冲区在消费完成后才可复用
:width: 95%

重叠改变开始时刻，不改变数据必须先完成才能被使用的依赖。
```

## 10.2 两类依赖不能混淆

生产者把第 $i$ 块数据写入某个缓冲槽，消费者随后读取它。这里有「数据已准备好」的依赖：消费者必须等待对应加载完成。消费者使用完毕以后，生产者才能把第 $i+S$ 块写进同一个槽，其中 $S$ 是槽数。这里还有「槽可复用」的依赖：加载下一块不能覆盖仍在参与矩阵指令的旧数据。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 用 shared buffer 表达生产消费，用 `T.Pipelined` 为 K 循环给出软件流水机会，再由 lowering 插入必要等待与复用同步。`num_stages` 影响缓冲容量和调度距离，不等于越大越快。必须同时核对目标、pass、真实搬运指令和资源报告，不能只看 Python 参数。

::::

::::{tab-item} Triton
:sync: triton

手工编写异步内核时，这两类依赖可能由不同 barrier 或提交组表示。Triton 常把它们表达为循环中的值依赖，再由编译器进行软件流水调度、共享内存 staging 与等待指令插入。因此 `num_stages` 应理解为编译器调度参数；它并不等同于 Python 创建了同样数量的数组，也不保证每个架构使用完全相同的异步指令。必须检查目标架构、Triton 版本、编译产物与实际性能。

::::
:::::

本书不把上游 B200 的 TMA → `tcgen05.mma` → TMEM 协议直接搬到 RTX 5060。SM120 的可执行路径以当前编译器支持的 warp 级矩阵操作为依据；是否使用 `cp.async`、共享内存 swizzle 或其他加载形式，应以本机生成代码为证据。算法层面的生产消费关系可迁移，具体指令与 barrier 名称需要重新确认。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/fundamentals.py
:language: python
:pyobject: _matmul_pipelined_kernel
:linenos:
:caption: 在 K 循环上表达 num_stages 的软件流水变体
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _matmul_pipelined_kernel
:linenos:
:caption: 在 K 循环上表达 num_stages 的软件流水变体
```

::::
:::::

默认教学配置为 `tile_m=32, tile_n=64, tile_k=32, num_warps=4`，`pipelined` 的 `num_stages=3`。公共 `config` 字典可以覆盖这五个键；合法范围与资源预检在 `matmul` 的源码中定义。未知键会报错，配置超出保守预算或编译失败也不会偷偷退回 PyTorch GEMM。

## 10.3 为缓冲区做容量预算

忽略布局 padding，一个 fp16 K 槽需要

$$
S_{\rm one}=2B_K(B_M+B_N)\ \text{bytes}.
$$

若使用 $S$ 个槽，则输入 staging 至少需要 $S S_{\rm one}$。例如 $B_M=B_N=64,B_K=32$ 时，单槽逻辑容量为 8 KiB；这是由尺寸算出的容量示例，不是本机编译器资源报告。实际编译器还可能添加 padding、改变共享布局或保留其他临时缓冲。请读取资源信息，而不要把公式结果当作 launch 配置的保证。

共享内存容量决定每个 SM 可同时驻留多少 CTA，寄存器容量也有类似限制。把 stages 从 2 增到 4，可能隐藏更多加载延迟，也可能让同时驻留的工作减少一半。对只有 26 个 SM 的笔记本 GPU，这种阶跃会显著改变全局并发。分析时应分别问：单个 CTA 是否更快，以及整个网格是否仍有足够多的 CTA 同时运行。

## 10.4 在正确性相同的前提下比较变体

下面的例子把所有变体与同一个 fp32 参考比较。它只用于结果验证；不要用包围这个 Python 循环的墙钟时间比较内核，因为其中包含分配、编译与断言。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import matmul

torch.manual_seed(10)
a = torch.randn((512, 1024), device="cuda", dtype=torch.float16)
b = torch.randn((1024, 384), device="cuda", dtype=torch.float16)
expected = (a.float() @ b.float()).half()
for variant in ("tiled", "pipelined", "persistent"):
    actual = matmul(a, b, variant=variant)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)
    print(variant, tuple(actual.shape))
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060.fundamentals import matmul

torch.manual_seed(10)
a = torch.randn((512, 1024), device="cuda", dtype=torch.float16)
b = torch.randn((1024, 384), device="cuda", dtype=torch.float16)
expected = (a.float() @ b.float()).half()
for variant in ("tiled", "pipelined", "persistent"):
    actual = matmul(a, b, variant=variant)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)
    print(variant, tuple(actual.shape))
```

::::
:::::

统一实验入口会记录各变体的数据，后续分析应从实际 JSON 出发：

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

比较至少覆盖短 K 与长 K、方形与长条输出。若只选择很大的方阵，容易把流水线启动成本、边界空算和低并行度等问题全部隐藏起来。同一配置在一种形状上的优势，不应直接推广到另一种形状。

## 10.5 用证据区分三种瓶颈

若共享内存和寄存器增加了，时间却几乎不变，可能是算力已经限制吞吐，也可能预取没有成功重叠。若时间改善但有效带宽没有明显增长，可能只是减少了等待间隙。若短 K 变慢而长 K 变快，则流水线启动成本的解释较合理。以上都是待检验的假设，不能只凭曲线形状确诊。

先用无 profiler 的计时确定现象，再用 Nsight Systems 查看主机提交间隙与 GPU 工作连续性，最后使用 Nsight Compute 针对代表性内核检查资源和停顿。Profiler 会扰动时间，尤其是指标重放与串行化；不要拿 profiler 中的一次内核时长替换公开基准的稳定计时。具体命令与边界说明见[测量附录](appendix/benchmarking.md)。

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

本机 FP16 512³ 实验的基础与流水线普通提交中位时间为 31.760 与 30.400 μs；BF16 为 30.064 与 29.888 μs，见 `results/gemm.json`。这点差异尚不足以独立归因为访存重叠，应结合生成代码，并把图内批量计时和普通提交分开。

::::
:::::

## 练习

1. 代入 $T=2$ 与 $T=32$，推导 $L=C$ 时理想流水线相对串行的加速上限。为什么较深流水线对前者通常不划算？
2. 为一个三槽环形缓冲写出连续六次迭代的槽号与使用者，指出每次覆盖前必须已满足的条件。
3. 选择一组短 K 与一组长 K 形状，在相同输入和计时边界下比较 `tiled` 与 `pipelined`。若结果不符合预期，先记录观测，再提出两个互相竞争的解释。

本章保留上游 `chapter_gemm_async` 的依赖分析方法，并按 SM120 与两种 DSL 重新解释执行机制；固定版本来源见[迁移表](appendix/sources.md)。
