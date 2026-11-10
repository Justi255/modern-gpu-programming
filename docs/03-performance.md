(ch-performance)=
# 03　性能模型：先算字节，再看时间

优化需要一个能够被测量推翻的解释。只说“GPU 擅长并行”无法判断增大 tile、融合算子或增加流水级数是否有用。本章用工作量、数据量、并行度与计时边界建立一个可用的模型。模型不必预测每一纳秒，但应指出下一次实验要改变什么、观察什么。

## 3.1 两个工作量与一个下界

设计算工作量为 $F$ FLOP，跨某一层存储边界传输的数据为 $Q$ 字节，对应计算吞吐上限 $P$ 与带宽上限 $W$。理想化时间满足

$$
t\ge\max\left(\frac{F}{P},\frac{Q}{W}\right),
\qquad I=\frac{F}{Q},
\qquad P_{\rm achieved}\le\min(P,IW).
$$

$I$ 是算术强度，单位 FLOP/字节。这是 roofline 模型的核心。$Q$ 必须注明边界：HBM/显存到缓存、L2 到 SM、共享内存到寄存器，会得到不同的强度。RTX 5060 使用的外部显存也不应笼统写成数据中心 HBM。本文默认先用“逻辑全局数据量”作粗模型，再用 profiler 区分真实流量。

```{figure} static/foundations-roofline.svg
:name: fig-roofline
:alt: 无具体数值的 roofline 示意图，低算术强度受带宽约束，高算术强度受计算峰值约束

Roofline 是约束示意图。坐标没有填入本设备的未测峰值。
```

向量加法 $z=x+y$ 每个元素做一次加法。FP32 输入读两次、输出写一次，逻辑数据量是 $12N$ 字节，强度为 $1/12$ FLOP/字节。因此只减少一条地址指令通常不如保证合并访问重要；把 FP32 加法换成 Tensor Core 也不是这个问题的合理方向。

矩阵乘法 $C=AB$ 的工作量约为 $2MNK$，忽略读取旧 C 时，理想最少全局数据量为

$$
Q_{\min}=s_A MK+s_B KN+s_C MN.
$$

但“最少”意味着每个输入只读一次，实际 tile 划分可能在多个 CTA 之间重复读 A 或 B。FP16 方阵 $M=N=K=1024$，输入输出均按 2 字节计算，$F=2^{31}$ FLOP，$Q_{\min}=6\times2^{20}$ 字节，理想强度约为 $341.3$ FLOP/字节。这是数学推导，不是实测吞吐；小矩阵还可能被 launch 与并行度限制。

## 3.2 tile 为什么能够提高复用

一个 CTA 计算 $B_M\times B_N$ 输出，在一个 K tile 内读取 $B_M B_K$ 个 A 元素和 $B_K B_N$ 个 B 元素，完成 $2B_MB_NB_K$ FLOP。两路输入均为 FP16 时，这一步的输入强度为

$$
I_{\rm tile}=
\frac{2B_MB_NB_K}{2(B_MB_K+B_KB_N)}
=\frac{B_MB_N}{B_M+B_N}.
$$

这里 $B_K$ 消去了。增大 $B_K$ 仍可能减少循环、改善指令调度，却不直接改善这个简化的 A/B 复用比。把输出 tile 从 $32\times32$ 改成 $64\times64$，模型强度从 16 增为 32 FLOP/字节，同时累加器元素数增为四倍。模型必须把节省的流量与新增资源放到同一张账本上。

## 3.3 时延、吞吐与固定开销

吞吐高不代表单次请求时延低。小 decode 可能只有几个 CTA，远未达到设备的饱和吞吐；再漂亮的算术强度也不能创造缺失的并行工作。相反，大批量 GEMM 的单次时延较长，却可能有更高 TFLOP/s。

粗略写成

$$
t_{\rm observed}\approx t_{\rm fixed}+t_{\rm work},
$$

其中固定部分可能包含 host dispatch、launch、图重放与计时开销，具体取决于测量边界。把一个 3 微秒工作减少到 2 微秒，如果完整调用还需额外开销，端到端收益会小于 kernel 时间比例。这些数值只是解释固定成本的假设例子，不是本设备结果。

CUDA Graph 将重复工作记录并重放，可以减少逐次 host dispatch 成本；它也要求稳定的地址、显式工作区和避免 capture 中分配。比较 eager 与 graph 时应分别报告，不能把一种路径的基线与另一种路径的新实现相除。

## 3.4 计时边界是实验的一部分

CUDA launch 默认异步返回。下面的 host 计时只测提交耗时，不能直接称为 kernel 耗时：

```python
# 这是反例：两个时间点之间未必包含 GPU 完成。
import time
start = time.perf_counter()
result = operation()
elapsed = time.perf_counter() - start
```

GPU events 应放在同一个 stream，并在读取结果前等待。以下是可改用本书算子的最小示例：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import vector_add

x = torch.randn(1 << 20, device="cuda")
y = torch.randn_like(x)
for _ in range(10):
    vector_add(x, y)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    z = vector_add(x, y)
end.record()
end.synchronize()
print("mean milliseconds:", start.elapsed_time(end) / 100)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060.fundamentals import vector_add

x = torch.randn(1 << 20, device="cuda")
y = torch.randn_like(x)
for _ in range(10):
    vector_add(x, y)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    z = vector_add(x, y)
end.record()
end.synchronize()
print("mean milliseconds:", start.elapsed_time(end) / 100)
```

::::
:::::

这个短例子仍包含输出分配及重复地址带来的缓存效应，不能替代完整基准协议。项目基准负责更明确地控制预热、候选配置与统计，见{doc}`appendix/benchmarking`。小规模输入反复使用同一缓存驻留数据时，有效带宽甚至可能超过外部显存带宽；这并不违反物理规律，而是在提醒你区分逻辑字节与 DRAM 字节。

## 3.5 用正确统计支持正确结论

报告中至少保留 shape、dtype、mask、缓存布局、实现版本和 timing 模式。多轮结果用中位数减少偶发干扰，再看分位数或离散程度。GPU 上同时运行其他程序、笔记本温度爬升以及自动调频都可能改变排序。两种实现若差异小于轮间波动，应报告差异不确定。

speedup 定义为同一工作负载上的 $t_{\rm baseline}/t_{\rm candidate}$。跨工作负载汇总时，几何平均数适合表达比例，但仍需保留逐项结果；少数大胜不能掩盖关键场景的大幅回退。没有可运行的相同语义原生基线时，结果应写“基线缺失”，不能据此宣称 SOTA。

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

本机基础实验反复使用约 12 MiB 读写量的向量加法输入，设备 L2 为 32 MiB。因此，即使据 `results/basics.json` 算出很高的有效字节率，也不能直接称其为显存带宽。实际外存流量需要缓存控制或 profiler 证据；这正是 roofline 中“字节数”必须先界定层次的原因。

::::
:::::

## 练习

1. 分别计算 FP16 与 FP32 向量加法的逻辑算术强度。为什么两者不能仅用相同 TFLOP/s 评价？
2. 对 $B_M=64,B_N=128,B_K=32$ 计算输入强度，再把 $B_K$ 改为 64。哪些成本没有改变，哪些资源可能增加？
3. 两组微秒耗时分别为 `[10,10,10,11,40]` 和 `[11,11,11,11,11]`，比较均值与中位数给出的排序，解释各自反映什么。
4. 设计一项实验区分“tile 太大导致并行度不足”和“tile 太大导致寄存器 spill”。说明固定的变量、改变的变量和必要证据。
