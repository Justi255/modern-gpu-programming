(decode)=
# 14. Decode：用 split-KV 补足并行度

decode 每条序列通常只有一个新查询。如果直接让一个 program 扫描整段 KV，程序数大致与批次和查询头数成正比；小批次时，长缓存中的大量工作被少数 program 串行承担。与 prefill 相比，它更容易受到显存带宽、加载延迟和并行度限制。split-KV 把一条逻辑 KV 序列切成若干段并行处理，再用一个小归约合并在线 softmax 状态。

## 14.1 读缓存的语义先于切分策略

调用形式是 `decode(q, k, v, *, kv_lens, block_table=None, scale=None, out=None, workspace=None, num_splits=None, config=None)`。Q 为 `[B, Hq, D]`，输出同形状。K/V 可以是稠密缓存，也可以是上一章的物理页布局；分页要求 GPU int32 的 `kv_lens` 与合法块表。

缓存已经包含当前 token，decode 读取 $j=0,\ldots,L_b-1$ 的所有有效键。内核不追加 K/V、不更新长度、不分配物理页。若框架先把当前 token 写进缓存再调用本函数，参考与所有性能基线都必须看到更新后的同一份缓存。否则即使输出误差很小，也是在比较不同的数学问题。

## 14.2 为什么把 KV 轴切开

假设批次为 B、查询头数为 $H_q$。逐查询头的 vector 路径在不切分时有 $BH_q$ 个工作单元；拆成 R 段后，第一阶段有 $BH_qR$ 个工作单元。grouped 路径则让一组共享 KV 的查询头一起计算，第一阶段工作数为 $BH_{kv}R$。每个工作只读取自己负责的连续逻辑 KV 区间。物理分页并不妨碍逻辑连续切分，因为每个 token 仍能通过页表独立寻址。

当前分段边界由缓存容量 C 确定：每段跨度为 $\lceil C/R\rceil$，第 r 段从 $r\lceil C/R\rceil$ 开始，以有效长度 $L_b$ 截断。调度不依赖把 GPU 上的实时长度读回主机，因此可保持图捕获所需的调用方式；代价是容量很大、实际长度很短时，可能启动许多空段。默认 split 数是按容量与工作数量作出的启发式选择，显式 `num_splits` 允许在 1 到 64 中选择二的幂。

更多段提高并行度，但也增加 scratch 写入、第二阶段读入与合并开销，还可能让每段太短、启动成本占比过高。较大批次原本就能填满 GPU，过度切分往往适得其反。因而 `num_splits` 是可测的工作划分参数，不是越大越好。本书不把某个固定分段数当作所有上下文长度的答案。

```{figure} static/kernels-split-kv.svg
:alt: 一个查询的 KV 区间由多个 program 并行计算局部 softmax 状态，再合并成最终输出
:width: 96%

split-KV 的第二阶段合并的是带指数基准的状态，不能直接求局部输出的平均值。
```

## 14.3 推导可合并状态

第 r 段维护 $m_r,\ell_r,u_r$，含义与在线 softmax 相同。令全局最大值为 $m=\max_r m_r$，则

$$
w_r=e^{m_r-m},\qquad
\ell=\sum_r w_r\ell_r,\qquad
u=\sum_r w_ru_r,\qquad O=u/\ell.
$$

每一段的局部指数都变换到同一个最大值，才能相加。若 scratch 保存的是已归一化局部输出 $o_r=u_r/\ell_r$，则合并权重应是 $w_r\ell_r$，不是只使用 $w_r$。阅读代码时应先辨认 scratch 中每个张量的定义，再对照公式，不能只看变量名 `partial_o` 就猜测它是否归一化。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 也保存未归一化 FP32 `partial`，与 `maxima`、`sums` 组成可合并状态。使用 `exp2` 的实现先把分数乘 `scale * log2(e)`，merge 也用相同的二进制指数基准，不能重复乘换底系数。用一组极大、极小 logits 和空段验证这项不变量，比只看普通随机误差更有价值。

::::

::::{tab-item} Triton
:sync: triton

本项目保存的是未归一化 `partial`。源码中的分数先乘 $\text{scale}\log_2e$，所以 scratch 中的 `maxima` 也在以 2 为底的指数域，合并使用 `exp2(maxima-safe)`。这与上面的自然指数公式等价，不能再额外乘一次换底系数。

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/attention.py
:language: python
:pyobject: _merge_splits
:linenos:
:caption: 第二阶段合并：共同最大值、指数校正与一次最终归一化
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: _merge_splits
:linenos:
:caption: 第二阶段合并：共同最大值、指数校正与一次最终归一化
```

::::
:::::

空段必须携带零分母和零加权和，并在求指数时避免 $-\infty-(-\infty)$。当整条序列长度为零时，最终分母仍为零，输出按接口规定写零。分段数多于有效 token 数、不同批次长度差别很大以及最后一段不足块大小，都是必要测试。

## 14.4 scratch 的成本与生命周期

若每段保存一个 D 维 fp32 向量和两个 fp32 标量，逻辑 scratch 规模约为

$$
4BH_qR(D+2)\quad\text{bytes}.
$$

本项目返回 `DecodeWorkspace` 数据类：`partial` 是 `[B,Hq,R,D]`，`maxima` 和 `sums` 都是 `[B,Hq,R]`，三者均为连续 CUDA fp32 张量；另有主机整数 `num_splits` 标识分段数。因此上述公式对应当前三块张量的逻辑容量。两阶段算法多付出一次全局写和一次全局读，收益来自把更长的 KV 流量分给更多并行工作。只有当减少的串行等待大于这些额外成本时，split-KV 才更快。

`decode_workspace(q, k, *, block_table=None, num_splits=None, config=None)` 提供显式 scratch。分配与执行应使用相同的 config、块表与 split 数，尤其默认选择可能随 mode 改变。调用时必须让 workspace 的形状、设备、类型与分段策略匹配本次 decode。三个 scratch 张量占用的地址范围不能彼此重叠，也不能与 Q/K/V、长度、页表或输出的地址范围重叠；允许使用同一底层 allocation 中互不重叠的连续切片。包装按地址范围检查，对带步长输入使用保守的外包范围。并发流或重叠调用必须使用不同 workspace；一次调用结束以前不能把其中张量交给别的计算覆盖。输出张量有同样的生命周期约束，且不能与 Q/K/V 共享底层存储。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/_contracts.py
:language: python
:pyobject: decode_workspace
:linenos:
:caption: 在调用或 CUDA Graph 捕获之前显式分配 scratch
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: decode_workspace
:linenos:
:caption: 在调用或 CUDA Graph 捕获之前显式分配 scratch
```

::::
:::::

当 R=1 时，第一阶段直接完成归一化并写输出，不启动 `_merge_splits`，执行也不需要 scratch。R>1 且未提供 workspace 时，包装函数会自动分配；这方便普通使用，但分配成本应与预先分配的稳定内核时间区分开。

## 14.5 显式分配的可运行例子

以下代码先分配输出和工作区，再调用 decode。用不同有效长度覆盖部分段为空的情形，参考使用同样的 KV 内容与 GQA 映射。分段数在分配和执行中保持一致。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import decode, decode_workspace
from fa5060.reference import attention_reference

torch.manual_seed(14)
q = torch.randn((2, 8, 64), device="cuda", dtype=torch.float16)
k = torch.randn((2, 257, 2, 64), device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
lengths = torch.tensor([257, 17], device="cuda", dtype=torch.int32)
out = torch.empty_like(q)
scratch = decode_workspace(q, k, num_splits=4)
decode(q, k, v, kv_lens=lengths, out=out,
       workspace=scratch, num_splits=4)
ref = attention_reference(q, k, v, causal=False, kv_lens=lengths)
torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060 import decode, decode_workspace
from fa5060.reference import attention_reference

torch.manual_seed(14)
q = torch.randn((2, 8, 64), device="cuda", dtype=torch.float16)
k = torch.randn((2, 257, 2, 64), device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
lengths = torch.tensor([257, 17], device="cuda", dtype=torch.int32)
out = torch.empty_like(q)
scratch = decode_workspace(q, k, num_splits=4)
decode(q, k, v, kv_lens=lengths, out=out,
       workspace=scratch, num_splits=4)
ref = attention_reference(q, k, v, causal=False, kv_lens=lengths)
torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
```

::::
:::::

分页版本只需把 K/V 改为 `[P,page,Hkv,D]`，并在 workspace 分配、decode 和参考调用中传入同一份 `block_table`。注意这里不能对已经分页的张量调用 `.reshape(B, capacity, Hkv, D)` 充当逻辑顺序：物理页排列可以任意打乱。

CUDA Graph 要求捕获期间不触发需要主机参与的首次编译或隐式内存分配。显式 `out` 与 workspace 是基础，还应在捕获之前完成对应形状的预热，并检查完整调用链没有同步取回 GPU 标量。图重放比较也必须让基线使用相同的分配和捕获边界。图本身降低主机提交成本，并不会减少 KV 的数学读取量。

## 14.6 vector、grouped 与公共 dispatch

`config['mode']` 可以选择 `vector`、`grouped` 或 `auto`。vector 路径让一个 program 负责一个查询头，加载 K 后以 fp32 元素乘法和归约计算分数，再以 fp32 加权归约计算输出。它不依赖把一个查询伪装成足够大的矩阵，适合较小的 GQA 组或较短容量。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/attention.py
:language: python
:pyobject: _decode_vector_kernel
:linenos:
:caption: 逐头 vector 路径：元素运算、fp32 归约与可选 split 状态
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: _decode_vector_kernel
:linenos:
:caption: 逐头 vector 路径：元素运算、fp32 归约与可选 split 状态
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang grouped 路径把共享 KV 头的 G 个查询头组织成矩阵行，用 `T.gemm` 完成 QK 与 PV，并用 FP32 fragment 保留在线统计和输出累计。行数为矩阵指令补齐到至少 16，G 小时存在空算；概率在 PV 前转换成输入类型。复用 KV 的同时 CTA 数从 $BH_qR$ 降为 $BH_{kv}R$，所以必须与 splits 一起测量。

::::

::::{tab-item} Triton
:sync: triton

grouped 路径把同一 KV 头对应的 G 个查询头作为矩阵的行，使用 `tl.dot` 完成 QK 与 PV，两次矩阵乘法共用同一在线归约结构。行数会向上补到至少 16，以满足矩阵计算的形状需要；G 较小时会支付较多空算。概率在 PV 前转成查询类型，最大值、分母和累计仍为 fp32。这个路径减少同一 program 内的 KV 重复加载，但也把第一阶段 program 数从 $BH_qR$ 降为 $BH_{kv}R$，所以通常需要同时考虑 split 数。

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的 grouped 入口复用第十三章的在线 Tensor Core factory；下列短函数表达 dispatch，完整 shared/fragment 主循环见 `_prefill_kernel`。

```{literalinclude} ../src/fa5060/tilelang/attention.py
:language: python
:pyobject: _decode_grouped_kernel
:linenos:
:caption: 同 KV 头的查询组：矩阵乘法复用与在线归约
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: _decode_grouped_kernel
:linenos:
:caption: 同 KV 头的查询组：矩阵乘法复用与在线归约
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的 `mode="auto"` 与原接口使用相同初始规则：G 至少为 4 且容量至少为 1024 时选 grouped，其余选 vector；这不是逐形状最优保证。`block_n`、`num_warps` 和 `num_stages` 控制对应调度，vector 使用单 stage，grouped 对 KV 循环应用流水策略。decode 只接受 pointer loader。

::::

::::{tab-item} Triton
:sync: triton

当前 `auto` 在 G 至少为 4 且容量至少为 1024 时选择 grouped，否则选择 vector。这是明确、可检查的初始 dispatch 规则，不代表每个形状的最优点。`block_n`、`num_warps` 与 `num_stages` 控制相应的块、warp 与流水参数；vector 固定使用单 stage，grouped 的循环使用配置中的 stages。decode 只接受 pointer loader。

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/attention.py
:language: python
:pyobject: decode
:linenos:
:caption: 真实 decode 包装：输入检查、路径选择、scratch 校验与最终合并
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: decode
:linenos:
:caption: 真实 decode 包装：输入检查、路径选择、scratch 校验与最终合并
```

::::
:::::

长度、活动页号合法性及连续最后一维的合同与 prefill 相同。包装只检查主机可见的形状、dtype、设备与 stride，不在每次调用前同步检查 GPU 元数据的数值。图捕获不会代替调用者维护这些不变量。

## 14.7 带宽模型与公平比较

忽略缓存复用时，每条序列读取 K 和 V 的最低逻辑字节数约为 $2L_bH_{kv}Ds$，其中 s 为每元素字节数。GQA 会让多个查询头共享 KV，但实际读取次数取决于内核的跨头组织与缓存行为。因此用「逻辑字节数除以延迟」得到的是有效带宽，不等于 DRAM 控制器测得的总流量带宽。若有效带宽大于硬件带宽，先检查复用与分子定义，不要立即宣布超越物理限制。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m fa5060.benchmark --backend tilelang --suite smoke --phase decode --output results/local/tilelang/decode-smoke.json
python -m fa5060.benchmark --backend tilelang --suite full --phase decode --rounds 5 --output results/local/tilelang/decode-full.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m fa5060.benchmark --backend triton --suite smoke --phase decode --output results/local/triton/decode-smoke.json
python -m fa5060.benchmark --backend triton --suite full --phase decode --rounds 5 --output results/local/triton/decode-full.json
```

::::
:::::

预先分配的 steady-state 时间和包含分配的单次 API 时间回答不同问题。公开比较应固定其中一种边界，并说明 scratch 需求。原生分页基线不可被先 gather 后调用的稠密路径取代；若某个 provider 不支持本机架构或该语义，它应标记为不兼容或缺失，不能写成自定义内核的胜利。

## 本机实验观察

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

{download}`attention.json <../results/tilelang/attention.json>` 的 decode 使用 Q=[3,4,D]、Hkv=2、KV 有效长度 [79,53,0]、容量 80。默认 auto 在该形状选择 vector，pointer `block_n=64`、4 warps，显式 `num_splits=2`；每次完整调用包含 main 与 merge。

| dtype / D | dense μs | paged μs | 两布局最大误差 |
|---|---:|---:|---:|
| FP16 / 64 | 129.984 | 132.832 | 0.00006103515625 |
| BF16 / 128 | 117.584 | 120.896 | 0 |

四条记录均通过 blockwise FP32 参考。长度为 0 的 batch 与末 split 的边界均进入实际测试；BF16 本次零误差不能推广到其他输入。计时为 eager CUDA events，预热 3 次、采样 10 次取中位；输出和 FP32 split workspace 已预分配，无缓存更新，但仍可能包含 host 提交间隙。这里没有性能基线，也没有 split=1 的同形状计时，不能由这些数字判断拆分是否获益。

独立编译导出在 B=2、Hq:Hkv=8:2、D=64、容量 1024、split=4 上确认：vector main/merge 为 REG 167/21，grouped main/merge 为 71/21；仅 grouped main 含 `mma.sync`，四个 kernel 均无 `tcgen05`。这些资源来自生成 CUDA 重编译，静态 SHARED 各 1024 B 不含动态共享分配，不能当作两种模式的吞吐或占用率比较。教学 eager 时间不与历史 Triton 图内时间相除，完整性能状态见{doc}`15-results`。

::::

::::{tab-item} Triton
:sync: triton

**历史 Triton 实验（2026-11-02，原环境）**

教学实验固定 split=2，覆盖同一组 [79,53,0] 长度和随机页表；`results/attention.json` 中稠密与分页 decode 均通过，FP16 最大误差 0.00006103515625，BF16 在这次输入上与存回低精度的参考相同。它验证了两阶段合并与空分段处理，不构成大上下文性能结论。

调优记录中的首个 decode（B=1、L=512、Hq=Hkv=32、D=64、FP16、dense）展示了 split 数的折中。以下是完整算子的一轮图内试测，包含最终 merge；正式选中配置另测五轮。

| mode | splits | 单轮试测 μs |
|---|---:|---:|
| vector | 1 | 13.844 |
| grouped | 1 | 9.706 |
| vector | 4 | 7.017 |
| grouped | 4 | 6.468 |
| vector | 16 | 11.702 |
| grouped | 16 | 10.767 |

本项选择 grouped、split=4。继续增加 split 后延迟回升，说明不能只优化第一阶段的并行度而忽略局部状态与合并工作。这还是 MHA 输入，`grouped` 只是内核模式名，不会把输入语义改成 GQA。查看 `results/full-tuned.json` 的对应候选，可以进一步比较不同 KV 头数和长度是否选择相同模式。

::::
:::::

## 练习

1. 构造两段，它们的最大分数相差很大，且局部 value 常数不同。手算正确合并值，再算简单平均，比较两者差异。
2. 取 $B=1,H_q=8,D=128$，推导 R 从 1 到 16 时 scratch 的逻辑容量与第一阶段工作数如何增长。哪些额外成本没有进入这个模型？
3. 固定同一份 KV，比较不同 `num_splits`，分别观察短缓存和长缓存。预先写下你的预测，记录所有分段数的结果，避免只保留最优点。
