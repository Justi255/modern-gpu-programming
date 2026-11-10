(attention-math)=
# 12. 注意力：把二次中间矩阵变成在线状态

注意力可以写成两个矩阵乘法和一个 softmax，但逐步调用这三个算子会把巨大的分数矩阵写回显存。FlashAttention 系列的关键贡献是改变计算的存储方式：按块产生分数，立即消费，再丢弃；保留的是每行很小的归一化状态。本章从数学恒等式出发推导这个状态，先解决「如何保持同一个答案」，再讨论 SM120 上的计算组织。

## 12.1 明确批次、头与位置

prefill 的查询形状为 `[B, Sq, Hq, D]`。稠密 K/V 的形状为 `[B, capacity, Hkv, D]`；有效长度 $L_b$ 可以小于容量。GQA 要求 $H_q$ 是 $H_{kv}$ 的整数倍，令组大小 $G=H_q/H_{kv}$，查询头 $h$ 使用 KV 头 $\lfloor h/G\rfloor$。这是连续分组的映射，不是 $h\bmod H_{kv}$。

固定一个批次与查询头，分数、概率和输出为

$$
s_{ij}=\sigma\,q_i\cdot k_j,
\qquad p_{ij}=\frac{e^{s_{ij}}}{\sum_{u\in\mathcal V_i}e^{s_{iu}}},
\qquad O_i=\sum_{j\in\mathcal V_i}p_{ij}v_j,
$$

其中 $\sigma$ 默认为 $1/\sqrt D$，$\mathcal V_i$ 是有效键的位置集合。掩码位置的分数视为 $-\infty$。存储类型可以是 FP16 或 BF16，行最大值、分母和输出累加使用 fp32。低精度乘法、指数近似和不同归约顺序仍会产生舍入差异，所以正确性检查使用明确的误差容限，而不要求逐位相同。

如果完整保存 fp32 分数，单个头需要 $4S_qL_b$ 字节；自注意力中 $S_q=L_b=L$ 时是 $4L^2$。例如长度翻倍，中间矩阵容量变为四倍。分块融合避免了这次显存物化，但并没有把精确密集注意力的乘加量变成线性；算术工作仍随查询数与键数的乘积增长。

## 12.2 从稳定 softmax 推导在线更新

对已经处理过的键集合 $A$，维护三项状态：

$$
m_A=\max_{j\in A}s_j,\qquad
\ell_A=\sum_{j\in A}e^{s_j-m_A},\qquad
u_A=\sum_{j\in A}e^{s_j-m_A}v_j.
$$

最终输出是 $u_A/\ell_A$。现在读入新块 $B$，令

$$
m'=\max(m_A,m_B),\quad
\alpha=e^{m_A-m'},\quad
\beta_j=e^{s_j-m'}\ (j\in B).
$$

旧状态只需换到同一个指数基准：

$$
\ell'=\alpha\ell_A+\sum_{j\in B}\beta_j,
\qquad
u'=\alpha u_A+\sum_{j\in B}\beta_jv_j.
$$

这不是近似地丢掉历史分数；指数的平移因子在分子和分母中相同，归一化后抵消。重要的是 $u$ 与 $\ell$ 必须同时重缩放。只改分母、或先把每块输出各自归一化再相加，都会改变数学结果。

```{figure} static/kernels-online-softmax.svg
:alt: 当前分数块与旧的最大值分母加权和合并，旧状态按新最大值重缩放
:width: 95%

一个查询块只保留每行的最大值、分母与输出向量，分数块被消费后即可释放。
```

若内核使用 `exp2`，应把分数乘以 $\log_2e$，因为 $e^x=2^{x\log_2e}$。这个换底因子与默认 attention scale 是两件事：前者由指数函数实现决定，后者由模型语义决定。把两者重复应用或漏掉任意一个，输出都可能看起来平滑，却完全不符合参考。

## 12.3 空行必须定义，不能交给 NaN

在线算法常从 $m=-\infty,\ell=0,u=0$ 开始。若一个块全部被掩码，直接计算 $-\infty-(-\infty)$ 会得到 NaN。因此实现必须用安全最大值或显式分支，让无效指数贡献为零，并在最终分母为零时写出零。本项目规定空 KV 的输出为零；右对齐 causal 也可能让某些查询行没有任何可见键。

这种处理是接口语义，不是测试中的特殊补丁。对于空行，既要避免把 NaN 写回输出，也要避免 NaN 进入中间状态后污染后续块。随机生成的长序列往往不会触发它，必须构造长度为零以及 $L_b<S_q$ 的例子。

## 12.4 右对齐 causal 的位置公式

查询块可能是已有上下文末尾的一段，而不是从位置零开始。本项目把查询行 $i$ 对齐到绝对位置

$$
a_i=L_b-S_q+i.
$$

causal 的可见条件是 $0\le j<L_b$ 且 $j\le a_i$。例如 $L_b=7,S_q=3$，三行分别位于位置 4、5、6，可见键数为 5、6、7。若 $L_b=2,S_q=4$，最前面两行对齐到负位置，因此为空行。这与单纯的 $j\le i$ 不同，也不能用 KV 容量代替 $L_b$。

decode 的公共输入是 `[B, Hq, D]`，表示每条序列当前一个查询。缓存包含当前 token，所以默认读完整个有效 KV 范围。追加缓存、分配新页和更新 block table 属于调用者的缓存管理工作，不能在计时或正确性对照中暗中混进不同语义。

## 12.5 参考实现也要避免巨型分配

一个可靠参考不必在 GPU 上创建完整的 $S_q\times L$ 矩阵。本项目的 `attention_reference` 沿查询轴分块，使用 fp32 运算，也处理稠密与分页 KV、GQA、空行和右对齐 causal。临时分数张量的规模至多约为 $H_q\times\text{chunk\_size}\times L$；它不是沿 KV 轴维护在线状态，而是让 PyTorch 对每个查询块执行直接公式。分页时，参考先按块表 gather 出逻辑 KV，长度与页号检查也会同步到主机。这些步骤服务于校验，参考不作为计时基线。

```{literalinclude} ../src/fa5060/reference.py
:language: python
:pyobject: attention_reference
:linenos:
:caption: 独立的查询分块参考：直接公式、主机元数据检查与分页 gather
```

拿待测内核和参考内核共享同一处错误逻辑仍有风险，因此还应针对小形状与直接密集公式进行独立核对。下面的 CPU 例子另外用显式行、头循环计算可见区域，避免只验证输出形状。

下面在 CPU 上构造一个可逐行检查的 GQA 例子。查询数多于有效 KV 长度，前两行应为零；这段例子不需要 CUDA。

```python
import torch
from fa5060.reference import attention_reference

torch.manual_seed(12)
q = torch.randn((1, 4, 4, 32), dtype=torch.float16)
k = torch.randn((1, 5, 2, 32), dtype=torch.float16)
v = torch.randn_like(k)
lengths = torch.tensor([2], dtype=torch.int32)
o = attention_reference(q, k, v, causal=True, kv_lens=lengths)
torch.testing.assert_close(o[:, :2], torch.zeros_like(o[:, :2]))
manual = torch.zeros_like(q)
for i in range(q.shape[1]):
    visible = max(0, 2 - q.shape[1] + i + 1)
    for h in range(q.shape[2]):
        if visible:
            scores = (k[0, :visible, h // 2].float()
                      @ q[0, i, h].float()) / (32 ** 0.5)
            manual[0, i, h] = (scores.softmax(0)[:, None]
                               * v[0, :visible, h // 2].float()).sum(0)
torch.testing.assert_close(o, manual, rtol=2e-3, atol=2e-3)
print(tuple(o.shape))
```

在目标 GPU 上运行统一 attention 实验：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m fa5060.experiments attention --backend tilelang --output results/local/tilelang/attention.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m fa5060.experiments attention --backend triton --output results/local/triton/attention.json
```

::::
:::::

### FP32 数值合同与两条执行路线

Q/K/V 使用 FP16 或 BF16，分数累计、行最大值、分母、输出累计和 split scratch 使用 FP32；PV 之前的概率转换是两套实现与全 FP32 参考之间需要审查的舍入来源。MHA 为 $H_q=H_{kv}$，GQA 为 $H_q/H_{kv}>1$，MQA 为 $H_{kv}=1$，映射均为连续分组 $h_{kv}=\lfloor h_q/G\rfloor$。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

在 TileLang 中先标出 shared Q/K/V 与低精度概率缓冲，再标出 FP32 fragment 的 score、maximum、denominator、accumulator。`T.gemm` 完成两次矩阵乘加，`T.reduce_max/sum` 与 `T.Parallel` 完成稳定更新。空行要在归一化之前显式保护，不能等 NaN 写回后再处理。

::::

::::{tab-item} Triton
:sync: triton

在 Triton 中先标出 `tl.dot` 的 FP32 accumulator，再标出 `tl.max/sum` 的 FP32 在线统计与 PV 前的概率类型转换。空行需要 safe maximum 和零分母分支；mask 既要约束分数，也要保护实际输入与页表读取。

::::
:::::

## 12.6 哪些 FA4 思想可以迁移

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

在线状态、softmax 融合、GQA 与 causal 裁剪都不依赖 TMEM。TileLang 主线在 shared/fragment 范围内组织 QK、归约与 PV，面向 SM120 编译 warp MMA。SM100 的 TMEM、`tcgen05`、专职 warp 与条件重缩放仍需要另外的架构和数值证明，不能由语言名称推断已经移植。

::::

::::{tab-item} Triton
:sync: triton

在线状态、两次矩阵乘法之间融合 softmax、GQA 映射以及 causal 的块级裁剪，都不依赖 TMEM。FA4 在 SM100 上还使用 TMEM、`tcgen05`、专职角色和条件重缩放等机制；这些有各自的资源与数值分析。本项目先采用清楚的在线归约，并按 SM120 可执行的两种 DSL 路径组织计算。不能因为算法名字相似，就把一种平台的指令吞吐率和性能结果当作另一种平台的证据。

::::
:::::

## 本机实验观察

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

{download}`attention.json <../results/tilelang/attention.json>` 实际测试 B=3、GQA Hq:Hkv=4:2、有效 KV 长度 **[79,53,0]**。prefill 的 Q=[3,17,4,D]，decode 的 Q=[3,4,D]；dense 容量为 80，paged 使用 page=16、块表 [3,5]。均走 pointer 路径；prefill 为右对齐 causal，decode 非 causal 且显式 split=2。

| dtype / D | prefill dense、paged 最大误差 | decode dense、paged 最大误差 |
|---|---:|---:|
| FP16 / 64 | 均为 0.00048828125 | 均为 0.00006103515625 |
| BF16 / 128 | 均为 0.00390625 | 均为 0 |

全部 8 条记录通过独立 blockwise FP32 参考（FP16 `rtol=atol=0.01`，BF16 为 0.03）。默认 `block_n=64`，长度 79 使末端查询实际跨越两个 KV 块，检验了在线状态的更新。同一逻辑上下文跨 dense/paged 都成立，包含尾部、GQA、空序列和 split 合并；这为在线归一化的实际实现提供了有限输入上的数值证据。BF16 decode 的零误差只是本次输出舍入后与参考相同，不能推广为精确算法。此文件只有数值参考，**没有 PyTorch 或原生分页性能基线**；完整性能状态见{doc}`15-results`。

::::

::::{tab-item} Triton
:sync: triton

**历史 Triton 实验（2026-11-02，原环境）**

本机 `results/attention.json` 使用三条长度 79、53、0 的序列、17 条查询与 GQA 4:2，同时运行稠密和打乱页表的分页版本。两种布局的 prefill 最大绝对误差均为 FP16 0.00048828125、BF16 0.00390625；空序列为零。相同逻辑上下文的两种存储由独立参考核对。

::::
:::::

## 练习

1. 用两个标量分数块手算在线更新，验证改变块顺序后实数结果不变；再解释浮点结果为何可能略有变化。
2. 写出 $L_b=5,S_q=8$ 时每行可见键数，标出空行。然后让 KV 容量变成 64，说明哪些答案应保持相同。
3. 设 $H_q=12,H_{kv}=3$，列出全部查询头对应的 KV 头。设计一组每个 KV 头常数不同的输入来检查映射。

本章对应上游 FlashAttention 章节的数学主线，执行架构的迁移范围见[来源与迁移表](appendix/sources.md)。
