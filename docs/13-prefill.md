(prefill)=
# 13. Prefill：查询分块、GQA 与分页 KV

prefill 一次处理多个查询位置，通常有足够多的查询块让 GPU 并行工作。主要机会是复用：一个 K/V 块被同一 program 中的多条查询消费，分数和概率停留在片上，而不是把整张矩阵存入显存。主要风险则是状态容量：每多保留一条查询，就多出一个长度为 $D$ 的 fp32 输出累计。本章把数学状态落实到公共接口与地址计算，强调稠密存储和分页存储必须共享同一个逻辑序列语义。

## 13.1 公共接口与输出所有权

调用形式为 `prefill(q, k, v, *, causal=True, scale=None, kv_lens=None, block_table=None, out=None, config=None)`。输入 Q 是 `[B, Sq, Hq, D]`，返回同形状、同类型的输出。Q、K、V 使用相同的 FP16 或 BF16；D 为 64、128 是性能重点，D 为 32 还承担小规模正确性检查。默认缩放是 $1/\sqrt D$。内核只读 KV，不更改缓存内容。

Q/K/V 和输出的最后一维必须连续，外层地址使用各自的实际 stride。输出布局还须通过按 stride 排序的非重叠检查：支持外层转置和间隔切片，拒绝展开的零 stride 及无法证明分离的特殊 `as_strided` 视图，防止多行并发写入同一地址。接口还拒绝需要梯度的正常 autograd 调用；这是推理前向算子，没有反向规则。`out` 若与任意输入共享底层存储会被拒绝，即使它只是同一存储上的另一个 view，也不能借此原地覆盖输入。

最容易推理的工作划分是：每个 program 拥有一个批次、一个查询头和一段查询行。它读取该查询头对应的 KV 头，沿有效 KV 长度逐块扫描，维护上一章的 $m,\ell,u$。独立 program 写互不重叠的输出片段，因此无须跨 program 原子加。较大的查询块增加 K/V 复用，也增大 fp32 输出状态和分数临时块；最佳尺寸必须由资源报告与测量决定。

GQA 的存储优势来自较少的 KV 头，不等于多个查询头在每一级缓存中都只会读取一次 KV。本项目的查询头调度、L2 行为与输入形状共同决定跨头复用。报告中应区分「逻辑 KV 容量减少」与「测得的外存字节减少」，不要把前者直接当作后者。

下面是 pointer 路径的完整在线主循环。二维 launch 的第一个轴负责查询块，第二个轴把批次与查询头压平。`maximum` 保存已经乘过 $\log_2e$ 的分数最大值，因此校正因子与概率使用 `exp2`。`denominator`、`maximum` 和 `accumulator` 是 fp32；为使用第二次矩阵乘法，概率在 `tl.dot` 前转成 Q 的低精度类型。这与全 fp32 参考有合理舍入差异，不能由「fp32 accumulator」推断整条计算链都使用 fp32。

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: _prefill_kernel
:linenos:
:caption: 真实 prefill 内核：查询分块、在线 softmax 与原生分页加载
```

公共包装负责检查元数据的形状、类型、设备与布局，并把真实 stride 传入内核。`config` 可以设置 `block_m`、`block_n`、`num_warps`、`num_stages` 与 `loader`；其中 stages 是编译器流水调度参数。默认使用 `loader='pointer'`。连续稠密 BSHD 输入还可以显式选择 `loader='tma'`：该版本使用真实 TensorDescriptor 加载，生成 `cp.async.bulk.tensor` 与 `mma.sync`，仍以寄存器保存 fp32 累计。它不使用 SM100 的 `tcgen05` 或 TMEM。

TMA 版本要求 D=64/128、输入连续、基地址与行距符合 16 字节对齐；分页输入会明确报错。描述符把 batch 与序列展平，边界块还必须显式屏蔽跨 batch 和有效长度之外的数据。否则，描述符的分配边界零填充不能代替逻辑序列的掩码，未使用的 KV 槽位中若存在 NaN，仍可能污染乘法。

```{literalinclude} ../src/fa5060/attention_tma.py
:language: python
:pyobject: _prefill_tma_kernel
:linenos:
:caption: TensorDescriptor 搬运与在线 softmax：保持相同的右对齐语义
```

用 `prefill(q, k, v, config={"loader": "tma", "block_m": 64, "block_n": 64, "num_stages": 2})` 选择该路径。运行 `pytest -q tests/test_attention_tma.py` 可核对 FP16/BF16、非对齐长度、空 KV、GQA 和 CUDA Graph 等边界；性能要另行测量，正确性或指令证据不能证明 TMA 一定快于 pointer。

```{literalinclude} ../src/fa5060/attention.py
:language: python
:pyobject: prefill
:linenos:
:caption: 公共 prefill 包装：校验、资源预检与 launch
```

## 13.2 稠密容量与有效长度

稠密 K/V 使用 `[B, capacity, Hkv, D]`。不提供 `kv_lens` 时，有效长度等于容量；提供时，第 $b$ 条序列只读前 $L_b$ 个 token。causal 的偏移使用 $L_b-S_q$，而不是 `capacity-Sq`。容量只是存储分配，不能改变模型的可见上下文。

下面的例子同时覆盖 GQA、变长、查询数多于有效长度，以及只读的 KV 语义。它使用本项目的独立 fp32 参考核对输出：

```python
import torch
from fa5060 import prefill
from fa5060.reference import attention_reference

torch.manual_seed(13)
q = torch.randn((2, 5, 4, 64), device="cuda", dtype=torch.float16)
k = torch.randn((2, 9, 2, 64), device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
lengths = torch.tensor([9, 3], device="cuda", dtype=torch.int32)
actual = prefill(q, k, v, kv_lens=lengths)
expected = attention_reference(q, k, v, causal=True, kv_lens=lengths)
torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
torch.testing.assert_close(actual[1, :2], torch.zeros_like(actual[1, :2]))
```

在输出为空的查询行中，结果仍为定义明确的零。调用者若传入 `out`，需要提供匹配的形状、类型、设备与可接受布局，且不能让输出覆盖仍会被读取的 Q/K/V。显式输出可把分配移出稳定计时区，但不会自动消除 JIT 编译或首次使用的开销。

## 13.3 分页只改变寻址，不改变注意力

分页 K/V 形状为 `[P, page, Hkv, D]`，其中 P 是物理页数，page 为 16、32 或 64。`block_table[b, logical_page]` 保存物理页编号；块表是 GPU int32 的二维张量，最后一维 stride 为 1，`kv_lens` 是 GPU 上连续的 int32 `[B]` 张量。对逻辑 token 位置 $j$，地址转换为

$$
g=\lfloor j/\text{page}\rfloor,\quad
r=j\bmod\text{page},\quad
p=\text{block\_table}[b,g].
$$

然后读取 `K[p, r, hkv, d]` 与 `V[p, r, hkv, d]`。逻辑页可以映射到任意合法物理页，物理页之间不必连续。最后一页超出 $L_b$ 的槽位不能贡献概率，空序列也不能因为计算了无效页号而越界读取。这里需要同时 mask 逻辑 token 与相应的页表访问。

长度与页号的数值合法性属于调用者合同：$0\le L_b\le\text{capacity}$，所有会访问的物理页号都在 `[0,P)` 内。公共 GPU 接口检查张量元数据，不把长度和页表拷回 CPU 逐项扫描，从而保留 CUDA Graph 可捕获性。输入 mask 负责保护有效长度之外的加载，不能把它当成修复非法活动页号的机制。

```{figure} static/kernels-paged-kv.svg
:alt: 两条逻辑序列通过块表映射到不连续物理 KV 页，最后一页只部分有效
:width: 95%

先确定逻辑位置是否有效，再通过页表生成物理地址；causal 仍使用逻辑位置。
```

下面用四个物理页承载两条序列，并刻意改变页顺序。待测内核直接读取页表与物理缓存，没有预先 gather 成稠密 KV。独立参考会在自身内部 gather 逻辑 KV 后执行 fp32 公式，这条校验路径不进入性能比较。

```python
import torch
from fa5060 import prefill
from fa5060.reference import attention_reference

torch.manual_seed(130)
q = torch.randn((2, 7, 8, 64), device="cuda", dtype=torch.float16)
k = torch.randn((4, 16, 2, 64), device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
lengths = torch.tensor([27, 18], device="cuda", dtype=torch.int32)
out = torch.empty_like(q)
prefill(q, k, v, block_table=table, kv_lens=lengths, out=out)
ref = attention_reference(q, k, v, causal=True,
                          block_table=table, kv_lens=lengths)
torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
```

## 13.4 causal 的块级裁剪

对查询块中最大的绝对查询位置 $a_{\max}$，所有 $j>a_{\max}$ 的 KV 块均不可见，可以跳过。对完全落在所有查询可见前缀内的块，可以不逐元素检查 causal 三角边界；只有交界块需要精确掩码。这个优化必须先保证右对齐偏移正确，且不同批次的有效长度独立处理。

把「跳过整块」与「把无效分数设为负无穷」混为一谈，会遗漏一个重要差别：前者减少加载和矩阵乘法，后者通常仍支付计算代价。性能报告若使用因果注意力，应说明有用 FLOP 的计算约定；直接套非因果 $4S_qLD$ 会高估有效吞吐。延迟始终是最直接、最少依赖计数约定的比较指标。

当前 pointer 实现设置 `stop=min(length, (block+1)*BM+length-SQ)`，并把负值截为零，从而跳过查询块完全不可见的未来 KV 块；循环内仍对各行应用精确 causal 条件。它没有另外分出完全可见前缀的无 causal mask 循环。阅读其他变体时，同样应区分这两项优化，而不是仅凭函数名包含 causal 就认定都已实现。

## 13.5 先确认语义，再谈调优

正确性矩阵至少要覆盖 FP16/BF16、D32/64/128、GQA 比例、非因果与右对齐因果、零长度、非整页长度，以及物理页打乱。对于页表错误，应让每一页或每个 KV 头携带可辨认的常数，避免随机数恰好掩盖索引错误。对于分母错误，则应同时观察输出误差与空行处理。

运行本项目统一实验与 smoke 基准：

```bash
python -m fa5060.experiments attention --output results/attention.json
python -m fa5060.benchmark --suite smoke --phase prefill --output results/prefill-smoke.json
```

分页性能比较只接受能原生读取同样页表语义的基线。若先把页缓存重排成稠密张量，该步骤既改变工作流，也可能把显著成本移出计时边界；它可以帮助语义调试，却不能作为原生分页 kernel 的公平性能对手。


## 本机实验观察

教学实验 `results/attention.json` 保留了 Q=[3,17,4,D]、有效 KV 长度 [79,53,0] 的 pointer 路径。稠密与分页都通过，FP16/BF16 最大误差分别为 0.00048828125 和 0.00390625。TMA 的独立 GPU 测试还检查了无效缓存中填 NaN 的情形；完整基准中的配置与延迟见结果章。

完整调优记录还提供了一个直接的反例：B=1、Sq=L=512、Hq=Hkv=32、D=64、FP16、dense，在公开候选搜索的一轮试测中得到下表。计时为图内完整算子，单位 μs；它用于解释配置选择，正式所选配置另测五轮。

| 加载/块尺寸 | stages | 单轮试测 μs |
|---|---:|---:|
| pointer，默认 32×64 | 2 | 56.177 |
| pointer，32×64 | 1 | 58.238 |
| pointer，64×64 | 2 | 53.002 |
| pointer，32×128 | 2 | 61.745 |
| pointer，64×128 | 2 | 55.038 |
| TMA，64×64 | 2 | 59.527 |

本项选择了 pointer 64×64；TMA 有真实指令与正确性证据，却没有在本项胜出。增加 N 块也没有单调改善延迟。原始候选记录位于 `results/full-tuned.json` 的首个 case，应结合资源与更多轮次继续研究，不能由一轮差值断言具体停顿原因。

## 练习

1. 对 page=16、有效长度 35，列出逻辑 token 15、16、34 的页号与页内位置。再用一个非单调块表写出物理页号。
2. 为 $S_q=8,L_b=19$ 的查询块画出可见区域，分别标出可整块跳过、完全可见以及需要逐元素 causal mask 的区域。
3. 在保证逻辑数据完全相同的前提下，重排物理页与块表，验证输出不变。说明仅重排块表却不移动数据为什么不是同一个测试。
