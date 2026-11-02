(results)=
# 15. 从可运行内核到可审查的性能结论

一套完整教材不仅要给出能运行的内核，还要说明什么证据足以支持「更快」。性能目标、工作负载和失败处理必须在看见结果以前确定，否则很容易在反复试验后只展示有利形状。本章给出本项目的预注册判据、结果文件的产生方式和解释方法。实现完成、正确性通过与达到性能目标是三个独立结论，任何一个都不能代替另外两个。

## 15.1 预先约定比较对象

每个 case 都有自己的语义：prefill 或 decode、稠密或分页、因果模式、有效长度、页大小、数据类型、头数与 D。比较只在这些条件一致时成立。prefill 的右对齐 causal 不能换成左上三角；decode 的缓存包含当前 token；分页必须让 provider 原生读取同样的逻辑页映射。

候选基线包含能够在 SM120 上正确执行的 FA4 实现、FlashInfer 的 SM120 路径与 PyTorch 路径。某个名字出现在 Python 环境里，并不意味着它支持当前 GPU、dtype、页布局或 causal 对齐。每个 provider 都要经过兼容性判断与正确性检查。对于每个 case，以所有正确、兼容基线中最快的稳定时间作为分母参照；不能挑选其中较慢的一家来放大优势。

定义 $t_i^{\rm ours}$ 为本项目在 case i 的延迟，$t_i^{\rm best}$ 为该 case 最快合格基线的延迟，加速比为

$$
s_i=\frac{t_i^{\rm best}}{t_i^{\rm ours}}.
$$

$s_i>1$ 表示本项目更快。若某个必要原生基线缺失、编译失败或不具备可审查的运行证据，应把覆盖状态写清楚。特别是缺少必要原生分页基线时，不能借稠密 gather 路径补位，也不能宣称已完成 SOTA 对照。

## 15.2 按 phase 独立判断

prefill 与 decode 分别统计，不把二者混在一个平均数中。每个 phase 的目标是几何平均加速比至少 1.10 倍，同时最差 case 至少 0.95 倍：

$$
G=\exp\left(\frac1n\sum_{i=1}^{n}\log s_i\right)\ge1.10,
\qquad W=\min_i s_i\ge0.95.
$$

几何平均适合乘法比例，也避免用绝对延迟较长的 case 自动主导结论；最差值约束则防止平均值掩盖严重退化。1.10 和 0.95 是事先约定的验收门槛，不是对当前实现已经达到的描述。若没有足够的合格 case、必要基线不完整或任一正确性失败，该 phase 不能被标成达标。

调优与最终评估应分开。可以用预先规定的候选配置和代表性 case 选择块尺寸、stages 与 split 数，再冻结策略运行全套。若在全套结果里逐 case 人工选择最优配置，必须把这种选择策略公开，并让它成为可重复执行的 dispatcher，而不能只把离线挑选的最好数字写进表格。

## 15.3 可重建的结果链

在仓库根目录执行以下命令，输出保存到独立文件。探针记录设备与软件环境，教学实验记录分解过程，smoke 先检查路径，full 才用于完整覆盖。第一次全套运行可以不带 `--tune`，以便保留默认实现的基准；带调优的实验应另存文件。

```bash
python -m fa5060.probe --output results/environment.json
python -m fa5060.experiments basics --output results/basics.json
python -m fa5060.experiments gemm --output results/gemm.json
python -m fa5060.experiments attention --output results/attention.json
python -m fa5060.benchmark --suite smoke --phase all --output results/smoke.json
python -m fa5060.benchmark --suite full --phase all --rounds 5 --output results/full.json
python -m fa5060.benchmark --suite full --phase all --rounds 5 --tune --output results/full-tuned.json
python scripts/report_results.py --input results/full-tuned.json
```

运行结果与源码版本应一同归档。一个 JSON 文件应能回答：输入是什么、哪些 provider 尝试执行、为何跳过、正确性是否通过、计时边界是什么、每轮测得什么、聚合如何得到。若只保存最终加速比，就无法判断它来自真正的内核改善、错误的基线选择，还是一次偶然的频率波动。

正式报告采用 `schema_version` 标识格式，顶层保留 `environment`、`suite`、`rounds`、`cases` 与 `summary`。逐 case 的 `providers` 记录各 provider 状态和时间，`fastest_baseline` 指明参照，`speedup_vs_fastest` 给出比例；phase 汇总保留 `geometric_mean_speedup`、`worst_speedup` 与 `target_met`。这些字段需要连同覆盖状态一起读，单独出现 `target_met` 不能替代必要原生基线的有效性检查。

离线选中的配置可以通过同一个公共接口重放。下面以报告中的一条 decode 为例，输入生成与计时区外的参考检查保持一致；这段代码只验证配置复用，不替代五轮基准：

```python
import json
import torch
from fa5060 import decode, decode_workspace
from fa5060.reference import attention_reference
from fa5060.workloads import AttentionCase, make_inputs

report = json.load(open("results/full-tuned.json"))
record = next(c for c in report["cases"] if c["phase"] == "decode")
case = AttentionCase.from_dict(record)
inputs = make_inputs(case)
choice = record["providers"]["ours"]["selected_config"]
workspace = decode_workspace(inputs.q, inputs.k, block_table=inputs.block_table,
                             config=choice["config"], num_splits=choice["num_splits"])
actual = decode(inputs.q, inputs.k, inputs.v, kv_lens=inputs.kv_lens,
                block_table=inputs.block_table, workspace=workspace,
                config=choice["config"], num_splits=choice["num_splits"])
expected = attention_reference(inputs.q, inputs.k, inputs.v, causal=False,
                               kv_lens=inputs.kv_lens, block_table=inputs.block_table)
tolerance = 0.003 if case.dtype == "fp16" else 0.02
torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
```

## 15.4 笔记本 GPU 的实验纪律

笔记本功耗与热状态会随时间变化。同一张 GPU 在电池、外接电源、不同风扇模式和长时间负载后，可能给出不同的持续频率。实验不需要假装这些变化不存在；需要做的是记录条件、预热、重复测量，并交替比较候选与基线，降低时间顺序造成的偏差。

CUDA event 测量流上的 GPU 工作，主机墙钟测量一次 API 调用的端到端成本。前者适合比较已编译、已分配的稳定内核，后者可揭示 Python 包装、分配与提交开销。两者都可以报告，但不能把本项目的 GPU-only 时间与基线的端到端时间相除。CUDA Graph 比较也必须保持捕获、预热和内存复用方式一致。

重复轮次不能只保留最快值。应保留每轮统计及聚合规则，检查结果分散程度。如果 1% 的差异伴随更大的运行波动，适当结论是证据不足，而不是根据某一轮顺序挑一个胜者。Profiler 用于诊断，公开的延迟仍来自未插桩的计时运行。

## 15.5 阅读逐 case 表，再阅读总指标

一张可审查表至少保留 phase、布局、B、Sq、KV 长度、Hq/Hkv、D、dtype、页大小、自定义时间、最快合格基线与加速比。对变长 case，要保留长度分布，而不只写最大长度。错误、OOM、缺失 provider 和不支持的形状都要显式列出，不能从分母中悄悄删除。

先看最差 case，判断退化是否集中在短序列、小批次、BF16 或分页；再看几何平均，判断整体收益。随后把性能现象连回前面几章的模型：prefill 可能被寄存器状态限制，decode 可能缺少并行度，页表访问可能降低加载规整性。模型解释应由测量支持，不能用「带宽受限」四个字替代 profiler 与形状敏感性证据。

## 15.6 本仓库的实测记录

本节由 `scripts/report_results.py` 从完整结果文件生成，收录设备信息、覆盖状态、逐 case 数据来源以及两个 phase 的判据结论。脚本会重新计算汇总并检查冻结项数，避免手工抄表改变比较集合。

<!-- GENERATED_RESULTS_START: root integrates actual probe/benchmark evidence here. -->

本次发布在实际 RTX 5060 Laptop GPU 上完成冻结的 **312 项**测试：144 项 prefill、168 项 decode；每项保留五轮测量。软件为 Python 3.13、PyTorch 2.12.1（CUDA 13.0）、Triton 3.7.1、FlashInfer 0.7.2.post1、FA4 4.0.0b37。设备为 26 SM、32 MiB L2、每 CTA opt-in 99 KiB 共享内存，CUDA 可用总显存约 7682 MiB。

**本版没有达到约定的 SOTA 性能目标。** 下表描述所有 case 对逐项最快正确兼容基线的延迟比；正确性失败或基线执行错误还会独立阻止达标结论。不能把局部胜出概括为全面领先。

| Phase | 完成/冻结项数 | 几何平均加速 | 最差加速 | 回退超过 5% | 覆盖与正确性完整 | 达标 |
|---|---:|---:|---:|---:|---|---|
| prefill | 144/144 | 0.920× | 0.688× | 89 | 否 | 否 |
| decode | 168/168 | 1.130× | 0.748× | 25 | 是 | 否 |

这里报告的是**离线逐 case 固定网格选配置、再单独五轮测量**的结果。所有候选、失败和最终 `selected_config` 均保留在 JSON。公开 `prefill`/`decode` 默认配置不自动加载这张表，因此本表不是默认调用的性能保证。读者可按相同 shape/dtype/layout 显式传入 `config` 与 decode 的 `num_splits`；换设备或软件后应重新验证。配置搜索使用同一冻结集合，不能视作独立保留集上的泛化证据。

### 数据、计时与复现边界

- {download}`完整原始结果 <../results/full-tuned.json>`：逐项输入、七个 provider、全部候选、误差与多轮数据。
- {download}`完整 case 表 <../results/cases.csv>`、{download}`所有回退项 <../results/regressions.csv>`、{download}`provider 状态与接口时间 <../results/providers.csv>`。
- {download}`冻结清单与源码 SHA256 <../results/manifest-full.json>`、{download}`离线所选配置 <../results/selected-configs.json>`、{download}`每秒 GPU 遥测 <../results/full-telemetry.csv>`。
- {download}`原生 provider 实现与已安装源码哈希 <../results/provider-sources.json>`：FA4 的实际路径为 `FlashAttentionForwardSm120`；Torch 分别强制 Flash/cuDNN/Efficient，FlashInfer 分别请求 FA2/FMHAv2。
- {download}`8 项 smoke <../results/smoke.json>` 用于路径检查，不承担完整矩阵结论。

主指标把完整算子多次捕获进同一个 CUDA Graph，每轮重放一次、除以图内调用数，再取五轮中位数；decode 包含 split 与 merge。规划、JIT、参考和校验不计入，单独的 eager event/墙钟数值保留接口分配与提交成本。输入与页表固定，provider 顺序按公开种子打乱。FlashInfer dense FMHAv2 公开包装内部的 KV 打包计入两种计时，不能与预先打包的私有入口混淆。

未锁定频率或风扇模式；外接电源状态未自动采集，也未关闭桌面应用或隔离设备。因此这是公开条件下的本机观测，不能声称实验室独占测量。遥测提供功耗、时钟、温度和显存序列，功率上限若为 N/A 则保持缺失，不推测数值。

全程取得 930 个每秒采样：功耗 8.03–79.17 W，SM 时钟 487–2775 MHz，温度 48–77 °C，已用显存 1851–6794 MiB。范围包含预热、JIT、参考与测量阶段，不能把峰值功耗直接视作任一内核的平均功耗。

### 最差项与后续实验

| Phase | B | L | Hkv | D | 类型/缓存 | 本实现 μs | 最快基线 | 基线 μs | 加速 |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| prefill | 1 | 8192 | 8 | 128 | fp16/paged | 30207.457 | flashinfer_fmha_v2 | 20770.369 | 0.688× |
| prefill | 1 | 512 | 1 | 128 | fp16/dense | 122.628 | flashinfer_fmha_v2 | 84.509 | 0.689× |
| prefill | 1 | 2048 | 8 | 128 | fp16/paged | 1827.829 | flashinfer_fmha_v2 | 1331.924 | 0.729× |
| prefill | 4 | 8192 | 1 | 128 | fp16/paged | 115319.839 | flashinfer_fa2 | 84178.947 | 0.730× |
| prefill | 4 | 2048 | 32 | 128 | bf16/paged | 8008.160 | flashinfer_fa2 | 5847.680 | 0.730× |
| decode | 1 | 8192 | 8 | 128 | fp16/paged | 51.231 | flashinfer_fa2 | 38.302 | 0.748× |
| decode | 16 | 512 | 1 | 64 | fp16/dense | 7.768 | torch_cudnn | 6.139 | 0.790× |
| decode | 1 | 8192 | 8 | 128 | bf16/paged | 44.463 | flashinfer_fa2 | 35.942 | 0.808× |
| decode | 1 | 32768 | 8 | 64 | bf16/dense | 218.091 | torch_cudnn | 185.320 | 0.850× |
| decode | 1 | 32768 | 8 | 128 | bf16/paged | 426.824 | flashinfer_fa2 | 371.951 | 0.871× |

先用这些固定 case 检查因果裁剪、块尺寸和 split 数的敏感性，再收集寄存器、spill、内存流量和占用证据。本机没有安装 Nsight Systems/Compute，当前没有足够 profiler 证据把所有回退归因于某一种硬件瓶颈。prefill 仍按查询头独立扫描 KV，decode 的分段与合并也有额外工作；它们是可检验的优化方向，不能替代诊断结论。

### 基线状态

| Provider | 正确并计时 | 明确不支持 | 数值失败 | 执行错误 |
|---|---:|---:|---:|---:|
| fa4 | 156 | 156 | 0 | 0 |
| flashinfer_fa2 | 312 | 0 | 0 | 0 |
| flashinfer_fmha_v2 | 308 | 0 | 4 | 0 |
| ours | 312 | 0 | 0 | 0 |
| torch_cudnn | 156 | 156 | 0 | 0 |
| torch_efficient | 52 | 260 | 0 | 0 |
| torch_flash | 156 | 156 | 0 | 0 |

共 4 条 provider 失败记录，全部留在原始文件中；失败者不参与最快合法基线选择，也不被静默删除。这些数值失败指未满足本书冻结的误差合同，原始 RMS 误差也同时保留，便于读者按自己的数值预算审查。

其中的例子：

- `prefill-b1-l512-hq32-hkv32-d128-fp16-dense` / `flashinfer_fmha_v2`：CorrectnessError: reference mismatch: max_abs=0.0046386719, max_relative_atol_floor=0.42661032, normalized_rms=0.00088240052, atol=0.003, rtol=0.003, rms_tolerance=0.02
- `prefill-b1-l8192-hq32-hkv32-d128-fp16-dense` / `flashinfer_fmha_v2`：CorrectnessError: reference mismatch: max_abs=0.0035095215, max_relative_atol_floor=0.4571279, normalized_rms=0.0014981168, atol=0.003, rtol=0.003, rms_tolerance=0.02
- `prefill-b4-l2048-hq32-hkv32-d128-fp16-dense` / `flashinfer_fmha_v2`：CorrectnessError: reference mismatch: max_abs=0.00390625, max_relative_atol_floor=0.59064227, normalized_rms=0.0010871076, atol=0.003, rtol=0.003, rms_tolerance=0.02
- `prefill-b4-l2048-hq32-hkv1-d128-fp16-paged16` / `flashinfer_fmha_v2`：CorrectnessError: reference mismatch: max_abs=0.0068359375, max_relative_atol_floor=0.54709119, normalized_rms=0.0010918253, atol=0.003, rtol=0.003, rms_tolerance=0.02

初次 dense FMHAv2 分离布局错误及固定工作区不足的中间结果保存在 `results/diagnostics/`，不合并进本表。扩大工作区后重新完整执行；没有缩小测试形状或放宽 FP16 0.003 / BF16 0.02 的逐元素绝对与相对阈值，另检查归一化 RMS 误差。

### 教学分解实验与验证

基础、GEMM 和小注意力分别保存于 {download}`basics.json <../results/basics.json>`、{download}`gemm.json <../results/gemm.json>` 和 {download}`attention.json <../results/attention.json>`。这些普通提交 event 数据包含主机供给空隙，用于逐章学习，不与上表图内延迟直接相除。三组共 26 条记录均先通过数值检查，涵盖尾块、外层 stride、三种 GEMM 调度、变长 GQA、空 KV 与随机页表。

完整测试与文档构建命令及结果另见 {download}`验证记录 <../results/verification.json>`。CPU CI 不把跳过 GPU 测试视作 GPU 通过；真正的设备验证在本机执行。

<!-- GENERATED_RESULTS_END -->

## 练习

1. 构造三个加速比，使算术平均大于 1 而几何平均小于 1。解释为什么单看算术平均会误导。
2. 假设所有已运行 case 都较快，但一个必要分页 provider 不可用，写一段准确的结果说明，区分局部事实与整体结论。
3. 选择最差 case，只改变一个调优参数重新测量。先写出可证伪的解释，再保留改动前后全部轮次，不只记录最好时间。
