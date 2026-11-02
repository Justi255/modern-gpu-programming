(debugging)=
# 附录 C：按数据流定位错误

GPU 错误经常在真正出错的调用之后才被报告，因为 CUDA 执行默认异步。调试的第一步不是修改十个参数，而是建立一个小到可以理解的复现，区分编译失败、非法访存、数值错误与单纯变慢。对每个张量回答四个问题：逻辑形状是什么，物理地址如何得到，谁写入，谁在何时读取。

## C.1 先把失败固定下来

记录种子、dtype、输入 shape 与 stride、有效长度、页大小、块表以及配置。保存第一条错误而不只保存最后一层 Python traceback。对照[安装附录](install.md)生成环境 JSON，然后运行最小测试或 smoke：

```bash
python -m fa5060.probe --output results/environment.json
python -m pytest -m gpu -x
python -m fa5060.benchmark --suite smoke --phase all --output results/debug-smoke.json
```

测试失败时先保持输入不变。把 dtype、页大小、块尺寸和分段数同时改掉，也许会绕开问题，却无法知道哪项改变了执行路径。如果失败涉及较大的模型形状，应逐步缩小一个轴，同时保持触发条件，例如尾块、页边界或右对齐偏移。

## C.2 编译失败：从合法性开始

Triton 会根据 constexpr、dtype、目标设备和 stride 编译特化。错误可能来自不支持的 `tl.dot` 形状、过大共享内存、非法 layout conversion，或工具链根本不支持 SM120。先确认报错在编译阶段，随后核对目标与资源预算，不要把未编译成功的配置写成「运行很慢」。

本项目 GEMM 接受最后一维 stride 为 1 的二维输入，外层行 stride 可以不同于逻辑宽度。转置 view 往往不满足这个约束；这时公共接口抛出异常是契约检查，不是内核数值错误。attention 的布局条件同样以公共验证代码为准，不能假设任意 PyTorch view 都能零成本传入。

对于浮点计算，分清输入类型、乘法精度、累计精度与存储类型。fp32 accumulator 不保证输入乘法没有舍入，也不保证不同归约顺序逐位相同。应依据模型与输入范围定义误差合同，而不是看到第一个差异就把阈值放宽十倍。

## C.3 非法访存：沿地址链逐级检查

为便于定位，可在一次独立调试进程中启用同步报错：

```bash
CUDA_LAUNCH_BLOCKING=1 python -m fa5060.experiments attention --output results/attention-blocking.json
```

这会改变执行与性能，不应用于正式计时。若安装了 Compute Sanitizer，可进一步检查内存访问：

```bash
compute-sanitizer --tool memcheck python -m fa5060.experiments attention --output results/attention-memcheck.json
```

分页访问链是 `logical token -> logical page -> physical page -> page offset -> head -> feature`。逐项核对 int32 页号、表宽、有效长度与总物理页数。mask 必须保护实际执行的读取：只在最终分数里屏蔽无效 token，却已经提前加载越界块表，仍然会出错。

非法访存可能使 CUDA context 进入错误状态，后续无关操作继续报错。此时重新启动 Python 进程，再运行最小复现；反复清空 PyTorch cache 不能修复已损坏的上下文。

## C.4 数值错误：让输入具有可辨认结构

随机输入适合广泛覆盖，结构化输入适合解释错误。让每个 KV 头使用不同常数，可以检查 GQA 分组；让每个物理页使用不同常数，可以检查页表映射；令所有 logits 相等时，输出应为可见 value 的平均值，可以检查 causal 与有效长度。对于 $L_b<S_q$，显式验证前面的空行恰好为零。

| 观察到的错误 | 优先检查的依赖 |
|---|---|
| 只在最后一行、列或 K 块出错 | 输入 mask、输出 mask、尾部零填充 |
| 所有查询头以固定周期错位 | GQA 使用了取模而非连续分组 |
| 稠密正确，分页跨页后错误 | 页号除法、页内余数、块表 stride |
| 非因果正确，因果整体偏移 | 使用了容量或查询行号，漏掉 $L_b-S_q$ |
| 只有空行出现 NaN | $-\infty-(-\infty)$ 与零分母处理 |
| split=1 正确，更多 splits 出错 | 局部状态定义、指数基准与合并权重 |
| 持久化第一个块正确，后续错 | 累加器与地址是否在每个输出块重置 |

这张表是搜索顺序，不是证明。每个假设仍应由一个最小输入或一次地址推导确认。不要通过让待测内核与参考共享相同 gather 辅助函数来消除差异，那可能把同一个索引错误复制到两边。

## C.5 同步与缓冲区复用

对手工异步协议，把每条交接边列成「生产者、消费者、准备信号、复用条件」。上游 SM100 warp-specialized 内核常有显式 barrier phase，本项目常由 Triton 编译器生成等待与 staging。具体同步指令不同，但过早覆盖仍在使用的缓冲区会导致相同类型的数据竞争。

跨调用的 workspace 也需要所有权。两个流并发执行 decode 时，不能让它们共享同一个可写 scratch；图重放期间也不能让其他调用写入同一输出。单流下偶尔正确不代表并发安全。调试时先使用独立工作区固定问题，再考虑在外部调度中安全复用。

## C.6 正确但慢：一次只检验一个解释

先排除 JIT、首次分配、错误计时边界和后台 GPU 工作，再观察形状变化。短 K GEMM 可能付不起流水线启动成本，小批次 decode 可能缺少并行度，大查询块可能导致寄存器 spill。每个解释应对应一个可观察变化，例如修改 split 数后第一阶段并行度增加，但第二阶段开销也增加。

Profiler 只能提供证据，不能自动替你选择根因。保留未插桩时间，选择一个代表 case，针对性收集资源与停顿指标，再做单项修改。若修改只改善训练用形状而损害正式 suite，应该保留这个事实并重新考虑 dispatch，而不是删除不利 case。

提交问题时附上最小复现、环境、完整命令、预期结果与实际结果；涉及性能时附上计时边界和多轮数据。上游异步调试附录的固定来源见[来源表](sources.md)，其中 `tcgen05`、TMEM 与 warpgroup barrier 的具体协议用于比较理解，不作为 SM120 运行说明。

## C.7 本机实例：接口接受，不代表内核存在

本次安装的 FlashInfer 0.7.2.post1 提供了接受分离 Q/K/V 的 FMHAv2 接口。最初的 dense smoke 调用没有抛出不支持异常，却产生了错误输出；一个 FP16 case 的最大绝对误差超过 5，而同机原生分页路径正确。失败原始记录保存在仓库 `results/diagnostics/smoke-separate-layout-rejected.json`，不进入最终性能结论。

沿已安装源码检查 `flashinfer/jit/attention/fmha_v2/fmha_library.py` 的 `flash_valid` 可以看到：普通 D64/128 的 SM120 配置没有启用 `SEPARATE_Q_K_V` 布局；该布局的标准路径依赖另一组架构条件。Python 参数校验与底层可生成内核集合在这里没有形成完整保护。这个实例说明，既不能只检查 import 成功，也不能只检查 API 没有抛错；计时前必须核对数值。

最终适配器改用公开的 `BatchPrefillWithRaggedKVCacheWrapper(backend="fmha_v2")`。该包装内部把 K/V 用 `torch.stack` 打包为受支持的布局；打包仍在每次 `run()` 内，图与普通接口计时都包含它，元数据标为 `input_copies=true, packing_timed=true`。输入语义和数值阈值没有因此改变。分页适配器直接读取原生页表，不通过 gather 改成 dense。修正后再从同一组输入开始验证，而不是把原来的失败输出宣布为「误差较大但可用」。
