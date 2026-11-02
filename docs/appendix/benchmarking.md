(benchmarking)=
# 附录 B：计时、对照与性能诊断

测量回答「多快」，诊断回答「为什么」。把两者混在一起，容易因为 profiler 的重放、主机同步或编译缓存而得到误导结论。本项目先用统一基准产生未插桩的时间，再为一个具体问题使用 profiler。公开结果的计时边界、正确性标准与基线选择规则见[结果章](../15-results.md)。

## B.1 把计时边界写成清单

稳定 GPU 内核时间通常包含完整 kernel 链，包括 split-KV 的两个阶段；不包含输入创建、首次 JIT、参考计算、正确性断言和固定 workspace 分配。若 provider 有计划阶段或 cache 初始化，应在预热阶段完成，并对所有 provider 使用同类边界。对于 paged attention，页表查询是内核工作的一部分，不能预先 gather 成稠密布局后仍称原生分页时间。

端到端单次调用则可能有意包含输出分配和 Python 包装。这个指标更接近某些应用路径，但应单独标记。一个多内核函数也不能只挑其中最快或最主要的一个 kernel 计时：decode 的局部状态与最终合并共同构成一次正确调用。

## B.2 一个明确边界的 CUDA event 示例

下面的代码先完成预热，再把多次相同调用放入同一条 CUDA 流。输出与 scratch 均在计时区外分配，因此结果是 steady-state GPU 调用成本；这里没有声称这是统一 benchmark CLI 的内部实现。

```python
import torch
from fa5060 import decode, decode_workspace

q = torch.randn((1, 8, 64), device="cuda", dtype=torch.float16)
k = torch.randn((1, 1024, 2, 64), device="cuda", dtype=torch.float16)
v = torch.randn_like(k)
lengths = torch.tensor([1024], device="cuda", dtype=torch.int32)
out = torch.empty_like(q)
scratch = decode_workspace(q, k, num_splits=4)

def call():
    return decode(q, k, v, kv_lens=lengths, out=out,
                  workspace=scratch, num_splits=4)

for _ in range(10):
    call()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    call()
stop.record()
stop.synchronize()
print("mean GPU milliseconds:", start.elapsed_time(stop) / 100)
```

这个简单例子仍可能受到主机提交速度影响：如果 Python 不能持续供给很短的 kernel，流上的事件间隔可能包含 GPU 等待提交的空隙。CUDA Graph 可以减少这类开销，但捕获前必须完成编译并分配所有缓冲。图时间与普通提交时间应分别报告，不能把两种模式相除。

本项目正式基准把多次完整调用捕获进**同一个图**，每轮用 event 测量一次图重放，再除以图内调用数。不能只捕获一次调用，然后在 Python 循环里重放几千次：这样仍可能把提交空隙混入短内核时间。报告的 `graph_operator_calls_per_replay` 保留实际批量，`gpu_us_rounds` 保留五轮结果；普通接口的 event 和墙钟时间另存为 `eager_us_rounds` 与 `eager_wall_us_rounds`。教学用的 `experiments` 则保留普通提交路径，不能把其微秒数解释成隔离后的纯内核延迟。

不要在每次调用后立即 `torch.cuda.synchronize()` 来「保证精确」，那会把多次运行变成强制串行的主机往返。正确性检查可以在测量前后同步，计时循环内部则应保持事先定义的执行方式。

## B.3 多轮与汇总

统一命令为：

```bash
python -m fa5060.probe --output results/environment.json
python -m fa5060.benchmark --suite full --phase all --rounds 5 --output results/full.json
```

保留每轮与每 case 数据，使用明确聚合规则，不静默丢弃慢轮。对同一个输入，最快合格基线来自实际通过正确性与兼容性检查的 provider；缺失 provider 不是无穷慢。按 phase 计算几何平均和最差比值之前，应先确认 case 集合相同，且没有错误被剔除。

GPU 独占、外接电源、固定风扇模式和足够预热能减少波动，但也要记录这些条件。长期持续负载可能进入不同温度与功耗区间，故应交替运行候选与基线或随机化执行顺序，避免把热状态误当成算法变化。若差异小于观察到的波动，应增加证据，而不是挑一个有利样本。

## B.4 先看应用时间线

Nsight Systems 适合检查 CUDA 调用、GPU kernel、内存复制和主机空隙。安装工具并确认 `nsys --version` 可用以后，可对 smoke 工作负载采样：

```bash
nsys profile --trace=cuda,nvtx --output=results/attention-smoke python -m fa5060.benchmark --suite smoke --phase all --output results/profile-smoke.json
```

查看某次 attention 调用是否包含预期的 kernel 链，是否出现意外 D2H 同步、反复分配，或者 kernel 之间有明显主机空隙。不要只按「累计耗时最高」判断单次调用最慢；一个很小但被调用很多次的 kernel 也可能累计占比很高。工具版本与平台可能影响可用 trace 选项，真实诊断以本机帮助与生成报告为准。

## B.5 再看单个内核的资源

Nsight Compute 提供寄存器、共享内存、occupancy、访存与执行管线指标。先收集少量基础指标，避免一次申请所有指标造成大量重放：

```bash
ncu --set basic --target-processes all --export results/decode-basic python -m fa5060.benchmark --suite smoke --phase decode --output results/ncu-smoke.json
```

此命令可能采集多种 provider 与多个 case。先在时间线中找出关注的 kernel，再依据本机 `ncu --help` 增加 kernel 过滤和 launch 选择。若系统关闭了性能计数器权限，工具会报告权限错误；应按本机管理员政策处理，不能用一个空报告继续得出结论。

资源数字的解释需要联系代码。occupancy 低不一定慢：较大的块可能提高复用并保持 Tensor Core 忙碌。occupancy 高也不一定快：许多 warp 可以同时等待同一带宽瓶颈。shared memory 增加可能来自多 stage，也可能来自布局转换。需要提出一个具体假设，再改变一个参数验证，而不是按指标名称机械调优。

## B.6 看编译产物而不预设结论

Triton 编译对象提供 IR、PTX 等产物的访问能力，但具体属性与缓存结构随版本变化。分析时保存精确 Torch/Triton 版本，检查是否生成了预期矩阵指令、异步拷贝、共享布局转换以及寄存器 spill。不要把一个版本的缓存目录结构写成跨版本稳定 API。

上游还介绍 Proton、IKET 和 TIRx 内核内部阶段标注。这些工具适合其对应编译链，并不自动成为本项目 Triton 内核的依赖。可迁移的是「先确定时间边界，再定位主要阶段，再读资源与指令」的方法；具体插桩接口应查相应工具版本。固定来源见[来源表](sources.md)。
