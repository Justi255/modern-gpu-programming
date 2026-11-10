(sources)=
# 附录 D：来源、平台迁移与延伸阅读

本书参考了 MLC AI 的 *Modern GPU Programming for MLSys*，但不是把原文逐段翻译后替换设备名称。上游围绕 SM100/B200 和 TIRx 展开，本项目围绕 RTX 5060 Laptop 的 SM120、TileLang/Triton 对照、prefill/decode 与分页 KV 展开。正文、示例与 SVG 采用本项目自己的组织与表达；本表说明概念来源、保留的内容和必须改变的平台假设。

上游引用统一固定为提交 [`a78b4e6e9f2be5697705c174d49bbe4974e0c4a5`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5)。引用该提交是为了让读者复查相同文本，并不保证其中链接到的外部包与论文也随之冻结。转载代码或资源时应继续保留其原有许可与署名；本书的重新推导不改变上游文件自身的许可条件。

## D.1 全部上游核心章节的对应关系

| 上游固定版本章节 | 本书对应位置 | 保留与迁移原则 |
|---|---|---|
| [GPU Background](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_background/index.md) | [环境](../01-environment.md)、[执行模型](../02-execution.md) | 保留层次结构与执行概念，设备规模由本机探针提供。 |
| [What Makes a Kernel Fast](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_performance/index.md) | [性能模型](../03-performance.md)、[结果判据](../15-results.md) | 保留 roofline、复用与重叠分析；不移用 B200 峰值与实测图。 |
| [Data Layout](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_data_layout/index.md) | [内存布局](../04-memory-layout.md) | 区分逻辑坐标、地址与线程所有权，示例提供 TileLang 与 Triton 对照。 |
| [Layout Generations](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_layout_generations/index.md) | [内存布局](../04-memory-layout.md)、[双后端布局](../08-triton-layouts.md) | 保留跨硬件布局演化的理由，不把 TMEM layout 当成本机寄存器布局。 |
| [TMA](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_tma/index.md) | [异步与同步](../06-async.md)、[GEMM 流水线](../10-gemm-pipeline.md) | 保留异步搬运与完成依赖；本机生成的加载形式需实际核对。 |
| [Tensor Cores](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_tensor_cores/index.md) | [Tensor Core](../05-tensor-cores.md) | 保留 tile、精度和布局匹配，SM120 使用其支持的 warp 级路径。 |
| [Tensor Memory](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_tmem/index.md) | [Tensor Core](../05-tensor-cores.md)、[高级 GEMM](../11-gemm-advanced.md) | 作为 SM100 架构比较；不在 SM120 上执行 TMEM 分配或 `tcgen05`。 |
| [Async Barriers](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_async_barriers/index.md) | [异步与同步](../06-async.md)、[调试](debugging.md) | 保留完成与复用两类依赖，具体 barrier 协议按目标编译链重建。 |
| [Cluster Launch Control](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_clc/index.md) | [执行模型](../02-execution.md)、[高级 GEMM](../11-gemm-advanced.md) | 说明动态调度的动机；本项目持久化使用静态 stride 遍历，不伪称实现 CLC。 |
| [Introduction to TIRx](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_intro_tirx/index.md) | [双后端入门](../07-triton.md)、[基础 GEMM](../09-gemm-basics.md) | 将 scope/layout/dispatch 的思考方式映射到 两种 DSL 的 CTA、逻辑 tile 与编译特化。 |
| [TIRx Layout API](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_tirx_layout_api/index.md) | [双后端布局](../08-triton-layouts.md)、本附录 D.2 | 保留映射与 swizzle 概念；TIRx 的 `TileLayout` 不是 Triton API。 |
| [Building a Tiled GEMM](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_gemm_basics/index.md) | [基础 GEMM](../09-gemm-basics.md) | 保留空间分块与 K 归约，右操作数改为 `[K,N]`，累计位于 SM120 寄存器状态。 |
| [Pipelining GEMM with TMA](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_gemm_async/index.md) | [GEMM 流水线](../10-gemm-pipeline.md) | 保留生产消费和环形缓冲分析，两种 DSL 的 stages 均需生成代码与实测支持。 |
| [Scaling GEMM with Warp Specialization and Clusters](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_gemm_advanced/index.md) | [高级 GEMM](../11-gemm-advanced.md) | 实现可执行的持久化教学调度；专职 warp、2-CTA 与多 consumer 仅说明架构前提。 |
| [Flash Attention 4](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/chapter_flash_attention/index.md) | [注意力数学](../12-attention.md)、[prefill](../13-prefill.md)、[decode](../14-decode.md) | 重推在线 softmax 与 GQA；增加右对齐 causal、分页 KV、split-KV 和空序列语义。 |

## D.2 TIRx 布局、语言参考与编译附录

这部分是跨 DSL 阅读指南，其中生成 TIR/TIRx 只用于解释 TileLang 编译器，不提供手写第三后端。TIRx 把 scope、layout 与 tile primitive 的 dispatch 显式化；Triton 则让程序员写逻辑块与值依赖，再通过中间表示、layout conversion 和目标后端生成指令。本书还用 TileLang 显式声明 shared/fragment、copy 与 GEMM，再由编译器推导线程布局。三套资料都需要验证逻辑坐标如何成为物理地址、值由哪些线程持有，以及目标指令是否接受该布局，但 API 不能按名称一一替换，TileLang 也不是把 `tl.*` 改成 `T.*`。

| 上游参考页（同一固定提交） | 本书中如何使用 |
|---|---|
| [Language Reference 总入口](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/language_reference/index.rst) | 与第 7–8 章对照术语，不把它当作 `triton.language` 的函数表。 |
| [Parser Utilities](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/language_reference/cuda/parser_utils.rst) | 对照源代码解析、编译期构造与运行时表达式的区别。 |
| [Data Types and Expressions](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/language_reference/cuda/data_types.rst) | 对照类型、索引与精度，结合第 5、7、12 章。 |
| [Buffers and Memory](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/language_reference/cuda/buffers.rst) | 理解形状、layout、scope 和地址的独立性；本书对照 TileLang buffer/stride 与 Triton 指针表达式。 |
| [Control Flow](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/language_reference/cuda/control_flow.rst) | 对照常量展开、运行时循环与流水调度，第 9–11 章提供本机例子。 |
| [Threads and Synchronization](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/language_reference/cuda/threads_sync.rst) | 保留同步作用域和交接条件的检查方法，具体 SM100 barrier 操作不迁移为本机代码。 |
| [Compiler Internals 总入口](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/arch/index.rst) | 对照前端、设备分离、目标代码生成的整体职责。 |
| [TIRx Lowering Pipeline](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/tirx_guide/arch/lowering_pipeline.rst) | 了解 TIRx/TVM 的变换与合法化；TileLang tile lowering 与 Triton TTIR/TTGIR/PTX 均按固定版本检查。 |
| [中文树：IR 的组织方式](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/zh/tirx_guide/arch/ir_representation.rst) | 对照函数、buffer region、tile primitive 与副作用的表示。 |
| [中文树：Tile Primitive 与 Layout Lowering](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/zh/tirx_guide/arch/tile_primitive_layout_lowering.rst) | 理解 layout 如何约束 dispatch；其中 TMEM 描述仍属于上游平台。 |
| [Benchmarking GPU Kernels](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/appendix/benchmarking_gpu_kernels.md) | 本书[测量附录](benchmarking.md)保留计时与诊断分离，重新规定分页与多阶段边界。 |
| [Debugging Warp-Specialized Kernels](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/appendix/debugging_warp_specialized.md) | 本书[调试附录](debugging.md)保留数据交接与症状缩小方法，不照搬 SM100 协议。 |
| [Appendix Overview](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5/appendix/index.md) | 对照上游参考资料的组织范围，避免把本书未执行的工具当作依赖。 |

阅读 lowering 时可以追踪同一条运算：源代码中的 tile 乘法如何变成具有物理布局的操作，何时插入跨 lane 或共享内存转换，哪一步选择目标矩阵指令，最后如何得到指针与同步。对本项目，一个更实用的检查是比较 `tiled` 与 `pipelined` 的生成代码：逻辑乘法没有变，加载、staging 和等待的安排是否真的改变？这个问题比记住所有 pass 名称更能指导优化。

### TileLang 固定版本与原创迁移范围

本书 TileLang 来源固定为 **v0.1.15，提交 [`a35f8ddf45eba16c21211ec8822d56ce5363036f`](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f)**，调用器依赖 TVM-FFI 0.1.12。核心参考包括该提交中的[示例目录](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f/examples)与[文档源码](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f/docs)。保持原文件的许可与署名，项目自己的 TileLang kernel 按本书共同合同重新组织，不能据此声称上游原例已包含以下全部能力。

向量覆盖、两遍宽行 softmax、保留外层 stride 的转置/GEMM、三种 GEMM 调度、右对齐变长 causal、MHA/GQA/MQA、原生页表寻址、只读 decode 缓存、空 KV 零输出、FP32 split 状态与显式 workspace，都是需要单独审查和验证的本项目行为。与上游实例相比，接口、缓存布局、目标、数值契约和计时边界均可能不同。引用相同公式不代表可以搬用上游的硬件时间。

## D.3 本项目新增内容与主资料

右对齐变长 causal、稠密/分页 KV 的公共合同、空 KV 输出零、decode 读缓存语义、显式 workspace、逐 case 最快正确兼容基线以及分 phase 的几何平均/最差门槛，是本项目针对教学与本机实验制定的内容。它们不应被误读成上游原文已经实现或验证的功能。

进一步查证硬件与软件接口时，优先阅读以下主资料。在线文档会更新，正式实验应同时记录访问日期及实际安装版本：

| 资料 | 用途 |
|---|---|
| [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus) | 确认设备能力编号，避免把 SM100 与 SM120 混同。 |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 执行、内存一致性、同步与架构说明。 |
| [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | 核对具体指令的目标架构、形状和同步条件。 |
| [TileLang v0.1.15](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f) | Tile primitive、shared/fragment、布局推导、软件流水与调用器的固定源码。 |
| [Triton 文档](https://triton-lang.org/main/index.html) | 语言操作、教程、编译器与调试接口。 |
| [PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 核对 mask、GQA 与后端选择，尤其非方形 causal 的对齐差异。 |
| [FlashInfer 文档](https://docs.flashinfer.ai/) | 原生 paged prefill/decode 接口、workspace 与目标兼容性。 |
| [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention) | 查实际 provider 支持范围；包名称不能代替 SM120 执行证据。 |
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) | 在线最大值与归一化统计的更新依据，第 12–14 章重新推导并用于分块与分段合并。 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 不物化完整注意力矩阵的 IO-aware 分块思想与精确注意力语义。 |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 工作划分、并行度和非矩阵运算代价，辅助理解本项目为何仍需测量调度。 |
| [FlashAttention-4 论文](https://arxiv.org/abs/2603.05451) | 理解 SM100 方向的流水线和数值优化，性能结论按论文硬件与设置解读。 |

比较外部实现时，还需固定 provider 的精确版本或提交。本文固定的上游教材提交只保证阅读来源可定位，并没有替外部基线做版本冻结或正确性验证。最终可信的结论必须回到本机环境文件、实际输入、逐 case 检查和完整计时记录。
