# TileLang SM120 实际编译证据

本目录对应 2026-11-08 完成的 [manifest.json](manifest.json)，在 RTX 5060 Laptop GPU（SM120、26 SM）上经公开 API 实际调用 7 次，导出 9 个 kernel：三种 GEMM 各 1 个、两种 prefill 各 1 个、两种 split decode 各含 main 和 merge。所有 `launch_completed=true`，但 `correctness_validated=false`；这次运行只收集编译证据，没有数值断言或性能采样。教学数值结果另见 [basics.json](../basics.json)、[gemm.json](../gemm.json)、[attention.json](../attention.json)，完整性能状态见[发布说明](../README.md)。

环境为 TileLang 0.1.15、TVM-FFI 0.1.12、PyTorch 2.12.1（CUDA build 13.0）、NVCC/PTXAS/cuobjdump 13.1.115；实际目标均为 `sm_120`、执行后端 `tvm_ffi`。manifest 记录源码和产物 SHA-256、软件包版本、pass 配置、工具路径、调用输入和输出 stride。所列 TileLang 参考提交用于来源归属，不声称已安装 wheel 来自该 VCS 提交。

| 公开调用 | 实际输入（FP16） | 配置 |
|---|---|---|
| `matmul/tiled` | A=[512,1024]，B=[1024,512] | tile=32×64×32，4 warps，1 stage |
| `matmul/pipelined` | 同上 | tile=32×64×32，4 warps，3 stages |
| `matmul/persistent` | 同上 | tile=32×64×32，4 warps，3 stages |
| `prefill/pointer` | Q=[1,128,8,64]，K/V=[1,256,2,64]，有效 KV=[193] | 32×64，4 warps，2 stages，右对齐 causal |
| `prefill/TMA` | 同上 | 64×64，4 warps，2 stages，右对齐 causal |
| `decode/vector` | Q=[2,8,64]，K/V=[2,1024,2,64]，有效 KV=[1024,193] | pointer，block_n=64，4 warps，1 stage，split=4 |
| `decode/grouped` | 同上 | pointer，block_n=64，4 warps，2 stages，split=4 |

各 kernel 目录保存 `original_tir.py`（factory 原始程序）、`kernel.cu`（真实生成 CUDA）、`kernel.ptx`、`cuobjdump-resources.txt`。以下为本次文本证据的具体观察：

| kernel（链接到资源原文） | REG/thread | 静态 SHARED B | PTX `mma.sync` | PTX TMA load |
|---|---:|---:|---|---|
| [gemm_tiled/main](gemm_tiled/main/cuobjdump-resources.txt) | 120 | 1024 | 有 | 无 |
| [gemm_pipelined/main](gemm_pipelined/main/cuobjdump-resources.txt) | 78 | 1024 | 有 | 无 |
| [gemm_persistent/main](gemm_persistent/main/cuobjdump-resources.txt) | 112 | 1024 | 有 | 无 |
| [prefill_pointer/main](prefill_pointer/main/cuobjdump-resources.txt) | 93 | 1024 | 有 | 无 |
| [prefill_tma/main](prefill_tma/main/cuobjdump-resources.txt) | 157 | 2048 | 有 | 有 |
| [decode_vector_split/main](decode_vector_split/main/cuobjdump-resources.txt) | 167 | 1024 | 无 | 无 |
| [decode_vector_split/merge](decode_vector_split/merge/cuobjdump-resources.txt) | 21 | 1024 | 无 | 无 |
| [decode_grouped_split/main](decode_grouped_split/main/cuobjdump-resources.txt) | 71 | 1024 | 有 | 无 |
| [decode_grouped_split/merge](decode_grouped_split/merge/cuobjdump-resources.txt) | 21 | 1024 | 无 | 无 |

9 份 PTX 均未出现 `tcgen05`。这支持当前矩阵路径使用 warp `mma.sync`，以及选定 TMA prefill 确实生成 TMA load；vector decode 和 merge 不含 MMA。它不证明所有 shape 都采用同样指令，也不证明 TMA、流水线或 grouped decode 更快。

资源报告来自**使用实际生成的 CUDA 源码、相同 TileLang NVCC 选项和 SM120 目标重新编译的临时 CUBIN**，再执行 `cuobjdump --dump-resource-usage`；并非直接提取已加载的 FFI 模块。临时 CUBIN 退出后删除，保留其 SHA-256 和命令作为来源记录，发布目录只有文本与 JSON。`n_regs`、`n_spills` 等 Python 属性未被冒充为测得的 CUDA 资源。

**表中 SHARED 是静态共享分配，不含 launch 时申请的动态共享内存。** 例如生成 CUDA 中的 `extern __shared__ ... buf_dyn_shmem[]` 不计入该静态数字；1024 B 不能被解释为完整 GEMM 的总 SMEM。REG、静态 SHARED 以及这里同时出现的 STACK=0、LOCAL=0 也不足以独立推断实际占用率、流水重叠程度或性能瓶颈。本目录没有 Nsight 指标，不构成 profiler 证据或性能达标证明。不同 prefill 的 tile_m 还从 32 改为 64，不能把资源差异全部归因于 loader。

重建入口为仓库的 `scripts/export_tilelang.py`，使用版本锁定的私有编译接口；升级 TileLang/TVM-FFI 后需重新核对接口、哈希、目标与全部导出结果。与历史 Triton 或其他环境比较时，必须另外匹配输入合同、数值检查和计时边界，不能用本目录的编译观察替代测量。
