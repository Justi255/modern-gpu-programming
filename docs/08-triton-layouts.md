(ch-triton-layouts)=
# 08　TileLang 与 Triton 布局与编译：逻辑张量如何分到 lane

<span id="triton-lane"></span>

本章同时讲解 **TileLang 与 Triton 的布局与编译证据**。TileLang 侧追踪 shared/fragment、布局推导和 CUDA 代码生成，Triton 侧追踪 TTIR/TTGIR/PTX。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 在源码中显式区分 shared、fragment 与 global buffer，但逻辑 fragment 仍未直接列出每个值由哪个 lane/寄存器持有。自动布局推导、tile primitive lowering 和线程映射补齐这部分信息。本章把源码 scope 与最终地址、所有权、通信连起来。

::::

::::{tab-item} Triton
:sync: triton

Triton 源码让我们看到逻辑张量，却没有直接列出每个元素属于哪个寄存器。这是编译器的重要职责，也正是性能解释最容易断开的地方。本章把逻辑布局、存储布局与编译器布局连起来，学习如何检查一次 load、归约、转置或 dot 实际需要什么通信。

::::
:::::

## 8.1 一维向量也有分布布局

假设一个 warp 负责 128 个元素，每个 lane 平均持有四个值。一种教学用分布是

$$
i(\ell,r)=\ell+32r,\quad0\le\ell<32,\;0\le r<4.
$$

固定寄存器槽 $r$，32 个 lane 的下标相邻，很适合共同读取连续地址。另一种双射是 $i(\ell,r)=4\ell+r$，每个 lane 持有连续四个值；它是否高效取决于是否能生成合适的向量化访问。两者覆盖完全相同的 128 个逻辑元素，却有不同的逐指令访问模式。

```{figure} static/foundations-ownership.svg
:name: fig-ownership
:alt: 四个示意 lane 和四个寄存器槽分别采用按槽连续与按 lane 连续的两种元素所有权

图中缩小为四个示意 lane，真实 warp 有 32 个 lane。这是分布映射示例，不是宣称编译器固定采用某一种布局。
```

现在把同样思路扩展到二维 tile：布局把逻辑坐标映射到 warp、lane 和寄存器槽。生产者最适合的布局未必适合消费者，例如连续 load、行归约与 MMA fragment 通常各有偏好。编译器需要在这些偏好之间选择并插入转换。

## 8.2 三类布局要分开读

| 布局 | 回答的问题 | 常见约束 |
| --- | --- | --- |
| 全局张量的 shape/stride | $(i,j)$ 在哪个全局地址 | 连续维、对齐、跨行步长 |
| thread/register 分布 | $(i,j)$ 属于哪个 warp/lane/槽 | 合并访问、归约通信、MMA fragment |
| 共享内存编码 | 临时值位于哪个地址/bank | padding、swizzle、异步传输和矩阵 load |

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的布局推导依据访问模式、tile primitive 与目标能力为 fragment/shared buffer 选择物理映射。`T.annotate_layout` 可以提供约束；手写约束必须同时满足 copy、归约与 GEMM 消费者。推导出的 TVM/TIR 表达式与 pass 名称会随版本改变，不能把一次打印当成稳定 API。

::::

::::{tab-item} Triton
:sync: triton

Triton 的 GPU 中间表示经常使用 blocked、slice、dot operand、shared 等编码表达这些选择。具体文本形式会随版本改变，读 IR 时应关注其含义，而不要把某次打印出的字段顺序当成稳定公共 API。

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的 fragment 布局回答逻辑坐标由哪个 thread 的哪个局部槽持有；shared 布局回答逻辑坐标怎样映射到共享地址，可能包含 swizzle。`T.alloc_fragment` 与 `T.alloc_shared` 的 shape 只是逻辑区域声明，不是完整的 thread/layout 规格。读 lowered IR 时应分别追踪两类映射与转换。

::::

::::{tab-item} Triton
:sync: triton

blocked 布局通常表达每 thread 的元素块、warp 内 thread 的排列、CTA 内 warp 的排列与维度顺序。slice 布局可以理解为从一个更高维分布中去掉某个维度后的结果，例如归约输出的所有权。dot operand 则记录矩阵乘加所需的 fragment 组织。共享编码描述的是存储安排，不能直接与寄存器所有权画等号。

::::
:::::

## 8.3 逻辑操作不保证零成本

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的 buffer region、索引重排与 `T.copy` 先规定语义，再由布局和 lowering 决定通信。把一个 fragment 换一种下标解释可能只是寄存器重命名；从不同 thread 的所有权移交给 GEMM 或转置消费者，则可能产生 shuffle、共享中转与同步。

::::

::::{tab-item} Triton
:sync: triton

`tl.reshape`、`tl.trans`、广播和 reduction 首先表达数学语义，是否产生通信取决于输入和输出分布。一个 reshape 若只改变同一 thread 内元素的解释，可能不需要跨 thread 指令；若新消费者希望由另一组 lane 持有元素，就可能需要布局转换。

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

归约尤其能说明问题。`T.reduce_sum` 的某一行若集中在一个 thread，只需局部加法；若跨 lane 则需要 warp 通信，跨 warp 则可能经过共享内存和 CTA 同步。因此相同的归约 shape 仍可能对应不同代价，应同时检查 fragment 所有权与生成通信。

::::

::::{tab-item} Triton
:sync: triton

归约尤其能说明问题。若一行所有值恰好在同一个 thread 的寄存器中，行和只需 thread 内加法；若一行分散在多个 lane，需要 warp shuffle；若跨多个 warp，还可能经过共享内存与块内同步。相同的 `tl.sum(x,axis=1)`，因此可能对应完全不同的通信路径。

::::
:::::

这里没有“越少转换越好”的绝对规则。一次转换如果使后续多次 MMA 使用正确高效布局，可能很值得。真正应该避免的是反复在两种布局之间来回转换，却没有足够计算摊薄代价。阅读 IR 时可以把每个转换标成“为哪个消费者服务”，找不到理由的重复转换才是优先调查对象。

## 8.4 跟踪真实转置内核

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 转置 factory 显式声明共享 tile，在输入区域和转置后的输出区域之间搬运同一组元素：

::::

::::{tab-item} Triton
:sync: triton

本项目的转置代码用二维指针 tile 连续读取逻辑行，再把同一组值写入转置位置：

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```{literalinclude} ../src/fa5060/tilelang/fundamentals.py
:language: python
:pyobject: _transpose_kernel
:linenos:
```

::::

::::{tab-item} Triton
:sync: triton

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _transpose_kernel
:linenos:
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 源码中的 shared tile 使输入和输出可采用不同的 thread 映射。是否生成了高效合并访问、怎样安排共享布局和同步，仍需根据 lowered IR 与设备源码检查；一个显式二维 shared 数组不是性能保证。

::::

::::{tab-item} Triton
:sync: triton

写回地址交换了行列角色。编译器可以选择让读取很好合并、写回跨步，也可以为两端之间生成一次所有权交换。仅凭 Python 代码中没有共享数组，不能断言“完全没用共享内存”；也不能反过来因为它是转置，就宣称编译器一定产生最优共享内存转置。

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

先根据 `T.Kernel` 块坐标和 `T.Parallel` 迭代范围证明每个元素的来源与去向，再沿 shared buffer 和 fragment 追踪生成的 thread 下标、同步与 load/store。设备源码、PTX 与 profiler 是不同层次的证据，不应把其中一个替代另一个。

::::

::::{tab-item} Triton
:sync: triton

检查时先建立逻辑证据：给定 program id，`rows` 和 `cols` 的范围是什么？`values` 的 shape 是多少？然后看 IR 的 load 和 store 分布是否相同，有没有 `convert_layout` 一类操作，最后到 PTX 中查共享内存 load/store、barrier 或 shuffle。汇编文本回答的是“存在这条路径”，其运行代价仍需要 profiler 或受控计时。

::::
:::::

## 8.5 获取编译产物

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

下面从项目真实加法 factory 取得 JITKernel 并保存生成设备源码。它会触发编译；普通用户应调用公开 wrapper。不要在 CUDA Graph capture 内做首次构造或编译。

::::

::::{tab-item} Triton
:sync: triton

下面使用项目真实加法 kernel 的 `warmup` 接口取得编译对象。它需要 CUDA 环境并会触发编译，但不把 kernel 的运行时间当作结果；不要在 CUDA Graph capture 中做首次编译。

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
from pathlib import Path
import tilelang
from fa5060.tilelang.fundamentals import _vector_add_kernel

# 私有工厂仅用于本仓库的编译检查；普通计算调用 vector_add。
kernel = _vector_add_kernel(1003, "float32", 256)
destination = Path("results/local/tilelang/ir-add")
destination.mkdir(parents=True, exist_ok=True)
source = kernel.get_kernel_source()
(destination / "add.cu").write_text(source)
(destination / "add.prim.py").write_text(kernel.prim_func.script())
# 磁盘缓存命中时 kernel.artifact 可能为 None，重新 lower 获取 IR。
target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_120"})
with target:
    artifact = tilelang.lower(kernel.prim_func, target=target,
                              enable_host_codegen=False,
                              enable_device_compile=False)
(destination / "add.device.py").write_text(artifact.device_mod.script())
print("TileLang:", tilelang.__version__)
print("saved generated CUDA source:", destination / "add.cu")
```

::::

::::{tab-item} Triton
:sync: triton

```python
from pathlib import Path
import torch
import triton
from fa5060.fundamentals import _vector_add_kernel

n, block = 1003, 256
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)
z = torch.empty_like(x)
compiled = _vector_add_kernel.warmup(
    x, y, z, n, BLOCK=block, num_warps=4,
    grid=(triton.cdiv(n, block),),
)
destination = Path("results/local/triton/ir-add")
destination.mkdir(parents=True, exist_ok=True)
print("metadata:", compiled.metadata)
print("registers/thread:", getattr(compiled, "n_regs", "unavailable"))
for stage in ("ttir", "ttgir", "llir", "ptx"):
    if stage in compiled.asm:
        (destination / f"add.{stage}").write_text(compiled.asm[stage])
print("available stages:", sorted(compiled.asm))
```

::::
:::::

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

`_vector_add_kernel` 是仓库私有 factory，供本节编译检查使用；普通计算调用公开 `vector_add`。TileLang 0.1.15 的 JITKernel 可以导出生成设备源码；上例额外重新 lower 保存 device TIR，以避免磁盘缓存命中时 artifact 缺失。此版本 CUDA 路线不能依赖 `n_regs`/`n_spills` 属性（它们可能为 None），资源实值应从 ptxas、driver 或 Nsight 获取。只展示本机实际返回的产物，不把生成的 CUDA 源码冒称为 PTX。

::::

::::{tab-item} Triton
:sync: triton

`_vector_add_kernel` 是私有实现符号，这里专用于本仓库的学习和检查；用户计算应调用公开 wrapper。不同 Triton 版本可能改变编译对象细节，因此示例打印可用 stage，并对元数据使用兼容读取。这里没有伪造任何 PTX 输出，读者得到的是当前环境的实际产物。

::::
:::::

可在输出文件中用以下方式搜索，再结合上下文阅读：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
rg 'shared|__syncthreads|mma|cp_async|async' results/local/tilelang/ir-add/add.cu
```

::::

::::{tab-item} Triton
:sync: triton

```bash
rg 'blocked|slice|convert_layout|shared|dot' results/local/triton/ir-add/add.ttgir
rg 'ld.global|st.global|ld.shared|st.shared|shfl|bar.sync|mma' results/local/triton/ir-add/add.ptx
```

::::
:::::

加法不应需要 MMA；找不到 `mma` 是符合语义的。对 GEMM 才应重点找 MMA，对归约才应关注 shuffle 等通信。搜索到字符串不是完整验证：某些名字可能被版本替换，一条指令也可能只在特定分支执行。

### 批量导出 compiler evidence

单个加法足以学习导出接口；理解矩阵计算与注意力还需要保存对应的实际内核及完整配置。编译产物用于解释指令与资源，不构成正确性或性能验证。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

仓库提供 `scripts/export_tilelang.py`：临时 hook `fundamentals._compile` 与 `attention._compile`，由公开包装构造真实输入和配置，再捕获它们实际编译、调用的 kernel。它不硬编码巨大的私有 factory 参数列表，也不会用另写的简化 kernel 替代目标程序。

```bash
# 帮助只用 Python 标准库，不导入 Torch、TileLang 或 TVM-FFI。
python3 -S scripts/export_tilelang.py --help

source .venv-repro/bin/activate
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
python scripts/export_tilelang.py --output-dir results/local/tilelang/compiler
```

默认导出七种公开调用：`tiled`、`pipelined`、`persistent` GEMM，pointer/TMA prefill，以及 vector/grouped 的四段 decode。两个 decode 均包含实际 merge，因此通常得到九个 kernel。可用 `--cases` 选择子集，`--dtype bf16` 改为 BF16；每次必须选择新的或空的输出目录。

```bash
python scripts/export_tilelang.py --cases prefill_pointer decode_grouped_split --dtype bf16 --output-dir results/local/tilelang/compiler-bf16-subset
```

每个 kernel 保存 `kernel.cu`、`original_tir.py`、`kernel.ptx` 与 `cuobjdump-resources.txt`。`manifest.json` 记录 SM120、库版本、环境探针、实际 shape/stride/dtype、配置、源码 SHA256、各文件 SHA256、编译顺序与工具版本。原始 TIR 是编译边界处 `PrimFunc.script()` 的文本，不是本书新增的手写 TIRx 后端。

PTX 用该版本的私有 `_get_ptx()` 从实际生成 CUDA 编译，错误直接抛出；公开 `export_ptx()` 会吞掉异常，因此脚本不用它。资源文本来自同一 CUDA 源码、相同 TileLang NVCC 选项、SM120 目标重编译的临时 CUBIN，再运行 `cuobjdump --dump-resource-usage`，**不是直接提取已加载 FFI 模块**。CUBIN 不进入发布目录，退出临时目录后删除；保留的哈希用于核对这次资源报告对应的二进制。

此脚本锁定 TileLang **0.1.15** 与 TVM-FFI **0.1.12** 的接口，并记录相关编译器源码哈希。升级版本、找不到 CUDA 工具、目标不符、编译或 launch 失败均以非零状态结束；已有成功步骤可以保留，但 manifest 会写明失败，不能当完整证据。CUDA 路线的 `n_regs`/`n_spills` 属性不会被填成测量值。脚本每个公开算子执行一次，不做数值断言或性能计时，必须另外运行正确性矩阵和正式基准。

::::

::::{tab-item} Triton
:sync: triton

Triton 路线继续使用本节前面的 `warmup` 与 `compiled.asm` 保存真实 TTIR、TTGIR、LLVM IR 和 PTX，同时记录对应的 shape、dtype、stride、launch 配置与版本。对 GEMM 或注意力应取得其实际编译对象，再检查 MMA、搬运与同步；不能用加法的资源数字替代它们。

`export_tilelang.py` 对应 TileLang 的工厂与调用器；Triton 的编译对象和资源接口仍按本节示例读取。比较两端时共享的是输入合同、原始源码哈希与证据种类，编译器对象的属性不要求同名。

::::
:::::

### 沿同一算子追踪 lowering 编译链

```{figure} static/dual-compiler.svg
:name: fig-dual-compiler
:alt: 两条编译链共享输入合同和数学问题，分别经 TileLang tile primitive lowering 与 Triton GPU IR，最终生成 SM120 设备代码

源码 API 不同，证据链相同：数学语义 → 布局与同步 → 目标指令 → 数值与计时。
```

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

Python factory 先构造 `T.prim_func`；编译器做访问区域分析、布局推导、tile primitive lowering、线程映射、软件流水和 CUDA 代码生成，调用器通过 TVM-FFI 接收张量。生成的 TIR/TIRx 可用于解释地址、buffer 与同步，**它不是本书要求读者手写的第三后端**。调用器的可捕获性还需在显式输出、预热与 workspace 合同下实测。

::::

::::{tab-item} Triton
:sync: triton

`@triton.jit` 接收参数并做特化，前端构造 TTIR，GPU lowering 选择分布布局、通信与流水，再经 LLVM/NVPTX 生成 PTX/cubin。常见观察点为 `ttir`、`ttgir`、`llir`、`ptx`；阶段是否暴露取决于版本，打印本机实际可见键后再保存。

::::
:::::

## 8.6 用配置实验验证布局解释

固定 N 与 dtype，只改 `BLOCK` 或 `num_warps`。每组配置先做正确性检查，再保存资源与计时。向量加法的逻辑字节数相同，但 program 数、每 thread 元素数、指令向量化与调度成本都可能改变。

对二维 kernel，最好分别改变行列方向。比如把 softmax 宽度从 1024 改为 1025，逻辑元素只增加一个，但取下一个 2 的幂可能让内部 tile 扩大一倍。这种资源阶跃是布局与编译期 shape 带来的现象。更宽时，项目改用在线归一化两遍扫描，避免无界扩大寄存器 tile；代价则变为再次读取输入。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

对 `T.gemm`，thread 数、输出 tile 和 K tile 会共同影响 MMA fragment、输入 staging 与共享布局。即使寄存器数减少，也可能因为布局转换或并行度下降而变慢。保存正确性、编译资源、稳定时间三类证据后再给结论。

::::

::::{tab-item} Triton
:sync: triton

对 `tl.dot`，改变 warp 数还会影响 warp 之间如何拆分输出 tile；改变 K tile 则影响 A/B staging、指令数和共享内存。需要同时保存三种证据：正确性、编译资源、稳定时间。只看到 IR “更整齐”或者寄存器更少，不能直接宣布性能更好。

::::
:::::

## 8.7 为什么本书先不手写所有布局

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

显式布局适合控制已经定位的通信与硬件约束，但也带来架构和版本责任。本书先用 `T.Parallel`、shared/fragment、`T.gemm` 与自动推导建立正确基线，再读 lowering；只有明确某次转换为何存在，才尝试 `T.annotate_layout`，避免同时改变算法与物理映射。

::::

::::{tab-item} Triton
:sync: triton

底层布局 API 适合需要精确控制通信和硬件特性的场景，但也会把更多版本与架构约束交给程序员。本书先用稳定的 `triton.language` 表达算法，建立可运行基线，再通过中间表示理解编译结果。这样后续尝试显式布局时，有清楚的对照，而不是同时改变算法、语言和调度。

::::
:::::


:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的固定版本教程与 API 见[TileLang v0.1.15](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f)。生成的 TIR/TIRx 是编译器表示，读它是为了解释 lowering；本书不要求手写第三套 TIRx 后端。与原教材的差异见{doc}`appendix/sources`，下一章把 copy、GEMM 和 store 组成完整乘法。

::::

::::{tab-item} Triton
:sync: triton

Triton 的公开语言与教程见 [Triton documentation](https://triton-lang.org/main/index.html)。本书保留的上游 TIRx 材料属于另一套语言与布局接口，数学模型可以比较，API 不应直接替换；对应关系见{doc}`appendix/sources`。下一章把本章的 load、dot、store 布局连成第一个完整 GEMM。

::::
:::::

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

本机归约实验包含 17×781 和 128×1024 两个外层跨步输入，`results/basics.json` 中最大绝对误差分别为 1.79×10⁻⁷ 与 2.38×10⁻⁷。781 迫使最后一个逻辑归约块处理尾部；同时检查概率归一化与数值参考，比只测试 2 的幂宽度更有解释力。

::::
:::::

## 练习

1. 证明 $i=\ell+32r$ 和 $i=4\ell+r$ 都是 128 个位置的双射。固定一个寄存器槽后，二者对应的全局地址间隔分别是多少？
2. 编译加法的 `BLOCK=256,512,1024`，记录 program 数与实际寄存器数。哪些变化可以从平均元素数解释，哪些需要读 IR？
3. 比较 softmax 宽度 1024 与 1025 的逻辑 tile、资源和时间；分别写出由 padding 与通信产生的候选解释。
4. 找到一次真实布局转换，说明其生产者、消费者与可能的通信范围。再尝试消除它，要求同时保存正确性和性能结果，不得只以源码行数判断改进。
