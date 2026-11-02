(ch-triton-layouts)=
# 08　Triton 布局与编译：逻辑张量如何分到 lane

Triton 源码让我们看到逻辑张量，却没有直接列出每个元素属于哪个寄存器。这是编译器的重要职责，也正是性能解释最容易断开的地方。本章把逻辑布局、存储布局与编译器布局连起来，学习如何检查一次 load、归约、转置或 dot 实际需要什么通信。

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

Triton 的 GPU 中间表示经常使用 blocked、slice、dot operand、shared 等编码表达这些选择。具体文本形式会随版本改变，读 IR 时应关注其含义，而不要把某次打印出的字段顺序当成稳定公共 API。

blocked 布局通常表达每 thread 的元素块、warp 内 thread 的排列、CTA 内 warp 的排列与维度顺序。slice 布局可以理解为从一个更高维分布中去掉某个维度后的结果，例如归约输出的所有权。dot operand 则记录矩阵乘加所需的 fragment 组织。共享编码描述的是存储安排，不能直接与寄存器所有权画等号。

## 8.3 逻辑操作不保证零成本

`tl.reshape`、`tl.trans`、广播和 reduction 首先表达数学语义，是否产生通信取决于输入和输出分布。一个 reshape 若只改变同一 thread 内元素的解释，可能不需要跨 thread 指令；若新消费者希望由另一组 lane 持有元素，就可能需要布局转换。

归约尤其能说明问题。若一行所有值恰好在同一个 thread 的寄存器中，行和只需 thread 内加法；若一行分散在多个 lane，需要 warp shuffle；若跨多个 warp，还可能经过共享内存与块内同步。相同的 `tl.sum(x,axis=1)`，因此可能对应完全不同的通信路径。

这里没有“越少转换越好”的绝对规则。一次转换如果使后续多次 MMA 使用正确高效布局，可能很值得。真正应该避免的是反复在两种布局之间来回转换，却没有足够计算摊薄代价。阅读 IR 时可以把每个转换标成“为哪个消费者服务”，找不到理由的重复转换才是优先调查对象。

## 8.4 跟踪真实转置内核

本项目的转置代码用二维指针 tile 连续读取逻辑行，再把同一组值写入转置位置：

```{literalinclude} ../src/fa5060/fundamentals.py
:language: python
:pyobject: _transpose_kernel
:linenos:
```

写回地址交换了行列角色。编译器可以选择让读取很好合并、写回跨步，也可以为两端之间生成一次所有权交换。仅凭 Python 代码中没有共享数组，不能断言“完全没用共享内存”；也不能反过来因为它是转置，就宣称编译器一定产生最优共享内存转置。

检查时先建立逻辑证据：给定 program id，`rows` 和 `cols` 的范围是什么？`values` 的 shape 是多少？然后看 IR 的 load 和 store 分布是否相同，有没有 `convert_layout` 一类操作，最后到 PTX 中查共享内存 load/store、barrier 或 shuffle。汇编文本回答的是“存在这条路径”，其运行代价仍需要 profiler 或受控计时。

## 8.5 获取编译产物

下面使用项目真实加法 kernel 的 `warmup` 接口取得编译对象。它需要 CUDA 环境并会触发编译，但不把 kernel 的运行时间当作结果；不要在 CUDA Graph capture 中做首次编译。

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
destination = Path("results/ir-add")
destination.mkdir(parents=True, exist_ok=True)
print("metadata:", compiled.metadata)
print("registers/thread:", getattr(compiled, "n_regs", "unavailable"))
for stage in ("ttir", "ttgir", "llir", "ptx"):
    if stage in compiled.asm:
        (destination / f"add.{stage}").write_text(compiled.asm[stage])
print("available stages:", sorted(compiled.asm))
```

`_vector_add_kernel` 是私有实现符号，这里专用于本仓库的学习和检查；用户计算应调用公开 wrapper。不同 Triton 版本可能改变编译对象细节，因此示例打印可用 stage，并对元数据使用兼容读取。这里没有伪造任何 PTX 输出，读者得到的是当前环境的实际产物。

可在输出文件中用以下方式搜索，再结合上下文阅读：

```bash
rg 'blocked|slice|convert_layout|shared|dot' results/ir-add/add.ttgir
rg 'ld.global|st.global|ld.shared|st.shared|shfl|bar.sync|mma' results/ir-add/add.ptx
```

加法不应需要 MMA；找不到 `mma` 是符合语义的。对 GEMM 才应重点找 MMA，对归约才应关注 shuffle 等通信。搜索到字符串不是完整验证：某些名字可能被版本替换，一条指令也可能只在特定分支执行。

## 8.6 用配置实验验证布局解释

固定 N 与 dtype，只改 `BLOCK` 或 `num_warps`。每组配置先做正确性检查，再保存资源与计时。向量加法的逻辑字节数相同，但 program 数、每 thread 元素数、指令向量化与调度成本都可能改变。

对二维 kernel，最好分别改变行列方向。比如把 softmax 宽度从 1024 改为 1025，逻辑元素只增加一个，但取下一个 2 的幂可能让内部 tile 扩大一倍。这种资源阶跃是布局与编译期 shape 带来的现象。更宽时，项目改用在线归一化两遍扫描，避免无界扩大寄存器 tile；代价则变为再次读取输入。

对 `tl.dot`，改变 warp 数还会影响 warp 之间如何拆分输出 tile；改变 K tile 则影响 A/B staging、指令数和共享内存。需要同时保存三种证据：正确性、编译资源、稳定时间。只看到 IR “更整齐”或者寄存器更少，不能直接宣布性能更好。

## 8.7 为什么本书先不手写所有布局

底层布局 API 适合需要精确控制通信和硬件特性的场景，但也会把更多版本与架构约束交给程序员。本书先用稳定的 `triton.language` 表达算法，建立可运行基线，再通过中间表示理解编译结果。这样后续尝试显式布局时，有清楚的对照，而不是同时改变算法、语言和调度。

Triton 的公开语言与教程见 [Triton documentation](https://triton-lang.org/main/index.html)。本书保留的上游 TIRx 材料属于另一套语言与布局接口，数学模型可以比较，API 不应直接替换；对应关系见{doc}`appendix/sources`。下一章把本章的 load、dot、store 布局连成第一个完整 GEMM。

## 练习

1. 证明 $i=\ell+32r$ 和 $i=4\ell+r$ 都是 128 个位置的双射。固定一个寄存器槽后，二者对应的全局地址间隔分别是多少？
2. 编译加法的 `BLOCK=256,512,1024`，记录 program 数与实际寄存器数。哪些变化可以从平均元素数解释，哪些需要读 IR？
3. 比较 softmax 宽度 1024 与 1025 的逻辑 tile、资源和时间；分别写出由 padding 与通信产生的候选解释。
4. 找到一次真实布局转换，说明其生产者、消费者与可能的通信范围。再尝试消除它，要求同时保存正确性和性能结果，不得只以源码行数判断改进。
