(book-home)=
# 面向 RTX5060 的现代 GPU编程：Triton与TileLang对照实践

这本书从一台真实的 RTX 5060 Laptop GPU 出发，用 Triton 与 TileLang 对照实现向量加法、归约、转置、矩阵乘法和注意力。学习终点是能说明一个 kernel 为什么正确、数据为什么按这种方式移动，以及一次性能改进究竟改善了哪一个瓶颈。代码、实验命令和测量方法与文字放在同一个仓库，读者可以逐层修改并验证自己的解释。

本书的目标机器是 **SM120、26 个 SM、约 8 GB 显存的 RTX 5060 Laptop GPU**。这是一份具体设备的观测记录，不代表所有 Blackwell GPU 的配置。特别是，数据中心 Blackwell 的 `tcgen05`、Tensor Memory、双 CTA MMA 与相关调度技巧，不能直接套在这台机器上。本书的 FP16/BF16 矩阵乘法从 warp 级 `mma.sync` 路径出发，累加器保存在寄存器中。

读者需要 Python、PyTorch、矩阵乘法与基本 CUDA 的 thread/block、内存层次概念；本书把这些基础迁移到两种 tile DSL，PTX 与编译细节从需要证据的地方展开。前八章建立执行、访存、数值与布局模型，后七章把这些模型落实到 GEMM、prefill 和 decode。只想先跑通实验的读者可以从{doc}`01-environment`开始，按命令生成自己的结果，再回到原理章节解释它们。

首次打开页面默认选择 **TileLang**。点击任一后端标签会同步当前页的全部代码与解释，并在跨页、刷新和后续浏览中记住选择。URL 的 `?kernel-backend=tilelang` 或 `?kernel-backend=triton` 可指定阅读路线。公共数学公式、硬件事实与独立参考实现只讲一次；后端对应的代码、导入、命令和编译说明放在标签页里。

网页默认与 CLI 默认相互独立：网页先展示 TileLang，CLI 为兼容旧脚本仍默认 `--backend triton`。本书的新命令均显式指定后端，结果目录也分开。

<span id="rtx-5060-gpu-triton-flashattention"></span>

## 怎么读，怎么验证

每章遵循“建立模型 → 推导代价 → 阅读或运行代码 → 检查边界”的顺序。书中的算术例子用于说明模型；设备计时只有由脚本生成并保存的结果才构成测量证据。文中没有把显卡宣传峰值当作已测吞吐，也没有把某个 shape 上的胜出当作全面领先。

基础实验通过以下入口运行，完整安装步骤见{doc}`appendix/install`：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m fa5060.probe --output results/local/tilelang/environment.json
python -m fa5060.experiments basics --backend tilelang --output results/local/tilelang/basics.json
python -m fa5060.experiments gemm --backend tilelang --output results/local/tilelang/gemm.json
python -m fa5060.experiments attention --backend tilelang --output results/local/tilelang/attention.json
python -m fa5060.benchmark --backend tilelang --suite smoke --phase all --output results/local/tilelang/smoke.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m fa5060.probe --output results/local/triton/environment.json
python -m fa5060.experiments basics --backend triton --output results/local/triton/basics.json
python -m fa5060.experiments gemm --backend triton --output results/local/triton/gemm.json
python -m fa5060.experiments attention --backend triton --output results/local/triton/attention.json
python -m fa5060.benchmark --backend triton --suite smoke --phase all --output results/local/triton/smoke.json
```

::::
:::::

运行 GPU 实验需要支持本设备的 PyTorch、所选 DSL 与驱动。推荐用 `requirements-tilelang.txt` 在干净 `.venv-repro` 中安装并运行两种后端，TileLang 与 TVM-FFI 固定为 0.1.15 / 0.1.12。本次实际采集的 `.venv-tilelang` 通过 `--system-site-packages` 只读继承 conda 的 Torch/Triton/NumPy，详细包来源与推荐重建方式见{doc}`appendix/install`，运行状态见{doc}`15-results`。阅读和构建本书只需要文档依赖，文档配置不会导入 PyTorch，也不会初始化 CUDA。代码示例中 `torch.testing.assert_close` 用来表达需要验证的数值契约，不代表这些示例已经在每一个 PyTorch/DSL 版本组合上验证过。

## 学习路线

```{toctree}
:maxdepth: 2
:caption: 第一部分：硬件、执行与语言

01-environment
02-execution
03-performance
04-memory-layout
05-tensor-cores
06-async
07-triton
08-triton-layouts
```

```{toctree}
:maxdepth: 2
:caption: 第二部分：从 GEMM 到注意力

09-gemm-basics
10-gemm-pipeline
11-gemm-advanced
12-attention
13-prefill
14-decode
15-results
```

```{toctree}
:maxdepth: 2
:caption: 附录与资料

appendix/install
appendix/benchmarking
appendix/debugging
appendix/sources
```

## 代码与术语约定

`src/fa5060/fundamentals.py` 与 `attention.py` 是 Triton 实现；`src/fa5060/tilelang/` 是 TileLang 实现；`tests/` 表达形状、掩码、数值与缓存契约；`results/` 存放发布证据，`results/local/` 保存读者自己的输出。上游参考通过固定提交链接访问；章节对应关系、参考来源与架构差异见{doc}`appendix/sources`。

本书说“块”时会注明是逻辑 tile、CUDA thread block（CTA），还是缓存页。`program` 指 Triton 的一个程序实例，通常对应一个 CTA；TileLang 的 `T.Kernel` 也以 CTA 为协作范围；`warp` 固定包含 32 个 lane。GB/s 与 TFLOP/s 使用十进制单位，KiB 与 MiB 使用二进制单位；没有单位的 stride 均以元素计。注意力张量采用 $Q[B,S_q,H_q,D]$ 的存储约定，decode 的单 token 查询写成 $Q[B,H_q,D]$。
