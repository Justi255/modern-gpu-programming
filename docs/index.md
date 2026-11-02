(book-home)=
# 面向 RTX 5060 的现代 GPU 编程：从 Triton 到 FlashAttention

这本书从一台真实的 RTX 5060 Laptop GPU 出发，用 Triton 实现向量加法、归约、转置、矩阵乘法和注意力。学习终点是能说明一个 kernel 为什么正确、数据为什么按这种方式移动，以及一次性能改进究竟改善了哪一个瓶颈。代码、实验命令和测量方法与文字放在同一个仓库，读者可以逐层修改并验证自己的解释。

本书的目标机器是 **SM120、26 个 SM、约 8 GB 显存的 RTX 5060 Laptop GPU**。这是一份具体设备的观测记录，不代表所有 Blackwell GPU 的配置。特别是，数据中心 Blackwell 的 `tcgen05`、Tensor Memory、双 CTA MMA 与相关调度技巧，不能直接套在这台机器上。本书的 FP16/BF16 矩阵乘法从 warp 级 `mma.sync` 路径出发，累加器保存在寄存器中。

读者需要能够阅读 Python、理解数组下标和矩阵乘法；CUDA C++、PTX 和编译器知识都从用到的地方开始解释。前八章建立执行、访存、数值与布局模型，后七章把这些模型落实到 GEMM、prefill 和 decode。只想先跑通实验的读者可以从{doc}`01-environment`开始，按命令生成自己的结果，再回到原理章节解释它们。

## 怎么读，怎么验证

每章遵循“建立模型 → 推导代价 → 阅读或运行代码 → 检查边界”的顺序。书中的算术例子用于说明模型；设备计时只有由脚本生成并保存的结果才构成测量证据。文中没有把显卡宣传峰值当作已测吞吐，也没有把某个 shape 上的胜出当作全面领先。

基础实验通过以下入口运行，完整安装步骤见{doc}`appendix/install`：

```bash
python -m fa5060.probe --output results/environment.json
python -m fa5060.experiments basics --output results/basics.json
python -m fa5060.experiments gemm --output results/gemm.json
python -m fa5060.experiments attention --output results/attention.json
python -m fa5060.benchmark --suite smoke --phase all --output results/smoke.json
```

运行 GPU 实验需要支持本设备的 PyTorch、Triton 与驱动。阅读和构建本书只需要文档依赖，文档配置不会导入 PyTorch，也不会初始化 CUDA。代码示例中 `torch.testing.assert_close` 用来表达需要验证的数值契约，不代表这些示例已经在每一个 PyTorch/Triton 版本组合上验证过。

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

`src/fa5060/` 是本书原生 Triton 实现；`tests/` 表达形状、掩码、数值与缓存契约；`results/` 存放实验输出。仓库中的 `tutorial/` 是保留的上游参考材料，不参与本书构建。对应关系、参考来源与架构差异见{doc}`appendix/sources`。

本书说“块”时会注明是逻辑 tile、CUDA thread block（CTA），还是缓存页。`program` 指 Triton 的一个程序实例，通常对应一个 CTA；`warp` 固定包含 32 个 lane。GB/s 与 TFLOP/s 使用十进制单位，KiB 与 MiB 使用二进制单位；没有单位的 stride 均以元素计。注意力张量采用 $Q[B,S_q,H_q,D]$ 的存储约定，decode 的单 token 查询写成 $Q[B,H_q,D]$。
