# 面向 RTX5060 的现代 GPU编程：Triton与TileLang对照实践

一套面向有 CUDA 基础的读者的完整中文教材：从执行、访存、布局与编译，到三种 GEMM、在线 softmax、prefill、GQA/MQA、原生分页 KV 与 split-KV decode。目标设备为 RTX 5060 Laptop GPU（SM120）。

[完整教材](docs/index.md)的每个后端相关例子、命令与解释都提供 **TileLang / Triton 同步标签**，默认 TileLang，跨页和刷新保留选择。公共数学与硬件模型共享；旧章节 URL 和锚点保留。

GitHub 中的 Markdown 是教材源文件，同步选项卡在构建后的 HTML 中生效。可以按下面的命令本地阅读，或从 [Actions](https://github.com/Justi255/modern-gpu-programming/actions/workflows/verify.yml) 的成功运行下载 `rtx5060-textbook-html`。

- [安装与环境](docs/appendix/install.md)
- [15 章学习路线](docs/index.md#学习路线)
- [结果、覆盖状态与性能判据](docs/15-results.md)
- [固定来源与平台差异](docs/appendix/sources.md)

文档构建只需 CPU：

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -e '.[docs,test]'
python -m pytest -q -m 'not gpu'
python -m sphinx -n -W --keep-going -b html docs _build/html
python -m http.server --directory _build/html 8000
```

保持服务运行，打开 [http://localhost:8000](http://localhost:8000)。下载的站点产物解压后也可用 `python -m http.server --directory 解压目录 8000` 阅读，无需安装 GPU 编译器。

GPU 工具链安装见附录。推荐在全新 `.venv-repro` 中按 `requirements-tilelang.txt` 安装，两种后端使用同一环境；版本为 TileLang 0.1.15、TVM-FFI 0.1.12、PyTorch 2.12.1、Triton 3.7.1，CUDA Toolkit 13.1。本次实际 `.venv-tilelang` 使用 `--system-site-packages` 只读继承 conda 的 Torch/Triton/NumPy，编译器及新基线安装在 venv 内；完整版本与来源范围见 [环境记录](results/tilelang/README.md)。进入推荐环境后分别运行：

```bash
# TileLang 路线
.venv-repro/bin/python -m fa5060.experiments basics --backend tilelang --output results/local/tilelang/basics.json
.venv-repro/bin/python -m fa5060.benchmark --backend tilelang --suite smoke --phase all --output results/local/tilelang/smoke.json
# Triton 路线
.venv-repro/bin/python -m fa5060.experiments basics --backend triton --output results/local/triton/basics.json
.venv-repro/bin/python -m fa5060.benchmark --backend triton --suite smoke --phase all --output results/local/triton/smoke.json
```

网页默认 TileLang；CLI 为兼容旧脚本仍默认 Triton，新示例显式指定 `--backend`。两端都执行独立的 device 内核，PyTorch 仅用于分配、独立参考和验证。FP16/BF16 注意力使用 FP32 统计、scratch 与累加，decode 显式工作区支持预热后的 CUDA Graph 调用。SM120 使用 warp MMA 和寄存器累计，不照搬 SM100 的 TMEM/`tcgen05`。

新发布的两后端各 **312 项测量已完成**，自有算子输出全部通过检查。相对逐项最快正确兼容外部基线，TileLang 的 prefill/decode 几何平均加速为 **0.739×/0.980×**，Triton 为 **0.939×/1.132×**；prefill 只统计 140/144 项合格子集，另 4 项外部基线数值失败完整保留。**两端均未达到约定性能目标**：每个 phase 仍须同时满足平均至少 1.10×、最差至少 0.95× 及完整数值合同。最差项、全部配置、原始轮次、分段续测和遥测限制见[结果章](docs/15-results.md)及[实验记录](results/tilelang/README.md)。

最终本机 GPU+CPU 测试为 **557 项通过**，不含 GPU 依赖的文档环境为 **99 项通过**；GPU 跳过情况单独记录。[编译证据](results/tilelang/compiler/README.md)包含 9 个 kernel 的生成 CUDA/PTX 与资源报告。`results/full-tuned.json` 继续保存 **2026-11-02 历史 Triton** 记录，不用于新版后端间比值。

[Actions](https://github.com/Justi255/modern-gpu-programming/actions/workflows/verify.yml) 构建完整双后端 HTML 并保留 `rtx5060-textbook-html` 产物。CPU CI 不代表设备验证；本仓库不默认启用 GitHub Pages。

概念参考 MLC Community 的 [Modern GPU Programming for MLSys](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/a78b4e6e9f2be5697705c174d49bbe4974e0c4a5)。TileLang 来源固定为 [v0.1.15，提交 a35f8dd](https://github.com/tile-ai/tilelang/tree/a35f8ddf45eba16c21211ec8822d56ce5363036f)。本书用中文独立组织和推导，保留来源及许可说明，不提供手写 TIRx 第三后端。
