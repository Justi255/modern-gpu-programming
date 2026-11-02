(installation)=
# 附录 A：安装、环境探针与最小复现

本项目将文档构建、CPU 语义检查和 GPU 执行分开。阅读或构建教材不需要 CUDA，运行 Triton 内核则需要 Linux 上可用的 NVIDIA GPU、相容的驱动与包含 SM120 支持的工具链。RTX 5060 Laptop 与 B200 不是同一执行目标；软件中出现「Blackwell」字样，不足以证明当前包能在本机运行。

## A.1 建立独立 Python 环境

Python 最低版本以仓库的 `pyproject.toml` 为准，目前为 3.10。下面的命令使用标准 venv，不修改系统 Python：

```bash
git clone https://github.com/Justi255/modern-gpu-programming.git
cd modern-gpu-programming
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[docs,test]'
```

此时可以构建教材；不必为了文档安装 TIRx、FlashAttention 或 GPU 驱动。严格构建会把缺失的图片、引用与不合法指令暴露出来：

```bash
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m http.server --directory docs/_build/html 8000
```

浏览器访问 [本地教材预览](http://localhost:8000)。第二条命令持续运行，结束时按 Ctrl-C。它只是预览服务器，不参与基准测试。远程运行时，可以通过 SSH 端口转发访问同一页面。

## A.2 选择支持 SM120 的 PyTorch 与 Triton

不要仅根据 `torch.cuda.is_available()` 判断工具链完整。这个检查只能证明 CUDA 运行时看见设备，不能证明 PyTorch 所带矩阵库、Triton 的代码生成器与设备指令集匹配。应使用 PyTorch 官方提供的、支持该 GPU 的 CUDA wheel，并记录精确版本。下面给出 CUDA 12.8 wheel 索引的安装路径；具体版本与驱动条件仍应核对[官方安装选择器](https://pytorch.org/get-started/locally/)和版本说明。

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[gpu]'
nvidia-smi
python -m fa5060.probe --output results/environment.json
```

如果已有相容的 PyTorch 环境，先查看它，再决定是否升级，不必重复安装另一份 CUDA toolkit。PyTorch wheel 携带其运行时依赖；本项目的 Triton JIT 也有自己的编译链。只有在编译额外 CUDA 扩展或使用特定 profiler 时，才需要核对本地 toolkit、编译器和开发头文件。

要复现本仓库公开的数据，使用 Python 3.13，并按实际实验版本安装；上面的 CUDA 12.8 示例是一般安装路径，不是本次数据使用的环境：

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-experiment.txt
# 本机可选基线 JIT 使用的 toolkit 路径；其他机器按实际安装位置设置。
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
python -m fa5060.probe --output results/environment.json
```

`requirements-experiment.txt` 固定核心实验包版本，逐次记录中的 `environment` 固定运行时与驱动观测。它不替代 CUDA 驱动安装，也不保证不同平台得到同样的时钟与热状态。第一次 FlashInfer/CuTe 编译可能需要较长时间，编译时间不进入稳定性能比较。

探针文件是结果的一部分。检查其中设备名、compute capability、SM 数、显存以及 Torch/Triton/CUDA 版本，再与本次实验所声称的目标比对。驱动报告的「CUDA Version」表示驱动可支持的运行时上限，不等于当前 PyTorch 实际使用的 CUDA runtime 版本。

## A.3 从小检查扩展到完整基准

CPU 检查先验证指标聚合、工作负载定义和参考语义；GPU 检查验证实际编译与数值执行。没有 GPU 的机器应按测试标记筛选，而不是把跳过当作 GPU 正确性的证明。

```bash
python -m pytest -m 'not gpu'
python -m pytest -m gpu
python -m fa5060.experiments basics --output results/basics.json
python -m fa5060.experiments gemm --output results/gemm.json
python -m fa5060.experiments attention --output results/attention.json
python -m fa5060.benchmark --suite smoke --phase all --output results/smoke.json
```

只在这些路径已通过后，才运行 full suite。8GB 级显存需要注意其他进程占用；不要通过静默缩小正式 workload 来掩盖 OOM。如果为调试临时换用小形状，应把它标为独立调试实验，保留正式 case 的失败状态。

## A.4 可选基线不是运行时必需项

`baselines` extra 声明项目用于对照的可选 Python 包：

```bash
python -m pip install -e '.[baselines]'
python -m fa5060.benchmark --suite smoke --phase all --output results/smoke-with-baselines.json
```

这些包可能受平台、CUDA 版本、构建依赖和目标架构限制。安装成功后仍需对每个 case 检查是否真正使用 SM120 的兼容路径、是否通过正确性、是否原生支持分页。缺失项应留在报告中；没有原生基线不妨碍学习自定义内核，但会限制性能结论。

## A.5 保存足够的复现信息

复现报告应包含源码提交、环境 JSON、完整命令、输入形状、随机种子、是否外接电源、功耗/风扇模式和结果文件。不要粘贴访问令牌或私人目录内容。对于编译错误，保存第一条真实诊断与触发配置；对于数值错误，保存最小输入及误差分布；对于性能波动，保存所有轮次而不是只截一张最快结果。

本书不要求安装上游 TIRx 才能运行本项目。若希望研究上游 SM100/TIRx 实现，请使用[固定版本来源表](sources.md)中的独立链接，并在支持其指令路径的设备上建立另一套环境。不要把两份工程的依赖混在一起，误以为安装了上游包就能在 SM120 执行其 TMEM 内核。
