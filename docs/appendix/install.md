(installation)=
# 附录 A：安装、环境探针与最小复现

文档构建、CPU 合同检查和 GPU 执行使用不同依赖。文档配置不导入任何 GPU 编译器；GPU 路线则必须实际验证 SM120、矩阵指令、数值与调用方式。安装成功只完成工具链的第一步，最终证据见[结果章](../15-results.md)。

## A.1 建立独立 Python 环境

仓库基础 Python 最低要求由 `pyproject.toml` 定义；本次双后端实验固定使用 Python 3.13。先克隆项目，再建立供两种后端共同使用的干净环境：

```bash
git clone https://github.com/Justi255/modern-gpu-programming.git
cd modern-gpu-programming
```

```bash
python3.13 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip
# CUDA wheel 先单独选择，避免普通索引装成其他运行时版本。
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-tilelang.txt
```

`.venv-repro` 应为尚未使用的目录。`requirements-tilelang.txt` 定义本项目的安装入口，包含 `tilelang`、`baselines`、`docs`、`test` extras 和 CuTe DSL，并通过 `-c` 引用 `results/tilelang/packages-constraints.txt`。约束文件只固定被请求依赖的版本，不表示要安装其中列出的全部 conda 包；`packages-all.json` 则记录实际可见包的版本和 `venv` / `inherited-base` 来源范围。这里记录版本，没有把约束表冒充带 wheel 哈希的完整锁文件。

本次实际测量环境 `.venv-tilelang` 使用 `--system-site-packages`，只读继承已有 conda 基础 Python 的 Torch、Triton、NumPy；TileLang、TVM-FFI、新基线及文档包安装在 venv 本地。原 `.venv` 的 GPU 依赖未改动。该环境的继承关系与推荐的干净环境不同，不能称为完全隔离；两端配对测量仍统一在同一个 `.venv-tilelang` 中执行。读者使用上面的 `.venv-repro` 重建软件组合，再保存自己的探针和验证结果。

只构建教材时可以单独使用 `.venv-docs`，不安装 Torch、TileLang 或 Triton：

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -e '.[docs]'
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m http.server --directory docs/_build/html 8000
```

访问[本地预览](http://localhost:8000)。严格构建同时解析两套标签的所有源码引用；默认显示某一后端不意味着另一套内容被排除。浏览器选择可持久保存；`?kernel-backend=tilelang` 和 `?kernel-backend=triton` 可指定路线。

## A.2 选择支持 SM120 的 PyTorch 与 Triton

本节保留旧标题和锚点，涵盖两套编译器。`nvidia-smi` 报告驱动兼容上限，`nvcc --version` 报告 Toolkit，`torch.version.cuda` 报告 wheel runtime。当前记录为 CUDA Toolkit 13.1、Torch 2.12.1+cu130、Triton 3.7.1、驱动 595.99.02。SM120 不等同于数据中心 SM100，包名中包含 Blackwell 也不保证支持当前路径。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

`tilelang` extra 固定 `tilelang==0.1.15`、`apache-tvm-ffi==0.1.12`、`torch==2.12.1` 与 `triton==3.7.1`。TVM-FFI 是调用桥接依赖，生成的 TIR/TIRx 是编译器表示；不需要另装原教材的手写 TIRx 工程。固定上游提交用于来源引用；探针没有 wheel 的 VCS 信息时，不据此断言该 wheel 构建于那个提交。

```bash
source .venv-repro/bin/activate
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
python -m pip check
python -m fa5060.probe --output results/local/tilelang/environment.json
python -c "import tilelang, tvm_ffi; print(tilelang.__version__); print(tvm_ffi.__version__)"
```

TileLang CUDA 编译、TVM-FFI 模块加载和实际 SM120 launch 是三个检查点。首次执行会编译，须在计时与 Graph 捕获前完成。版本或路径变化后重新运行探针和数值测试，不沿用旧的“已验证”标签。

::::

::::{tab-item} Triton
:sync: triton

`gpu` extra 给出一般 Torch/Triton 下限；`requirements-experiment.txt` 记录历史发布的软件组合。新版双后端对照统一使用 A.1 的环境与 `requirements-tilelang.txt`，不用在两个环境中分别安装。Triton 的 JIT 工具链与系统 `nvcc` 职责不同，新增 Toolkit 不会自动修复旧 Triton 的目标支持。

```bash
source .venv-repro/bin/activate
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
python -m pip check
python -m fa5060.probe --output results/local/triton/environment.json
```

历史 2026-11-02 结果只描述原记录环境。重新安装后应保存新的探针，不覆盖旧文件来伪装相同环境。

::::
:::::

PyTorch wheel 的选择须核对[官方安装说明](https://pytorch.org/get-started/locally/)与驱动兼容性。上述 wheel 索引是本次环境的选择，不是对未来任意版本的安装承诺。设备应可见、compute capability 为 12.0；探针中还要保留 SM 数、共享内存、L2、显存、编译器与调用器版本。

## A.3 从小检查扩展到完整基准

先执行 CPU 合同检查，再逐步运行基础、GEMM、小注意力和 smoke。CPU 跳过 GPU 测试不能计为 GPU 通过：

```bash
python -m pytest -q -m 'not gpu'
```

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m pytest -q tests/test_tilelang_fundamentals.py tests/test_tilelang_attention.py
python -m fa5060.experiments basics --backend tilelang --output results/local/tilelang/basics.json
python -m fa5060.experiments gemm --backend tilelang --output results/local/tilelang/gemm.json
python -m fa5060.experiments attention --backend tilelang --output results/local/tilelang/attention.json
python -m fa5060.benchmark --backend tilelang --suite smoke --phase all --output results/local/tilelang/smoke.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m pytest -q tests/test_fundamentals.py tests/test_attention.py
python -m fa5060.experiments basics --backend triton --output results/local/triton/basics.json
python -m fa5060.experiments gemm --backend triton --output results/local/triton/gemm.json
python -m fa5060.experiments attention --backend triton --output results/local/triton/attention.json
python -m fa5060.benchmark --backend triton --suite smoke --phase all --output results/local/triton/smoke.json
```

::::
:::::

CLI 为兼容旧脚本仍默认 Triton，页面默认 TileLang；新命令始终显式选择。8 GB 显存须容纳输入、输出、参考、scratch 和桌面占用。正式 case 出现 OOM 时保留失败，不通过静默缩小形状获得“全通过”。逐 phase full 与同环境比较见第十五章。

## A.4 可选基线不是运行时必需项

`baselines` extra 固定本项目使用的 FlashInfer 与 FA4，A.1 的完整复现入口已包含它。只安装基础 extras 的环境可以单独补充；安装后仍逐 case 检查设备、dtype、分页和 causal 兼容性。两种自定义后端使用相同语义与选择规则。

```bash
python -m pip install -e '.[baselines]'
```

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m fa5060.benchmark --backend tilelang --suite smoke --phase all --output results/local/tilelang/smoke-with-baselines.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m fa5060.benchmark --backend triton --suite smoke --phase all --output results/local/triton/smoke-with-baselines.json
```

::::
:::::

原生分页 provider 不能被计时区外 gather 后的稠密路径替代。缺失基线不妨碍运行自定义内核，但会限制完整性能结论。初次 FlashInfer/FA4 编译、规划和工作区初始化移出稳态计时。

## A.5 保存足够的复现信息

保留源码提交、探针、完整命令、后端、随机种子、shape/stride/dtype、长度与页表策略、配置、基线状态、所有轮次，以及电源/散热条件。编译失败保留第一条诊断，数值失败保留最小输入，性能波动保留慢轮。

读者命令统一写入 `results/local/tilelang/`、`results/local/triton/` 和 `results/local/comparison/`。它们被 Git 忽略；需要归档时另行复制并记录来源。仓库已有 `results/` 和 `results/tilelang/` 文件属于发布证据，不用读者运行结果覆盖。

本书仅提供 TileLang 与 Triton 两套 device 实现。上游 SM100/TIRx 资料用于比较；生成 TIR/TIRx 的编译解释不要求手写第三种语言或在 SM120 执行 TMEM/`tcgen05`。固定来源见[来源表](sources.md)。
