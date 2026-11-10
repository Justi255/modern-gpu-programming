(ch-environment)=
# 01　先把机器与实验条件说清楚

GPU 编程的第一步不是选一个很大的 tile，而是确认程序实际运行在哪台机器上。同一个营销代号可能覆盖多种计算能力；同一个 kernel 也可能因驱动、编译器、功耗或输入布局不同而改变行为。记录环境能把“我的结果与你不同”转化为可以排查的问题。

本书使用 RTX 5060 Laptop GPU，设备探测得到计算能力 **12.0（SM120）**、**26 个 SM**，显存约 **8 GB**。2026-11-02 历史发布的软件版本是 PyTorch `2.12.1+cu130`、Triton `3.7.1`、CUDA Toolkit `13.1`、NVIDIA 驱动 `595.99.02`。这些数字是该次环境记录，安装时应优先保证版本兼容，而不是把它们当作永远成立的最低要求。便携机还受到功耗模式、散热和电源状态影响，比较结果前要记录这些条件。

## 1.1 三层 CUDA 版本各自回答什么问题

`nvidia-smi` 显示的 CUDA 版本表示驱动支持的 CUDA 兼容范围，不等于系统 `nvcc` 的版本。`nvcc --version` 对应本机 CUDA Toolkit；`torch.version.cuda` 对应 PyTorch 二进制构建时使用的 CUDA 版本。三者数值不同，本身并不说明环境损坏。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TileLang 的 Python 工厂先根据 shape、dtype、stride 与配置构造 `T.prim_func`，JIT 再把 tile 程序降低为设备代码并构建调用器。首次调用可能包含编译、加载模块与初始化；已有较新的 `nvcc` 也不能证明这一组 TileLang、TVM-FFI 与 SM120 目标可以工作。运行 PyTorch wheel 与编译 TileLang CUDA 代码使用的工具链职责不同，出现错误时分别检查驱动、wheel runtime 和本机 Toolkit。

::::

::::{tab-item} Triton
:sync: triton

Triton 在首次遇到一个新的特化 kernel 时进行 JIT 编译。它不等同于把 Python 代码交给 `nvcc` 编译，因此“系统有新版本 nvcc”不能保证旧 Triton 认识 SM120。反过来，运行一个预装 PyTorch wheel 也不一定要求本机安装完全相同版本的 Toolkit。需要编译额外 CUDA 扩展时，Toolkit 的要求才会进一步出现。

::::
:::::

在仓库根目录新建干净的 `.venv-repro`，让两种后端使用同一套 Python、Torch 与外部基线。下面的环境名称应尚未使用；详细的依赖与包约束说明见{doc}`appendix/install`：

```bash
python3.13 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-tilelang.txt
nvidia-smi
nvcc --version
python -m fa5060.probe --output results/local/environment.json
```

`requirements-tilelang.txt` 同时安装 TileLang、Triton、测试、文档与外部基线，并引用本次记录的包版本约束；CUDA PyTorch wheel 仍需先选择安装来源。读者探针与结果保存在 `results/local/`，避免改写发布证据。发生目标架构、编译或模块加载错误时，分别核对 DSL、TVM-FFI、Toolkit 与 SM120，不把 import 成功当作 kernel 已运行。

本次实际采集使用的 `.venv-tilelang` 由 `--system-site-packages` 创建：TileLang、TVM-FFI、新基线及文档包安装在该 venv 内，Torch、Triton、NumPy 从已有 conda 基础 Python 只读继承。原 `.venv` 的 GPU 依赖未改动。这不是完全隔离的环境；已安装包的版本与来源范围记录在 `results/tilelang/packages-all.json`。上面的干净环境是读者推荐流程，两种环境都需重新运行探针与实际设备检查。

下面的独立检查只读取环境与设备属性：

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
import tilelang

assert torch.cuda.is_available(), "需要可见的 NVIDIA GPU"
props = torch.cuda.get_device_properties(0)
print("torch:", torch.__version__, "built with CUDA:", torch.version.cuda)
print("tilelang:", tilelang.__version__)
print("device:", props.name, "SM:", torch.cuda.get_device_capability())
print("SM count:", props.multi_processor_count)
print("VRAM GiB:", props.total_memory / 2**30)
print("shared block bytes:", props.shared_memory_per_block)
print("shared opt-in bytes:", props.shared_memory_per_block_optin)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
import triton

assert torch.cuda.is_available(), "需要可见的 NVIDIA GPU"
props = torch.cuda.get_device_properties(0)
print("torch:", torch.__version__, "built with CUDA:", torch.version.cuda)
print("triton:", triton.__version__)
print("device:", props.name, "SM:", torch.cuda.get_device_capability())
print("SM count:", props.multi_processor_count)
print("VRAM GiB:", props.total_memory / 2**30)
print("shared block bytes:", props.shared_memory_per_block)
print("shared opt-in bytes:", props.shared_memory_per_block_optin)
```

::::
:::::

传统每块共享内存默认上限与可显式申请的 opt-in 上限可能不同。本设备每 CTA 可申请的共享内存上限为 **99 KiB，即 101376 字节**；这不是说任意配置都能自动申请到这么多，也不是说一个 SM 可以同时驻留许多个这样的 CTA。编译器分配的缓冲区、流水级数和后端限制都会参与资源检查。

## 1.2 Blackwell 的名称不足以选择指令

下表是本书选择技术路线时使用的边界。它区分“设备具备某项能力”和“本项目已经走到这条代码路径”：

| 问题 | 本书的 SM120 路线 | 不能照搬的 SM100 假设 |
| --- | --- | --- |
| FP16/BF16 的主要 MMA | warp 级 `mma.sync`，FP32 寄存器累加 | `tcgen05.mma` 与 TMEM 累加器 |
| 中间矩阵保存在哪里 | 寄存器，必要时经共享内存交换 | TMEM 分配、读取和回收协议 |
| 全局到共享内存传输 | 常规 load、编译器流水化；设备也支持 TMA | 看到 `tl.load` 就认定用了 TMA |
| 线程组织 | CTA 内多个 warp 协作 | 默认存在 SM100 warp specialization 路径 |
| 工作分发 | 常规 grid，或软件 persistent 循环 | 默认可使用 Cluster Launch Control |

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

TMA 是设备能力，能否用于当前 TileLang 的 `T.copy`，还取决于目标、传输区域、对齐、pass 配置和生成代码。不要从一个 `T.copy` 调用直接推断 TMA 已启用；后文把硬件能力、编译证据和性能观测分别记录。

::::

::::{tab-item} Triton
:sync: triton

TMA 是设备支持的能力，但能否在某个 Triton API、维度、对齐与布局组合下生成对应指令，仍需单独验证。后文会讨论它的语义，不把硬件支持等同于所有教材代码都已经使用它。

::::
:::::

## 1.3 让第一次运行可解释

先跑基础正确性，再跑短基准。首次 JIT 会花时间编译，不应进入稳态 kernel 时间。PyTorch 的分配器也可能在第一次分配时初始化；程序退出时打印的一行总耗时通常混合了导入、编译、分配与 GPU 工作。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```bash
python -m pytest tests/test_tilelang_fundamentals.py -q
python -m fa5060.experiments basics --backend tilelang --output results/local/tilelang/basics.json
python -m fa5060.benchmark --backend tilelang --suite smoke --phase all --output results/local/tilelang/smoke.json
```

::::

::::{tab-item} Triton
:sync: triton

```bash
python -m pytest tests/test_fundamentals.py -q
python -m fa5060.experiments basics --backend triton --output results/local/triton/basics.json
python -m fa5060.benchmark --backend triton --suite smoke --phase all --output results/local/triton/smoke.json
```

::::
:::::

### 第一次真实 launch

以下小例子只把 PyTorch 用于创建输入、参考与断言；加法由所选 DSL 的真实内核完成。首次调用可能编译，下一章会展开 CTA 覆盖与尾部保护。

:::::{tab-set}
:sync-group: kernel-backend

::::{tab-item} TileLang
:sync: tilelang

```python
import torch
from fa5060.tilelang import vector_add

x = torch.arange(1003, device="cuda", dtype=torch.float32)
y = torch.full_like(x, 0.5)
z = vector_add(x, y)
torch.testing.assert_close(z, x + y)
```

::::

::::{tab-item} Triton
:sync: triton

```python
import torch
from fa5060.fundamentals import vector_add

x = torch.arange(1003, device="cuda", dtype=torch.float32)
y = torch.full_like(x, 0.5)
z = vector_add(x, y)
torch.testing.assert_close(z, x + y)
```

::::
:::::

运行这些命令前检查空闲显存。8 GB 需要同时容纳输入、输出、参考实现中间值、工作区和框架缓存。单个 FP16 张量的存储量为

$$
S = 2\prod_i d_i\quad\text{字节}.
$$

例如 $[1,4096,8,64]$ 包含 $2^{21}$ 个元素，占 4 MiB。三个这样的 Q/K/V 占 12 MiB，但如果显式物化所有注意力分数，$[1,8,4096,4096]$ 的 FP32 张量已经占 512 MiB。这个简单计算解释了为什么只看输入大小会严重低估参考实现的峰值显存。

## 1.4 独立构建书籍

CPU 机器也可以构建文档。文档只读取文本与源码，不执行 GPU 示例：

```bash
python -m pip install -e '.[docs]'
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --directory docs/_build/html
```

打开本地端口后可以检查公式、导航、图和源码片段。`-W` 把警告视为失败；缺页、坏链接、找不到的源码和 MyST 解析问题都应该修正，不能通过排除整章来制造成功。

## 练习

1. 保存自己的探测 JSON，区分驱动、Toolkit、PyTorch CUDA 版本，并写出每个版本由哪个命令得到。
2. 计算 FP16 的 $[4,8192,8,128]$ 张量大小，再计算同一批次、8 个头的显式 FP32 分数矩阵大小。哪一个决定显存预算？
3. 找一篇针对 B200 的优化文章，圈出其中依赖 `tcgen05`、TMEM 或 Cluster Launch Control 的部分。为每一部分写出移植到 SM120 前必须验证的条件。
