(ch-environment)=
# 01　先把机器与实验条件说清楚

GPU 编程的第一步不是选一个很大的 tile，而是确认程序实际运行在哪台机器上。同一个营销代号可能覆盖多种计算能力；同一个 kernel 也可能因驱动、编译器、功耗或输入布局不同而改变行为。记录环境能把“我的结果与你不同”转化为可以排查的问题。

本书使用 RTX 5060 Laptop GPU，设备探测得到计算能力 **12.0（SM120）**、**26 个 SM**，显存约 **8 GB**。当前工作环境的软件版本是 PyTorch `2.12.1+cu130`、Triton `3.7.1`、CUDA Toolkit `13.1`、NVIDIA 驱动 `595.99.02`。这些数字是本次环境记录，安装时应优先保证版本兼容，而不是把它们当作永远成立的最低要求。便携机还受到功耗模式、散热和电源状态影响，比较结果前要记录这些条件。

## 1.1 三层 CUDA 版本各自回答什么问题

`nvidia-smi` 显示的 CUDA 版本表示驱动支持的 CUDA 兼容范围，不等于系统 `nvcc` 的版本。`nvcc --version` 对应本机 CUDA Toolkit；`torch.version.cuda` 对应 PyTorch 二进制构建时使用的 CUDA 版本。三者数值不同，本身并不说明环境损坏。

Triton 在首次遇到一个新的特化 kernel 时进行 JIT 编译。它不等同于把 Python 代码交给 `nvcc` 编译，因此“系统有新版本 nvcc”不能保证旧 Triton 认识 SM120。反过来，运行一个预装 PyTorch wheel 也不一定要求本机安装完全相同版本的 Toolkit。需要编译额外 CUDA 扩展时，Toolkit 的要求才会进一步出现。

在仓库根目录建立环境并安装项目，详细的依赖选择见{doc}`appendix/install`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[gpu,test,docs]'
nvidia-smi
nvcc --version
python -m fa5060.probe --output results/environment.json
```

`gpu` 依赖声明的是 Python 包要求，不负责替你选择合适的 PyTorch wheel 下载源；已有经过验证的 CUDA 环境时，可以按附录保留它并单独安装本项目。探测脚本的 JSON 应与后续结果一起保留。遇到 `no kernel image`、无效目标架构或 PTX 工具链错误，应先核对架构支持，再检查算法。

下面的独立检查只读取环境与设备属性：

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

TMA 是设备支持的能力，但能否在某个 Triton API、维度、对齐与布局组合下生成对应指令，仍需单独验证。后文会讨论它的语义，不把硬件支持等同于所有教材代码都已经使用它。

## 1.3 让第一次运行可解释

先跑基础正确性，再跑短基准。首次 JIT 会花时间编译，不应进入稳态 kernel 时间。PyTorch 的分配器也可能在第一次分配时初始化；程序退出时打印的一行总耗时通常混合了导入、编译、分配与 GPU 工作。

```bash
python -m pytest tests/test_fundamentals.py -q
python -m fa5060.experiments basics --output results/basics.json
python -m fa5060.benchmark --suite smoke --phase all --output results/smoke.json
```

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
