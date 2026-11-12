# FMHAv2 独立 FP32 accumulation 模式实现

日期：2026-11-11。交付仅涉及 `scripts/fmha_fp32acc.py`、
`tests/test_fmha_fp32acc.py` 与本文档。未改安装包、原生 FlashInfer 缓存、
benchmark provider/evaluator、reference、输入、seed 或阈值；未提交或 push。

**CUDA 尚未编译、尚未运行、尚未验证正确性或速度。** 本实现不能证明四个原生失败
已经解决，也不能单独构成冻结 SOTA 资格。原生失败及原始报告必须继续保留。

## 执行命令

在 `/home/justi/Workspace/cuda/fa` 执行。目录必须是新目录，示例路径如已存在需换名：

```bash
# 无 GPU：生成源码、原生快照、runner.patch 和 SHA256 manifest。
.venv-tilelang/bin/python -B scripts/fmha_fp32acc.py generate \
  --directory .work/fmha-fp32acc-v1

# 由持有 GPU 的主 agent 执行；编译两种 layout，总共 8 个 CU。
.venv-tilelang/bin/python -B scripts/fmha_fp32acc.py build \
  --directory .work/fmha-fp32acc-v1

# 同一 AttentionInputs 对象比较原生与修正模式，选择原报告里的错误 case。
.venv-tilelang/bin/python -B scripts/fmha_fp32acc.py validate \
  --directory .work/fmha-fp32acc-v1 \
  --native-report results/tilelang/full-completed.json \
  --output .work/fmha-fp32acc-failures-v1.json

# 然后遍历冻结 suite 的全部 156 个 FP16 case，包括 prefill/decode、dense/paged。
.venv-tilelang/bin/python -B scripts/fmha_fp32acc.py validate \
  --directory .work/fmha-fp32acc-v1 --suite full-fp16 \
  --native-report results/tilelang/full-completed.json \
  --output .work/fmha-fp32acc-full-fp16-v1.json
```

可重复 `--case <frozen-id>` 指定诊断子集；不存在的 ID、BF16 与 D64/128 之外的
case 均拒绝。`--help` 和各子命令的 `--help` 仅依赖 Python 标准库，包括
`python -S scripts/fmha_fp32acc.py --help`。generate 按需依赖 Jinja2 和已安装源码，
不导入 Torch 或 FlashInfer。

validate 是独立诊断，不写回原结果。所有 FP32acc case 的 eager、预分配、首次图执行
和再次 replay 均通过冻结 `_validate_output` 才返回 0，否则返回 1。CUDA 构建或
基础设施异常会直接报错。报告逐 case 原子保存，原始 native provider observation
原样附在 `original_native_observation`，同时保存原始报告路径与 SHA256。

## 与安装版的边界

- 使用安装版 `generate_kernel_spec(sm=120, dtype="fp16", output_dtype="fp16", ...)`
  生成 D64/D128 的原生几何，再仅 `dataclasses.replace(spec, dtype="fp16_fp32")`。
- 使用安装版 `encode_name`、`is_kernel_spec_valid`、`get_kernel_code`、`get_api_code`。
  每种 layout 两个 kernel CU，正向 `&& force_fp32_acc` dispatch gate 两处；
  输入 dispatch 仍为 `DATA_TYPE_FP16`。
- CPU 导入只在**内存 AST**中移除 `env`、`CompilationContext`、
  `write_if_different` 三个整合 import；向独立私有模块注入模板路径与类型注释占位。
  安装版纯生成函数、utils 与 Jinja 模板均未改。未注入或 monkeypatch 名为
  `flashinfer` 的模块。此适配写入 manifest。
- 私有 runner 仅改变 `acc_type = DATA_TYPE_FP32` 与
  `force_fp32_acc = true` 两个赋值。原生注释完整保留，可能与修正模式相反；
  `runner.patch` 是精度变更的权威记录。binding 原样复制。
- `native/` 保留生成器、runner、binding、所有 FMHAv2 header/template 及相关原生
  Python 接口源码；manifest 同时记录这些原生文件 SHA 与所有生成文件 SHA。
  每次 build 前核验生成文件和当前安装版原生源码，不匹配即拒绝。
- URI 为 `fa5060_fmha_v2_fp32acc_<layout>_<source-sha16>`；其摘要包含相关原生
  FMHAv2 源码/头文件/模板及脚本。binary 与生成源不混用原生 URI。
- build 导入 FlashInfer **之前**设置 `FLASHINFER_WORKSPACE_BASE=<directory>/cache`
  与 `FLASHINFER_CUDA_ARCH_LIST=12.0`。若进程已用另一个 cache 导入 FI 则拒绝。
  真实 GPU 必须为 SM120。使用实际 `flashinfer.jit.core.current_compilation_context`、
  `gen_jit_spec(...).build_and_load()` 和 `JitSpec.get_library_path()`；保留原生
  `use_fast_math=True` 及 include/编译 flag 规则，FMHAv2 include 指向本次原生快照。
- build.json 记录最终 NVCC flags、二进制路径/SHA、软件版本与设备信息。
  生成 manifest 始终写 `gpu_validation=not_run`；后续验证结果仅记在独立校验报告。
- runner 使用与 `prefill.py` 相同的 26 参数 ABI。Dense 在每次 eager/graph call 内
  `torch.stack((k,v), dim=1)`，随后将 KV tensor 同时传给 runner 的 K/V 参数；
  paged 保留原生共享 K/V allocation，预计算 `[B,2,M]` block offset。
  Q/K/V 和输出均保持 FP16，scale/mask/长度均与现有 adapter 一致。
- 同一个 inputs 与 reference 对象依次用于 native/FP32acc。调用原封不动的
  `_validate_output(...,"fp16")`，额外记录失败元素数、最大误差所在 batch/query/head
  和归一化 RMS。图捕获后先将输出填 NaN，再执行图；第二次 replay 前再次填 NaN，
  避免预热值掩盖没有写出的输出。

## 构建开销及主进程复用

私有 corrected 模块总共 8 个 translation units。CLI validate 同样把原生 provider
放在私有 cache，因此原生 dense/paged 的完整生成可能再次编译（各约 26 个 CU）。
这是确保原缓存不被修改的代价，不应误认为 corrected 模块本身需要那些 CU。

若主 agent 已在自己的进程加载原生 provider，可先在新进程执行上面的 build，
随后在原进程**直接读取私有二进制**，避免通过本脚本 build 再次导入/配置 FI：

```python
import importlib.util
import json
from pathlib import Path
import tvm_ffi

root = Path('/home/justi/Workspace/cuda/fa')
spec = importlib.util.spec_from_file_location('independent_precision', root / 'scripts/fmha_fp32acc.py')
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)
directory = root / '.work/fmha-fp32acc-v1'
precision.verify_manifest(directory)
built = json.loads((directory / 'build.json').read_text())
layout = 'CONTIGUOUS_Q_KV' if case.layout == 'dense' else 'Q_PAGED_KV_NHD'
library = Path(built['layouts'][layout]['library_path'])
assert precision.sha256(library) == built['layouts'][layout]['library_sha256']
module = tvm_ffi.load_module(str(library))
corrected = precision.corrected_provider(case, inputs, module)
# 保持 inputs/reference 与现有 native provider 完全相同。
observations = precision._observe(corrected, reference)
```

此复用方式尚未 GPU 验证；不会替换任何 native provider factory。BF16 继续使用原模式，
不转 dtype、不改变 baseline。性能测量与 SOTA qualification 需主 agent 另外进行。

## CPU 验证证据

```bash
.venv-tilelang/bin/python -B -m pytest \
  tests/test_fmha_fp32acc.py tests/test_benchmark.py tests/test_workloads.py \
  -m 'not gpu' -q
# 41 passed, 4 deselected in 0.83s
```

新增文件含 9 项 CPU 检查：无 site-packages 的 help、两赋值可逆 patch、真实安装版
生成器与原生源码证据、生成源篡改拒绝、全 FP16 suite、原生失败选择、冻结校验错误与
最差坐标、dense/paged 严格 26 参数 ABI。初次测试在实现缺失时失败，后续 dense ABI
测试还真实捕获了一次 KV placeholder 传参偏差并修正。

CPU 生成快照保存在 `/tmp/fa5060-fmha-fp32acc-cpu-v2`，包含全部源码与 manifest；
可由 GPU 主 agent 直接 build，也可在 `.work/` 重新 generate。没有 CUDA 编译证据。

一次扩展检查忘记排除已有 GPU 标记测试，触发了四次 `torch.cuda.is_available()`
检查后全部 skip；没有执行 CUDA kernel、CUDA 编译、显存操作或 nvidia-smi。随后
明确 `-m 'not gpu'` 的最终验证未包含这些测试。此事实已向主 agent 披露。
