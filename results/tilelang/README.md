# TileLang / Triton 新实验记录

本目录保存 2026-11-05 起采集的设备、依赖、验证、编译与配对性能证据。根目录 `results/full-tuned.json` 是 2026-11-02 的历史 Triton 发布，不能与这里的新 TileLang 时间直接相除。

## 实际环境与依赖记录

本次 `.venv-tilelang` 使用 Python 3.13.13 和 `--system-site-packages`。TileLang 0.1.15、apache-tvm-ffi 0.1.12、新基线及文档包安装在 venv 本地；Torch 2.12.1、Triton 3.7.1、NumPy 2.5.0 从已有 conda 基础 Python 只读继承。原 `.venv` 的 GPU 依赖未改动。该环境具有基础包继承关系，不是完全隔离的 venv；新版两后端配对运行使用同一个环境。

- [environment.json](environment.json) 记录设备、驱动、Toolkit、Python 和编译器版本。GPU 为 RTX 5060 Laptop、SM120、26 SM；Torch CUDA build 为 13.0，NVCC/PTXAS 为 13.1.115，驱动为 595.99.02。
- [packages-all.json](packages-all.json) 记录全部可见 distribution 的版本及 `venv` / `inherited-base` 来源范围。
- [packages-constraints.txt](packages-constraints.txt) 固定被请求依赖的版本；其中列出的 conda 包不是安装清单。它没有 wheel 哈希，不能单独保证构建与平台完全相同。
- [requirements-tilelang.txt](../../requirements-tilelang.txt) 是安装入口，选择项目 extras 和 CuTe DSL，再引用上述约束。
- [source-snapshot.json](source-snapshot.json) 与各任务的 manifest 保存源码快照；结果应连同运行命令和 SHA256 解读。

固定 TileLang 上游提交用于来源归属。探针的 wheel VCS 信息缺失时，不据此断言已安装 wheel 来自该提交。

## 读者在干净环境中复现

以下命令从仓库根目录运行，`.venv-repro` 应为尚未使用的目录。安装本机 CUDA Toolkit 13.1 后，先选 Torch CUDA 13.0 wheel，再安装项目依赖：

```bash
python3.13 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-tilelang.txt
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip check
python -m fa5060.probe --output results/local/environment.json
python -m pytest -q --tb=short
python -m fa5060.benchmark --backend tilelang --suite smoke --phase all --rounds 5 --output results/local/tilelang/smoke.json
python -m fa5060.benchmark --backend triton --suite smoke --phase all --rounds 5 --output results/local/triton/smoke.json
python scripts/compare_backends.py --triton results/local/triton/smoke.json --tilelang results/local/tilelang/smoke.json --output-dir results/local/comparison-smoke
```

两个后端在同一环境中运行，分别保存原始报告。`results/local/` 被 Git 忽略，读者复现不覆盖发布文件；需要保留多次实验时另选目录。完整五轮 full、调优和计时边界见[结果章](../../docs/15-results.md)。干净环境重建仍须通过实际设备测试，版本约束不是运行成功的替代证据。

## 完成状态的判读

[completion-run.json](completion-run.json) 记录本轮续测、Triton 全量和配对报告生成的完成状态及退出码，三步均为 0。[validation-run.json](validation-run.json) 保留原任务中断时的状态，其中的 `running` 不能理解为现在仍在测量。

两端各完成冻结的 312 项（144 prefill / 168 decode），各自最终输出 312/312 通过检查。必要基线组覆盖 312/312，但两端均有同样 4 个 FlashInfer FMHAv2 prefill 数值失败；比较生成器只将完整数值合同合格的 140 项 prefill 纳入外部几何平均。decode 为 168/168 合格。

| 后端 | Phase | 合格项 / 冻结项 | 几何平均加速 | 最差加速 | 低于 0.95× 的合格项 | 达标 |
|---|---|---:|---:|---:|---:|---|
| TileLang | prefill | 140/144 | 0.739× | 0.518× | 129 | 否 |
| TileLang | decode | 168/168 | 0.980× | 0.478× | 88 | 否 |
| Triton | prefill | 140/144 | 0.939× | 0.644× | 81 | 否 |
| Triton | decode | 168/168 | 1.132× | 0.769× | 21 | 否 |

加速比为各自运行中最快正确兼容原生外部 provider 时间 / 自有实现时间；Triton decode 的平均达标也不能抵消最差项回退。**本版没有达到整体性能目标。** 正确性、编译完成和性能判定分别审查，不能从合格子集或单一平均数推导全面领先。

- [full-completed.json](full-completed.json)：TileLang 离线调优 312 项，保留原 [full-tuned.json](full-tuned.json) 的 286 行，并在重启后于 11 月 10 日续测 26 项。`resume` 保存原文件哈希与新段环境。原检查点 SHA256 为 `ae2804258ff15b209833ce4bf80d9597f24793508a73c4ff5e2f9d70674355ff`，没有改写。
- [paired-triton.json](paired-triton.json)：同软件和硬件键、相同五轮协议的 Triton 312 项；与 TileLang 分段记录不是温度相同的交错测量。
- [comparison.md](comparison.md)、[comparison.csv](comparison.csv)、[comparison.json](comparison.json)：按 case ID 匹配的配对报告，包含原始输入字节 SHA256。上表的合格外部子集与 raw `summary` 中全部可比较项的描述性平均有区别，见结果章说明。
- [default-smoke-comparison/](default-smoke-comparison/)：8 项普通默认接口对照，配置模式为 `public_default`。全量为 `offline_tuned`，API 不自动加载离线胜出表；二者不得混用。
- [release-analysis.json](release-analysis.json)：原始 JSON/CSV 的 CPU 聚合，保存 provider 状态、调优候选错误、回退排序、遥测范围、采样空档和来源哈希。

每种后端均有 1,968 个调优候选：1,824 个通过数值与计时检查，144 个因保守 SM120 共享内存预算在配置检查中被拒绝。失败没有从原始候选列表删除。完整计时包括 decode 全部 split 与 merge；准备、首次调用（缓存缺失时可能含 JIT）与预热独立记录，首次调用不是纯编译器时间。

## 验证与测量限制

- [verification-final.json](verification-final.json) 汇总本轮验收及来源哈希；[browser-checks-final.json](browser-checks-final.json) 记录标签同步、跨页/刷新、URL 路线和历史深层链接检查。两个复制按钮均显示成功，Browser Use 虚拟剪贴板无法回读/粘贴，复制内容的往返验证仍未完成，不计为已通过。
- [anchor-checks-final.json](anchor-checks-final.json) 严格构建原 Git 快照，再与当前 HTML 比较，20 页的 160 个旧标题锚点全部存在。
- [pytest-final.log](pytest-final.log)：CUDA preflight 明确可用后运行完整测试，**557 passed**，覆盖真实 GPU 数值、分页与 CUDA Graph 重放。旧 524 项记录仍在 [pytest-all.log](pytest-all.log)。
- [cpu-final.log](cpu-final.log)：干净 `.venv-docs`，无 Torch/Triton/TileLang，**99 passed, 19 skipped, 4 deselected**。这些跳过不计为 GPU 通过。
- [pytest-final-sandbox.log](pytest-final-sandbox.log)：沙箱内设备不可见时的 **183 passed, 374 skipped**，作为诊断保留；它不是最终 GPU 验收。
- [compiler/README.md](compiler/README.md)：9 个 kernel 的源码、指令与资源证据及其来源限制。导出本身没有做数值或性能验证，不能替代上述测试。

遥测 [full-tuned-telemetry.csv](full-tuned-telemetry.csv)、[full-completed-telemetry.csv](full-completed-telemetry.csv)、[paired-triton-telemetry.csv](paired-triton-telemetry.csv) 包含编译、验证、空闲和主机挂起的影响。原段有 1 行截断，原段 / Triton 最大采样空档分别为 41949.637 / 2608.897 秒；没有填补缺失时间或改写原 CSV。温度范围分别为原段 28–68°C、续测 40–53°C、Triton 31–73°C。未锁频或证明设备独占，功率上限为 `[N/A]`；软件硬件键一致不能消除这些限制。源码层面的优化假设与最差 case 分析见[结果章](../../docs/15-results.md)，当前没有 Nsight 的瓶颈采集证据。
