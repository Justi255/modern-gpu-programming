# 中间失败记录

这里的文件用于复查错误，不作为最终性能表的数据源。

- `smoke-separate-layout-rejected.json`：最初直接使用分离 Q/K/V 的 dense FMHAv2 入口，数值失败。随后改为受支持的公开 ragged 包装，KV 打包计入计时；数值阈值又收紧，重新运行 smoke。
- `full-workspace128-interrupted.json`：第一次 full 调优运行的原子检查点，共保存 166 项。长上下文 FlashInfer FA2 的 FP32 部分输出与归一化统计超出固定 128 MiB 工作区，运行被主动中断。
- `manifest-workspace128.json`、`full-workspace128-telemetry.csv`：该次中间运行的冻结形状、源码哈希与遥测。

最终数据为上一级 `full-tuned.json`：扩大工作区后，从第一个 case 重新运行冻结的全部 312 项。没有合并这份中间结果的时间，也没有删除不利 case。最终仍超出数值阈值的 provider 保留 `incorrect` 状态，并阻止性能达标结论。详情见教程第 15 章和调试附录。
