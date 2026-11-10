### TileLang 与 Triton 的配对测量

配置模式：`public_default`；每项 5 轮。按 case ID 匹配 8 项；冻结清单匹配：是。清单匹配只说明 case ID 齐全，正确性和外部基线覆盖另行检查。

TileLang / Triton 是两种后端的延迟比，越小越快；直接加速是 Triton / TileLang，越大越快。外部基线几何平均仅纳入强制基线覆盖完整且全部数值检查通过的项，分别取每次运行中最快的正确、兼容原生 provider，与后端间比值分开统计。

| Phase | 有效配对 / 全部 | 两后端正确性完整 | TileLang / Triton 几何平均延迟比 | TileLang 对 Triton 几何平均加速 | 最差直接加速 | Triton 对最快外部基线加速 | TileLang 对最快外部基线加速 |
|---|---:|---|---:|---:|---:|---:|---:|
| prefill | 4 / 4 | 是 | 1.193 | 0.838 | 0.762 | 1.021 | 0.817 |
| decode | 4 / 4 | 是 | 11.033 | 0.091 | 0.062 | 0.408 | 0.037 |

| Phase | 后端 | 强制基线覆盖 / 全部 | 基线覆盖完整 | 外部几何平均有效项 / 全部 |
|---|---|---:|---|---:|
| prefill | triton | 4 / 4 | 是 | 4 / 4 |
| prefill | tilelang | 4 / 4 | 是 | 4 / 4 |
| decode | triton | 4 / 4 | 是 | 4 / 4 |
| decode | tilelang | 4 / 4 | 是 | 4 / 4 |

失败和缺失项保留在 JSON/CSV 中，几何平均只使用有效配对；有效项数不足时不能据此声称完整集合更快。逐元素 FP16 0.003 / BF16 0.02 和归一化 RMS 合同保持不变。图计时包含全部 split 与 merge，准备、首次调用（含缓存缺失时的 JIT）、预热单独记录，不计入稳定态延迟。

同一软件和硬件键并不代表设备独占、锁频或相同温度；功耗、时钟和温度应检查两份原始记录及遥测。

- `triton` 原始文件：`results/tilelang/paired-triton-smoke-default.json`；SHA256 `bc80c9062f5548a2a696d5a01521c5ece11110f7bea1363ddc3c0e6058d29409`。
- `tilelang` 原始文件：`results/tilelang/smoke-default.json`；SHA256 `f8942c3512892c2b4496aa18214972a5b3145477f46d2e5de4e5dcf44ef510dc`。
