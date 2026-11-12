# SM120 attention 优化批次：2026-11-11

这是发布基线 `2358dbc5d41058009267ff45fd13611b69647072` 之后的独立优化记录。
冻结用例仍为 312 项，seed=20261102，FP16/BF16 阈值及 FP32/RMS 检查不变。
当前目录包含失败试验，不能把所有文件当作合格性能结果。

第一轮固定策略已完成：`full-optimized-resumed.json` 保留312项，自己的输出
312/312正确。prefill合格子集140项的GM为1.075196×、worst为0.792248×；
decode168项的GM为1.071357×、worst为0.517778×。**尚未达到约定门槛。**
prefill另4项原生FMHAv2数值错误仍使完整合同比较不通过。
完整统计及24个最差项见 `full-optimized-analysis.json`；原始summary对全部144项
prefill的比例GM为1.073996×，不表示所有外部provider均正确。

`full-optimized.json` 是未改写的135项检查点，原进程停滞后终止；没有取得
可用调用栈，不推断其原因。续测文件追加177项并记录分段环境。结束后验证
`full-optimized-source-freeze.json` 的22个源码SHA全部一致。

## 验证与复现

本轮 `all-tests.log`：649 passed。`cpu-tests.log`：197 passed、452 deselected。
`racecheck-candidates.log`：4 passed，0 hazards；只覆盖选定的手工同步路径。
未对齐 storage offset 的原版/优化版错误分别保留在 `misaligned-reference.log`、
`misaligned-optimized.log`，修复后的独立验证见 `misaligned-scalar-fallback.log`。

在仓库根目录、安装附录指定环境中执行：

```bash
python -m pytest -q
python -m fa5060.benchmark --backend tilelang --implementation optimized \
  --suite full --rounds 5 --output results/local/optimized-full.json
python scripts/optimization_ab.py \
  --case prefill-b1-l2048-hq32-hkv8-d128-bf16-paged16 \
  --case prefill-b1-l2048-hq32-hkv8-d64-bf16-dense \
  --trials results/optimization-2026-11-11/trials-qasync.json \
  --output results/local/qasync-ab.json
```

`--implementation optimized` 测量预先固定的 metadata 调度，记录为
`explicit_configuration`，可经相同公共接口重放。没有逐项看完结果后手工选最优。
不带此参数保留教学默认；`--tune` 则执行公开的候选网格，另存结果。
所有 decode 测量包含 split 和 merge；GPU Graph、eager 接口与首调分开记录。

## 文件有效范围

- `interleaved-qasync.json`：两个代表性 prefill 的最终局部 A/B，包括全部兼容外部
  provider、同输入、9轮交替顺序、每轮遥测。仅能证明这两个形状的结果。
- `interleaved-segmented-qualified.json`：代表性长分页 decode 的外部对照。
- `interleaved-ab.json`、`interleaved-ordered-reduce.json`、
  `interleaved-async-manual.json`、`interleaved-unmasked-qreg.json`：较早版本的诊断。
  ordered-reduce 在第二个错误 case ID 处停止，已完成的首项有效，不能当完整运行。
- `interleaved-warp-grid.json`、`interleaved-segmented.json`：诊断脚本错误地拒绝
  `external` 变体，外部基线均未测。可比较同轮自有候选，**不能用于 SOTA 比值**。
  脚本已修复，并增加 CPU 回归检查。
- `interleaved-batch-reduce.json`：FP32 packed batched reduction 的编译失败记录。
  候选修复采用本项目局部绕过，未修改 TileLang 安装包。
- `grouped-w1-register.json`：旧脚本漏传 grouped 协议，split>1 部分无效。
  `grouped-w1-register-corrected.json` 才是修复后重测；不可引用错误的旧数字。
- 其余 probe JSON 是固定顺序的调参诊断；频率、温度会变化，避免跨运行绝对时间比值。
- `racecheck-decode.log`：首次选择表达式未选中测试，78 deselected；这是未执行记录，
  不是有效的 sanitizer 验证。有效检查在 `racecheck-candidates.log`。

每份诊断 JSON 保留脚本/源码 SHA；修改后以新文件重测，不覆盖旧证据。
Nsight 的二进制 `.ncu-rep`、cubin、编译缓存不提交，保留可审查的文本。

## 编译与正确性边界

`compiler/sync-review.md` 记录源码、lowering 和资源审查。M16/N32/128线程
segmented 路径每 tile 有4个显式 CTA barrier，warp内归约无额外 named barrier；
更大 N 的 warp 划分不同，当前工厂只在已验证的几何上启用此方法。
独立 shared 槽与 probabilities 区域不重叠。超预算 tile 回退原工厂。

`compiler/fmha-v2-accuracy-research.md` 保存原生 FMHAv2 FP16 精度调查，
`compiler/fmha-precision-implementation.md` 描述独立 FP32-acc 源码构建实验。
原生4项失败不被覆盖。独立构建模式在完成 GPU 验证以前不能计入合格基线。

完整312项结果和 prefill/decode 的 GM、worst 必须分别检查；局部达到1.10×
不能支持全套达标结论。正式结论以完整结果的覆盖、正确性和门槛字段为准。
