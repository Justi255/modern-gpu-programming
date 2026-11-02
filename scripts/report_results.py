"""Rebuild CSV tables and the book's measured-results section, without CUDA.

Run from the repository root after the complete benchmark finishes:
python scripts/report_results.py --input results/full-tuned.json
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from fa5060.benchmark import summarize_results
from fa5060.workloads import full_suite


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def telemetry_summary(path):
    with path.open(encoding='utf-8') as stream:
        samples = list(csv.DictReader(stream, skipinitialspace=True))
    result = {'samples': len(samples)}
    for field, label in [('power.draw', 'power_watts'), ('clocks.current.sm', 'sm_clock_mhz'),
                         ('temperature.gpu', 'temperature_celsius'), ('memory.used', 'memory_mib')]:
        values = []
        for row in samples:
            for key, value in row.items():
                if key.split()[0] == field:
                    try:
                        values.append(float(value.split()[0]))
                    except ValueError:
                        pass
        result[label] = {'minimum': min(values), 'maximum': max(values)} if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/full-tuned.json'))
    parser.add_argument('--chapter', type=Path, default=Path('docs/15-results.md'))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    cases = report['cases']
    if report['suite'] != 'full' or len(cases) != len(full_suite()):
        raise ValueError('Only the complete frozen full suite can generate the release chapter')
    assert report['summary'] == summarize_results(cases, full_suite(), suite='full')
    directory = args.input.parent
    telemetry = telemetry_summary(directory / 'full-telemetry.csv')
    (directory / 'telemetry-summary.json').write_text(json.dumps(telemetry, indent=2) + '\n')
    def interval(name):
        value = telemetry[name]
        return f"{value['minimum']:g}–{value['maximum']:g}" if value else '不可用'
    rows, providers, configs = [], [], {}
    for case in cases:
        ours = case['providers']['ours']
        best = case['providers'].get(case['fastest_baseline'], {})
        row = {name: case[name] for name in ('id', 'phase', 'batch', 'kv_len', 'heads_q',
                                            'heads_kv', 'head_dim', 'dtype', 'layout')}
        row.update(ours_gpu_us=ours.get('gpu_us_p50'),
                   fastest_baseline=case['fastest_baseline'], best_gpu_us=best.get('gpu_us_p50'),
                   speedup=case['speedup_vs_fastest'], correctness_passed=case['correctness_passed'],
                   baseline_coverage=case['mandatory_baselines_covered'])
        rows.append(row)
        configs[case['id']] = ours.get('selected_config')
        for name, provider in case['providers'].items():
            providers.append({'id': case['id'], 'provider': name,
                              'status': provider['status'], 'reason': provider.get('reason', ''),
                              'gpu_us': provider.get('gpu_us_p50'),
                              'eager_event_us': provider.get('eager_us_p50'),
                              'eager_wall_us': provider.get('eager_wall_us_p50')})
    write_csv(directory / 'cases.csv', rows)
    write_csv(directory / 'providers.csv', providers)
    regressions = [row for row in rows if row['speedup'] is not None and row['speedup'] < .95]
    # Keep the same header even if a future release has no regressions.
    with (directory / 'regressions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(regressions)
    (directory / 'selected-configs.json').write_text(json.dumps(configs, indent=2) + '\n')

    environment = report['environment']
    summary = report['summary']
    text = [
        '本次发布在实际 RTX 5060 Laptop GPU 上完成冻结的 **312 项**测试：144 项 prefill、168 项 decode；每项保留五轮测量。软件为 Python 3.13、'
        f"PyTorch {environment['packages']['torch']}（CUDA {environment['cuda_build']}）、"
        f"Triton {environment['packages']['triton']}、FlashInfer {environment['packages']['flashinfer-python']}、"
        f"FA4 {environment['packages']['flash-attn-4']}。设备为 26 SM、32 MiB L2、每 CTA opt-in 99 KiB 共享内存，CUDA 可用总显存约 7682 MiB。",
        '',
        ('**本次满足冻结集合的性能门槛。** ' if summary['target_met'] else '**本版没有达到约定的 SOTA 性能目标。** ')
        + '下表描述所有 case 对逐项最快正确兼容基线的延迟比；正确性失败或基线执行错误还会独立阻止达标结论。不能把局部胜出概括为全面领先。',
        '',
        '| Phase | 完成/冻结项数 | 几何平均加速 | 最差加速 | 回退超过 5% | 覆盖与正确性完整 | 达标 |',
        '|---|---:|---:|---:|---:|---|---|',
    ]
    for phase in ('prefill', 'decode'):
        values = summary[phase]
        count = sum(row['phase'] == phase for row in regressions)
        text.append(f"| {phase} | {values['completed_cases']}/{values['expected_cases']} | "
                    f"{values['geometric_mean_speedup']:.3f}× | {values['worst_speedup']:.3f}× | {count} | "
                    f"{'是' if values['coverage_complete'] else '否'} | {'是' if values['target_met'] else '否'} |")
    text.extend([
        '',
        '这里报告的是**离线逐 case 固定网格选配置、再单独五轮测量**的结果。所有候选、失败和最终 `selected_config` 均保留在 JSON。公开 `prefill`/`decode` 默认配置不自动加载这张表，因此本表不是默认调用的性能保证。读者可按相同 shape/dtype/layout 显式传入 `config` 与 decode 的 `num_splits`；换设备或软件后应重新验证。配置搜索使用同一冻结集合，不能视作独立保留集上的泛化证据。',
        '',
        '### 数据、计时与复现边界',
        '',
        '- {download}`完整原始结果 <../results/full-tuned.json>`：逐项输入、七个 provider、全部候选、误差与多轮数据。',
        '- {download}`完整 case 表 <../results/cases.csv>`、{download}`所有回退项 <../results/regressions.csv>`、{download}`provider 状态与接口时间 <../results/providers.csv>`。',
        '- {download}`冻结清单与源码 SHA256 <../results/manifest-full.json>`、{download}`离线所选配置 <../results/selected-configs.json>`、{download}`每秒 GPU 遥测 <../results/full-telemetry.csv>`。',
        '- {download}`原生 provider 实现与已安装源码哈希 <../results/provider-sources.json>`：FA4 的实际路径为 `FlashAttentionForwardSm120`；Torch 分别强制 Flash/cuDNN/Efficient，FlashInfer 分别请求 FA2/FMHAv2。',
        '- {download}`8 项 smoke <../results/smoke.json>` 用于路径检查，不承担完整矩阵结论。',
        '',
        '主指标把完整算子多次捕获进同一个 CUDA Graph，每轮重放一次、除以图内调用数，再取五轮中位数；decode 包含 split 与 merge。规划、JIT、参考和校验不计入，单独的 eager event/墙钟数值保留接口分配与提交成本。输入与页表固定，provider 顺序按公开种子打乱。FlashInfer dense FMHAv2 公开包装内部的 KV 打包计入两种计时，不能与预先打包的私有入口混淆。',
        '',
        '未锁定频率或风扇模式；外接电源状态未自动采集，也未关闭桌面应用或隔离设备。因此这是公开条件下的本机观测，不能声称实验室独占测量。遥测提供功耗、时钟、温度和显存序列，功率上限若为 N/A 则保持缺失，不推测数值。',
        '',
        f"全程取得 {telemetry['samples']} 个每秒采样：功耗 {interval('power_watts')} W，SM 时钟 {interval('sm_clock_mhz')} MHz，温度 {interval('temperature_celsius')} °C，已用显存 {interval('memory_mib')} MiB。范围包含预热、JIT、参考与测量阶段，不能把峰值功耗直接视作任一内核的平均功耗。",
        '',
        '### 最差项与后续实验',
        '',
        '| Phase | B | L | Hkv | D | 类型/缓存 | 本实现 μs | 最快基线 | 基线 μs | 加速 |',
        '|---|---:|---:|---:|---:|---|---:|---|---:|---:|',
    ])
    for phase in ('prefill', 'decode'):
        worst = sorted((row for row in rows if row['phase'] == phase), key=lambda row: row['speedup'])[:5]
        for row in worst:
            text.append(f"| {phase} | {row['batch']} | {row['kv_len']} | {row['heads_kv']} | {row['head_dim']} | "
                        f"{row['dtype']}/{row['layout']} | {row['ours_gpu_us']:.3f} | {row['fastest_baseline']} | "
                        f"{row['best_gpu_us']:.3f} | {row['speedup']:.3f}× |")
    text.extend([
        '',
        '先用这些固定 case 检查因果裁剪、块尺寸和 split 数的敏感性，再收集寄存器、spill、内存流量和占用证据。本机没有安装 Nsight Systems/Compute，当前没有足够 profiler 证据把所有回退归因于某一种硬件瓶颈。prefill 仍按查询头独立扫描 KV，decode 的分段与合并也有额外工作；它们是可检验的优化方向，不能替代诊断结论。',
        '',
        '### 基线状态',
        '',
        '| Provider | 正确并计时 | 明确不支持 | 数值失败 | 执行错误 |',
        '|---|---:|---:|---:|---:|',
    ])
    for name in sorted(cases[0]['providers']):
        counts = Counter(case['providers'][name]['status'] for case in cases)
        text.append(f"| {name} | {counts['ok']} | {counts['unsupported']} | {counts['incorrect']} | {counts['error']} |")
    failures = [p for p in providers if p['status'] in ('incorrect', 'error')]
    text.extend(['', f'共 {len(failures)} 条 provider 失败记录，全部留在原始文件中；失败者不参与最快合法基线选择，也不被静默删除。这些数值失败指未满足本书冻结的误差合同，原始 RMS 误差也同时保留，便于读者按自己的数值预算审查。'])
    if failures:
        text.extend(['', '其中的例子：', ''])
        for item in failures[:5]:
            text.append(f"- `{item['id']}` / `{item['provider']}`：{item['reason'][:400]}")
    text.extend([
        '',
        '初次 dense FMHAv2 分离布局错误及固定工作区不足的中间结果保存在 `results/diagnostics/`，不合并进本表。扩大工作区后重新完整执行；没有缩小测试形状或放宽 FP16 0.003 / BF16 0.02 的逐元素绝对与相对阈值，另检查归一化 RMS 误差。',
        '',
        '### 教学分解实验与验证',
        '',
        '基础、GEMM 和小注意力分别保存于 {download}`basics.json <../results/basics.json>`、{download}`gemm.json <../results/gemm.json>` 和 {download}`attention.json <../results/attention.json>`。这些普通提交 event 数据包含主机供给空隙，用于逐章学习，不与上表图内延迟直接相除。三组共 26 条记录均先通过数值检查，涵盖尾块、外层 stride、三种 GEMM 调度、变长 GQA、空 KV 与随机页表。',
        '',
        '完整测试与文档构建命令及结果另见 {download}`验证记录 <../results/verification.json>`。CPU CI 不把跳过 GPU 测试视作 GPU 通过；真正的设备验证在本机执行。',
    ])
    chapter = args.chapter.read_text()
    start = '<!-- GENERATED_RESULTS_START: root integrates actual probe/benchmark evidence here. -->'
    end = '<!-- GENERATED_RESULTS_END -->'
    before, tail = chapter.split(start)
    _, after = tail.split(end)
    args.chapter.write_text(before + start + '\n\n' + '\n'.join(text) + '\n\n' + end + after)
    checksum = hashlib.sha256(args.input.read_bytes()).hexdigest()
    print(json.dumps({'cases': len(rows), 'regressions': len(regressions), 'input_sha256': checksum}))


if __name__ == '__main__':
    main()
