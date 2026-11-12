# FlashInfer FMHAv2 FP16 accuracy investigation

Date: 2026-11-11. Scope: CPU-only inspection of this checkout, installed
FlashInfer `0.7.2.post1`, and existing generated JIT sources. No GPU calls,
package changes, installation, or commits were performed for this investigation.

## Finding and evidential limit

The installed native FMHAv2 FP16 path selects FP16 tensor-core accumulation,
including FP16 encoding of the BMM1/softmax scales. The benchmark's explicit
`backend="fmha_v2"` reaches that path even though the wrapper's plan parameter
`use_fp16_qk_reduction` defaults to `False`. That parameter is not forwarded to
the FMHAv2 entry point, whose public signature has no precision switch.

This is a confirmed implementation-mode mismatch with an expectation of FP32
reduction; it is a strong explanation for the four small FP16/D128 numerical
failures. A native-versus-FP32-accumulation GPU comparison is still needed to
establish that it accounts for every failing element. No claim is made that
the proposed corrected mode already passes or that native failures disappear.

The input layout, causal convention, page offsets, and attention scale in the
adapter match the installed public API. No evidence of a layout or scale
*adapter* bug was found. The native provider and its four failure records must
be retained. Reference math, thresholds, frozen cases, dtype, seed, and inputs
must remain unchanged.

## Existing failures

Source: `results/tilelang/full-completed.json`, seed `20261102`.
All are causal prefill, FP16, head dimension 128, 32 Q heads.

| Case ID suffix after `prefill-` | Max absolute error | Normalized RMS error |
| --- | ---: | ---: |
| `b1-l512-hq32-hkv32-d128-fp16-dense` | 0.0046386719 | 0.00088240052 |
| `b1-l8192-hq32-hkv32-d128-fp16-dense` | 0.0035095215 | 0.0014981168 |
| `b4-l2048-hq32-hkv32-d128-fp16-dense` | 0.00390625 | 0.0010871076 |
| `b4-l2048-hq32-hkv1-d128-fp16-paged16` | 0.0068359375 | 0.0010918253 |

These fail the existing elementwise check (`atol=rtol=0.003`); the RMS check
alone passes. That is not grounds to change either check.

## Concrete source evidence

All installed-package paths below are relative to
`.venv-tilelang/lib/python3.13/site-packages/flashinfer/`.

- `data/csrc/fmha_v2_run.cu:424-435`: FP16 input gets
  `acc_type=DATA_TYPE_FP16` and `force_fp32_acc=false`. The source explicitly
  states "FP16 kernels use FP16 accumulation."
- `data/csrc/fmha_v2_run.cu:189-213`: BMM1 and softmax scales use `acc_type`;
  for the native FP16 mode they are encoded in FP16 too. BMM2 scale remains
  input dtype for FP16/BF16. The requested `D**-0.5` scale is correct, but its
  native representation is lower precision.
- `data/csrc/fmha_v2/fmha/fragment.h:759-773`: the selected
  `Ampere_hmma_fp16_traits` accumulator emits
  `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`.
- `prefill.py:5022-5034`: ragged `plan()` defaults
  `use_fp16_qk_reduction=False`.
- `prefill.py:5538-5560`: automatic FMHAv2 selection requires
  `use_fp16_qk_reduction=True`. Explicit backend selection bypasses this
  automatic-selection guard.
- `prefill.py:6092-6137`: the ragged FMHAv2 `run()` branch packs K/V with
  `torch.stack([k,v], dim=1)` and forwards to `trtllm_fmha_v2_prefill()` with
  no precision argument. Merely passing `False` explicitly to `plan()` cannot
  change its accumulation mode.
- `prefill.py:8271-8297`: public `trtllm_fmha_v2_prefill()` signature has no
  FP32-accumulation option. `skip_softmax_threshold_scale_factor` defaults to
  zero, so sparse skipping is disabled.
- `prefill.py:8308-8330`: dense `CONTIGUOUS_Q_KV` and separate paged
  `Q_PAGED_KV_NHD` formats match `src/fa5060/benchmark.py:242-300`.
- `src/fa5060/reference.py:35-60`: reference promotes original input values
  to FP32 before QK, softmax, and PV, then casts output back to original dtype.

## Existing binary/registry availability

There is **no ready FP16-input/FP32-accumulation registry** in the observed
native JIT cache. Simply changing `force_fp32_acc` to true will fail dispatch.

For each of these generated directories under
`/home/justi/.cache/flashinfer/0.7.2.post1/120f/generated/`:

- `trtllm_fmha_v2_contiguous_q_kv_fp16_fp16/`
- `trtllm_fmha_v2_q_paged_kv_nhd_fp16_fp16/`

CPU inspection found 26 CU files (24 generated kernel units plus runner and
binding), 24 `!force_fp32_acc` gates in `fmha_v2_api.h`, zero positive
`&& force_fp32_acc` gates, and zero CU files using `hmma_fp32_traits`.
The dense registry's first D128 dispatch is at `fmha_v2_api.h:189-215`.
No `fp16_fp32`/`hmma_fp32` named cache artifacts or matching installed cubin
wheel were found by file inventory. This observation concerns this installation,
not every possible upstream wheel.

The installed source does include real FP16-input/FP32-accumulation FMHAv2
implementations:

- `data/csrc/fmha_v2/fmha/traits.h:380`: `Ampere_hmma_fp32_traits` has
  uint16 input/output and float accumulators.
- `data/csrc/fmha_v2/fmha/fragment.h:814-845`: its MMA uses
  `f32.f16.f16.f32` instructions.
- `jit/attention/fmha_v2/utils.py:37-95`: `fp16_fp32` maps to
  `hmma_fp32_traits`, `DATA_TYPE_FP16`, 2-byte inputs, and `fp16_t` output.
- `jit/attention/fmha_v2/fmha_library.py:338-349` accepts `fp16_fp32`
  standard flash-attention specs, and `:796-801` dispatches them when
  `force_fp32_acc` is true.

The default `generate_jit_sources()` uses only `dtype_values=[input_dtype]`
at `fmha_library.py:1267`; the normal module factory maps Torch FP16 to
`"fp16"` at `jit/attention/modules.py:2561-2569`. Neither enables the source
variant above for the existing public FP16 call.

## Transparent independent source-build proposal

This proposal changes accumulation mode inside the same installed FMHAv2
implementation. It must be labeled as a corrected FMHAv2 precision mode,
with source hashes/patch disclosed, while original native measurements and
failures remain available. It has not been compiled or run yet.

1. Use a separate generated-source directory, binary URI, and metadata, such
   as `fa5060_fmha_v2_fp32acc_{layout}`. Do not overwrite `site-packages` or
   native cache artifacts.
2. For each of `CONTIGUOUS_Q_KV` and `Q_PAGED_KV`, generate D64 and D128
   specs with installed `fmha_library.generate_kernel_spec()` using
   `sm=120`, `dtype="fp16"`, `output_dtype="fp16"`, `alibi=False`,
   `return_softmax=False`, `enable_attn_logit_softcapping=False`, and the
   appropriate `InputLayout`. Then use
   `dataclasses.replace(spec, dtype="fp16_fp32")`.
   Generating geometry first as FP16 preserves its supported cp.async and
   tiling decisions; passing `fp16_fp32` directly into that factory is not
   supported by its geometry branches.
3. Use installed `utils.encode_name()`, `fmha_library.get_kernel_code()`,
   and `fmha_library.get_api_code()` to generate kernel units and registry.
   The code remains FMHAv2; no custom attention algorithm or provider is
   substituted.
4. Make a private build copy of installed `data/csrc/fmha_v2_run.cu`. Change
   only its `acc_type` assignment to `DATA_TYPE_FP32` and
   `force_fp32_acc` assignment to `true`. Keep dtype detection/output dtype,
   masks, page addressing, scale values, and all other runner code unchanged.
   Copy `fmha_v2_jit_binding.cu` unchanged. Publish these small transformations
   or patch and source hashes, rather than silently modifying installed code.
5. Build with `flashinfer.jit.core.gen_jit_spec(...).build_and_load()` using
   the include paths and CUDA flags from `jit/attention/modules.py:2595-2635`.
   Restrict to the actual SM120 target as the native build does. Preserve
   native fast-math flags for the first comparison so accumulation is the
   intended changed variable.
6. Call the exact same `module.run` signature as
   `prefill.py:8567-8596`. Both Q/K/V and output stay FP16. Keep dense K/V
   packing in both eager and graph timing boundaries, and preserve the native
   paged input/page-table handling. Use the corrected mode consistently across
   relevant FP16 cases, not only the four failures. Do not alter BF16 inputs.

Static code generation was successfully exercised **in memory, CPU only**, by
loading the pure generation utilities with GPU imports removed. All four
generated specs passed `is_kernel_spec_valid()`; D64 sources contained three
`Ampere_hmma_fp32_traits` references and D128 six references. Both registries
had two positive precision gates and zero negative gates, retaining
`DATA_TYPE_FP16` dispatch. This checks generator support only, not CUDA
compilability or numerical correctness.

Minimal initial build cost is two kernel CU files plus runner and binding per
layout, eight translation units total for both layouts. Native all-feature
generation has 26 per layout. Actual compile latency and register/shared-memory
behavior are unmeasured; reserve time for a clean compile and inspect its output.
Use a distinct cache identity to prevent native/modified binaries from mixing.

## GPU verification for the main agent

First replay the four native failures with unchanged validation. This command
is provided for the GPU-owning main agent and was not run here:

```bash
.venv-tilelang/bin/python -B - <<'PY'
import json
import torch
from fa5060.benchmark import _flashinfer_provider, _validate_output
from fa5060.reference import attention_reference
from fa5060.workloads import AttentionCase, make_inputs

report = json.load(open('results/tilelang/full-completed.json'))
for row in report['cases']:
    if row['providers']['flashinfer_fmha_v2']['status'] != 'incorrect':
        continue
    case = AttentionCase.from_dict(row)
    inputs = make_inputs(case, seed=row['seed'])
    reference = attention_reference(inputs.q, inputs.k, inputs.v,
        causal=case.causal, kv_lens=inputs.kv_lens,
        block_table=inputs.block_table)
    provider = _flashinfer_provider(case, inputs, 'fmha_v2')
    actual = provider.graph_call()
    torch.cuda.synchronize()
    try:
        result = _validate_output(actual, reference, case.dtype)
    except Exception as exc:
        result = {'status': 'incorrect', 'reason': str(exc)}
    print(case.case_id, result, flush=True)
    del actual, reference, provider, inputs
PY
```

After building the separate mode, compare native and corrected output on each
same `AttentionInputs` object against the same unchanged reference. Record
elementwise-failure count, worst element/query/head, max absolute error, and
normalized RMS. Check eager, preallocated, captured, and replayed outputs.
Then run the frozen full suite with both native and corrected observations,
same dtype/seed/workloads/thresholds, preserving every native failure. Report
corrected-mode qualification separately until the benchmark contract explicitly
accounts for that transparent precision choice. Re-measure native and corrected
performance; no speed or accuracy improvement is assumed from static evidence.
