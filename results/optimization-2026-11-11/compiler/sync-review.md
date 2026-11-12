# Optimization synchronization/bounds review v2

CPU-only review in `/home/justi/Workspace/cuda/fa`, 2026-11-11. No GPU query,
`nvidia-smi`, adapter creation, kernel launch, numerical GPU test, or sanitizer
was run by this reviewer. TileLang 0.1.15 / apache-tvm-ffi 0.1.12 were lowered
with the production pass flags under an explicit `sm_120` target. Generated
CUDA was independently compiled with NVCC, and temporary CUBINs were inspected
with cuobjdump and removed. `CUDA_VISIBLE_DEVICES` was empty throughout.

## Conclusion

No new producer/consumer race, two-slot alias, live-page bounds error, or split
state protocol error was found in the final restricted M16/N32/128-thread
segmented path. This is a static/compiler conclusion. The main agent must
supply GPU correctness and race-check evidence.

Two concrete scope problems were found during review and were corrected by the
main agent: (1) unrestricted pipeline geometry can exceed the shared-memory
limit for an otherwise usable reference configuration, and (2) the prefill
async factories' page-fast path was not general to unaligned decode splits.
The larger-N segmented implementation was also shown to retain cross-warp
collectives; the main agent restricted this optimization to the geometry with
a verified warp-local mapping.

An input-base alignment concern was reported to the main agent: even outer
strides do not imply an aligned tensor data pointer. The main agent subsequently
added a data-pointer-selected scalar reference fallback in attention.py. This
review does not claim GPU validation of that later fallback or a GPU failure.

## Reviewed source identity and evidence

`results/optimization-2026-11-11/compiler/sync-v2-cpu-review-final-manifest.json`
records hashes before and after the seven-case lowering/compilation run;
the snapshots are equal. Relevant hashes in that run (before the later budget-bound refinement):

| Source | SHA256 |
|---|---|
| `_decode_pipeline.py` | `754af6d3ccb5c5f6b5a0d401f84f49e231b51beedae0ebfc0459ff19044f20ff` |
| `_prefill_async.py` | `bb1f9e8bc5a3fedc7b79c84baaefb0d9b898fd85eda1857618383fa774f47c48` |
| `_prefill_qasync.py` | `616ce00a8603e322be32e5228311fecc8777434524a439913f679113da5098ee` |
| `_prefill_qregister.py` | `bb54103385a8afdb429edc5378d093bc3710552ac5f04f94f0c6a374abcc4c7a` |
| `_prefill_register.py` | `b2551dc9a72559660741b23c986c6b43a64ef9f67b873c7098e82f6985d9fcc9` |

A subsequent three-case budget-guard run is recorded in
`sync-v2-cpu-review-budget-manifest.json`; source hashes again stayed equal
throughout. `_decode_pipeline.py` was then
`19be47f8ebac8f20948dd987b620d5a39ffa8c78dbf37daffa053a7b340ed050`.
Segmented BN32, ordered BN64 and the BN128/D128 reference fallback produce
byte-identical CUDA to their seven-case-run versions. The run's attention.py
hash is `6c6cf035d947565c8714e0de5859c362e9f7e878021384fdca75a19ce33cffea`;
scalar_memory was false for these three cases.

Evidence prefixes under the compiler directory:

- `sync-v2-cpu-review-budget-*`: after refining the shared-workspace bound;
  unchanged generated CUDA for all three compared geometries.
- `sync-v2-cpu-review-final-*`: after the main agent added shared-budget and
  factory-scope guards and restricted segmented geometry. Each successful case
  has generated CUDA, host/device TIR, and measured NVCC resource text.
- `sync-v2-cpu-review-target-*`: earlier candidate geometry/resource review,
  including the unrestricted BN64 segmented path and qregister.
- `sync-v2-cpu-review-config-*`: original large-shared comparison, BM32 fallback,
  and unsupported 8-warp/BM16/BN32 compiler rejection.
- `sync-v2-cpu-review-static-checks.json`: CPU enumeration of shared-store
  ownership, causal branch arithmetic, ordered integer transform, and a CPU
  storage-offset metadata example.
- The unqualified `sync-v2-cpu-review-manifest.json` is an initial reviewer
  setup failure (missing Target context), not evidence of a candidate defect.
  All subsequent runs explicitly enter Target context and succeed.

Reproduction scripts were created in `/tmp/fa5060-sync-v2-cpu-review*.py` and
`/tmp/fa5060-sync-v2-static-checks.py`; no project source was changed by this
reviewer. Binary resource measurements are from a fresh temporary CUBIN of the
shown CUDA, not from a GPU-loaded module.

## Synchronization and physical shared-memory layout

The retained segmented branch is gated at `_decode_pipeline.py:48` by
BM16/BN32/128 threads. In `sync-v2-cpu-review-final-seg-bn32.cu`:

- Qs starts at byte 0, size 4096; Ks at 4096, size 16384; Vs at 20480,
  size 16384; P at 36864, size 1024; `reduction_shared` at 37888, size 512.
  Device TIR declares one 128-float reduction buffer and 38400 dynamic bytes.
  These allocations are disjoint. The planner did not fold slot 1 over slot 0
  or into Q/K/V/P.
- Max publication writes reduction indices 0..63; sum publication writes
  indices 64..127. Generated CUDA lines 159–170 and 215–226 show the +64
  float offset and the barriers before peer reads. CPU enumeration confirms
  precisely one producer per address in each slot for all 128 threads.
- Local score reductions use `AllReduce<...,4,1>` and warp shuffles. Despite
  the `NamedBarrier<128>` template argument, these paths do not execute a
  named barrier: reduce.h uses shared/barrier recursion only at offset>=32.
- The four explicit CTA barriers per tile are loop top, max publication, sum
  publication, and P publication. No cross-warp helper barrier remains in this
  geometry. This is a substantial change from the old `pipeline-sync-review.md`
  which described scalar cross-warp reductions, and its 18 barriers/tile must
  not be reused for this candidate.

Ordering argument:

1. The loop-top `wait_group(0)` completes each thread's current-slot async
   copies. The following full CTA barrier publishes those stores plus the
   synchronous Q and tail stores before any QK read.
2. The same barrier retires previous PV readers before the next-slot prefetch
   overwrites the slot consumed two iterations earlier. Current QK/PV and the
   next prefetch use different ring slots.
3. Max and sum use different reduction slots. A fast warp may start writing
   the sum slot while a slow warp finishes reading max; they do not alias.
   Before either slot is reused next iteration, the top CTA barrier retires
   every preceding shared read. The extra P barrier also occurs before reuse.
4. All P stmatrix stores precede the explicit P barrier, followed by PV
   ldmatrix reads. The next loop-top barrier protects P reuse. The final loop
   iteration has no later P writer requiring another barrier.
5. Empty blocks execute no shared consumer, so no special publish barrier is
   necessary when length is zero or the split starts beyond the live length.

For the earlier unrestricted BN64 segmented case, generated CUDA lines 167
and 227 contain `AllReduce<...,128,64>` in four-iteration loops. Those helpers
execute sixteen named barriers per tile, in addition to four explicit CTA
barriers, and use separate workspaces at bytes 72192 and 72704. No alias was
found, but the pure-warp claim did not extend to this geometry. The final source
therefore uses ordered batched max/scalar batched sum for BN64; the final CUDA
contains two explicit plus eight helper CTA barriers per tile.

## Bounds, zero length, mask fast path, and split protocol

`_decode_pipeline.py:105` and `:145` retain both full-tile and page-alignment
conditions. For page64/BN32, `bn % page != 0` is constant, so lowering removes
cp.async entirely and uses the synchronous live-token path. Final public-valid
split8 cases were compiled for page64/capacity192/chunk24 and
page16/capacity208/chunk26. The generated table loads stay inside the exact
live-token predicates, and page/offset arithmetic uses each logical token.
Non-page-aligned starts cannot select the page-fast path.

The synchronous path zeroes both K and V for inactive tokens. This is necessary
for poisoned V tails because P=0 does not suppress `0*NaN`. `stop=min(length,
lo+chunk)` and `blocks=ceildiv(max(stop-lo,0),bn)` keep logical reads within the
live split. Live physical page indices remain the caller contract; manual
bounds legalization does not independently validate bad metadata.

When length is zero or lo>=length, acc=0, denom=0 and max=-inf remain initialized.
With one split, output is 0/max(0,1e-30)=0. With multiple splits, the factory
writes FP32 unnormalized acc=0, maxima=-inf, sums=0. `_merge_splits` keeps its
all-minus-infinity guard and denominator clamp, so the empty-state protocol is
unchanged. Source arrays and stores remain FP32; no reduced-precision state was
introduced.

The uint32 transform is monotonic for non-NaN FP32 values, including infinities;
CPU sampling of 9976 non-NaN bit patterns verified numeric ordering and exact
bitwise inversion. NaN-input equivalence was not claimed. The lowering emits
unsigned batch=2 max plus scalar batch=2 sum, with no missing float2 declaration
or NVCC error in the cases examined.

The unmasked causal branch at `_prefill_register.py:108` and
`_prefill_qregister.py:113` requires `start+bn<=stop` and
`start+bn<=bx*bm+length-sq+1`. The latter is precisely the inclusive causal
boundary of the first query row. Every later row has an equal or later boundary;
therefore all original per-element predicates are true. CPU enumeration checked
1141 admitted tiles over short, ragged, negative-right-alignment and tail-query
cases. Invalid query rows may compute internal results but remain excluded from
output stores.

## Prefill async/register layouts and resource budget

`_prefill_async.py` and `_prefill_qasync.py` retain FullRow QK/PV policies. The
same warp row partition owns the score/P registers and PV accumulator. Generated
CUDA contains register `probs`, no shared P allocation, no P publication barrier,
and no intermediate cross-warp P redistribution. Async Q is shared; qasync Q
remains a per-thread register array across the KV loop. The loop-top wait/barrier
still protects both KV slots; a shared-Q write is also published by the first
such barrier in async. Neither factory needs a further barrier for register P.

The main agent added `grouped or splits!=1` fallback at line 20 in both async
factories after the audit found that `lo//page` without `lo%page` checking was
not correct for arbitrary private decode splits. Final grouped/split4 lowering
for both factories returns production reference flags and byte-identical CUDA.
Public prefill already uses splits1/lo0; therefore this guard closes the private
factory contract without changing its intended public async path.

All resource cases below are FP16, paged, 128 threads. STACK=LOCAL=0 in each
measured temporary CUBIN. Dynamic bytes come from lowered device TIR, static
bytes from cuobjdump; they are added rather than confused with each other.

| Case | REG | Dynamic shared B | Static shared B | Total B |
|---|---:|---:|---:|---:|
| Final segmented M16/N32/D128 | 91 | 38400 | 1024 | 39424 |
| Final ordered M16/N64/D128 | 116 | 73728 | 1024 | 74752 |
| Async M64/N64/D128 | 188 | 81920 | 1024 | 82944 |
| Qasync M64/N64/D128 | 212 | 65536 | 1024 | 66560 |
| Qregister M64/N64/D128/stage2 | 214 | 32768 | 1024 | 33792 |
| Qregister M64/N128/D128/stage2 | 254 | 65536 | 1024 | 66560 |

The qregister lowered stage2 paged loop uses one K and one V allocation, with
compiler-inserted load/reuse barriers. Its resources do not imply ping-pong
staging. In particular, BM64/N128/D128 is not a shared-capacity problem for this
private qregister candidate, although the public prefill budget check is
conservative and may reject this geometry before selecting that candidate.
No attempt was made to change that public policy.

## Confirmed issue and remaining caveats

**Fixed public configuration regression.** Before the main agent's guard,
M16/N128/D128/stage1 grouped decode compiled to 140800 dynamic +1024 static bytes
in both dense and paged layouts. The reference compiled to 74752+1024. The new
unconditional KV double buffer therefore made this allowed reference geometry
exceed the SM120 shared-memory budget. `_decode_pipeline.py:27` now checks an
estimated budget and falls back; final lowering confirms the reference flags,
74752 dynamic bytes and the same CUDA SHA256 as the reference case.

**Budget-bound refinement was verified.** The initial guard used
`8*bm*(threads//32)` for reduction storage. For ordered M16/N32/128 threads its
estimate was 39424 bytes, whereas the generated ordered path has two 1024-byte
batch workspaces and measures 40960 total bytes, 1536 more. No inspected
geometry crossed 96KiB because of this discrepancy. The main agent changed the
term to `max(8*bm*(threads//32),16*threads)`, covering both shared publication
slots and the measured helper workspace requirement. The subsequent budget
manifest confirms unchanged CUDA and resources for segmented BN32, ordered
BN64 and the high-shared reference fallback. This estimate was checked against
the exported cases, not exhaustively proved for every arbitrary private
factory geometry.

**Base-pointer alignment needs the main agent's GPU check.**
The private pipeline/async factories inspect outer stride parity only. The CUDA uses cp.async<16> for aligned-looking contiguous strides,
which directly emits 16-byte `cp.async.cg.shared.global`. A FP16 allocation
viewed as `[1:].reshape(...)` can have identical even strides and a data pointer
2 mod 16 (CPU metadata example recorded); `_validate` has no input data-pointer
alignment condition. The reference also emits vector loads, so whether the
failure is new, existing, or prevented by another integration layer must be
separated with a real GPU reproduction. Merely adding a stride guard does not
prove pointer alignment. After this finding, public prefill/decode wrappers in
attention.py gained `any(x.data_ptr()%16 ...)` selection of scalar reference
compilation and avoid optimized factories for that branch; this later scalar
branch is outside this reviewer's GPU-validated evidence.

**Known unsupported configuration is not a new regression.** BM16/N32 with
8 warps fails layout inference: those 8 warps cannot evenly cover the tile.
This restriction is inherent to this GEMM geometry; it was recorded rather
than treated as a synchronization bug or a successful compiled case.

The final GPU validation should cover dynamic lengths 0,1,BN-1,BN,BN+1,
2*BN,2*BN+1,3*BN+1; empty splits; page64/BN32; an unaligned page16 split with
at least two live tiles; poisoned inactive V; FP16/BF16; graph replay changing
lengths and page mappings. This reviewer did not perform those GPU checks.
