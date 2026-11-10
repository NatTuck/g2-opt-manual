# Tier-1 verification of an external Gaudi2 cost-model sheet — 2026-11-07

**Status: leads checked against the shipped binaries, the vendor source corpus, and one bench session
on module 0.** Nothing here is citable as "the sheet says"; every surviving row carries *our* basis and
*our* artifact. The sheet itself (`notes/aaron-hardware-model.md`) is an unverified external lead list
(author: an internet source, self-tagged V/M/H/I).

Environment: torch 2.13.0a0+git2469d18, Habana 1.24.2, one 4-AIP Gaudi2 container, bench pinned to
`HLS_MODULE_ID=0`. Binaries examined:
`libSynapse.so` sha256 `6796f09dac786b95970f55541c8c27dd4005e87a5ecd1b95bb11daec6f3b3913` (the same
binary the manual already cites), `libSynapseMme.so` `467e0b56…`, `libTpcElfReader.so` (1.24.2),
`tpc-clang` 2.7.33. Scripts: `notes/tier1-2026-11-07/bench_mme.py`, `bench_fp8.py`; traces are
disposable (`/tmp`), the numbers below are the record.

---

## 1. VERIFIED — TPC cost model clock is 1800 MHz, and the BW getters are hard-wired at 1700

This closes the manual's §1.12 items 1–2 ("the TPC cost model's clock: not successfully checked"), and
**refutes the worry** that the TPC model consumes the 2000 MHz general reader. **source read.**

`libTpcElfReader.so`, `CostModelNodeValueFull::getClockFrequency(tpc_lib_api::_DeviceId)` @ `0x107030`:

| DeviceId | clock | immediate |
|---|---:|---|
| 1 (GAUDI) | 1700 MHz | `0x6a4` |
| **3 (GAUDI2)** | **1800 MHz** | `0x708` |
| 4 (GAUDI3) | 1800 MHz | `0x708` |
| default | 1700 MHz | `0x6a4` |

`getBWE`, `getLoadBWE`, `getStoreBWE` @ `0x106ea0` / `0x106f00` / `0x106f60` each `imul $0x6a4` —
**1700 MHz unconditionally, on every device**, i.e. the bandwidth-derived terms of the kernel cost
model use a clock 5.6 % below the 1800 MHz the cycle terms use on Gaudi 2. That is a real internal
inconsistency in the vendor toolchain, not a transcription error.

Cross-check: `getMaxNumCores(DeviceId)` returns **24** for GAUDI2 (`0x18`), 8 for GAUDI, 10 for GAUDI3.

## 2. VERIFIED — other compiler planning constants

`Gaudi2HalReader` getters, all **source read** from `libSynapse.so` (each a `mov $imm,%eax; ret`):

| getter | value | note |
|---|---:|---|
| `getHbmBwGBps` | **1945** | the GC planning constant (79.4 % of the 2450 nominal) |
| `getSramBwGBps` | **6554** | the GC's SRAM bandwidth constant — *the* figure the sheet quotes for SRAM |
| `getDmaBwGBps` | **970** | **per DMA engine**, not per card |
| `getNumDmaEngines` | **10** | the compiler's count (trace shows 8 tids / 5 HBM↔SRAM lanes — different numberings) |
| `getNumTpcEngines` / `…OnDcore` | 24 / 6 | 6 × 4 DCOREs |
| `getNumMmeEnginesOnDcore` | **1** | ⇒ **4 MME engines** on the card, matching the trace's 4 MME tids |
| `getNumMmeEngines` / `…CoresPerEngine` | 2 / 2 | a *different* numbering — see §6 |
| `getTPCICacheSize` | **65536** (64 KiB) | sheet's i-cache row confirmed |
| `getDCacheLineNr` | **32** | lines; line *size* is a runtime attribute, not an immediate |
| `getDRAMSizeInBytes` | 0x1800000000 (96 GiB) | |
| `getMmeMinimalWidthInElems` / `…Symmetric…` | 128 / 256 | |

`getClockFreqMHz` for Gaudi2 returns **2000** — confirming the manual's existing "general reader is
21 % high against the 1650 MHz SoC clock" note.

## 3. VERIFIED — sync-object counter is 15 bits (wrap 32768)

**source read**, vendor register structs (SynapseAI_Core and HCL copies agree):

- `gaudi2::sob_objs::reg_sob_obj` — `uint32_t val : 15` (plus `long_sob`, `trace_evict`, `inc`).
- `DCORE0_SYNC_MNGR_OBJS_SOB_OBJ_VAL_MASK = 0x7fff`;
  `GAUDI2_PKT_SHORT_VAL_SOB_SYNC_VAL_MASK = 0x7fff`; `reg_mon_arm.sod : 15`.
- `HL_MAX_SOB_VAL = 0x8000` (driver); `MAX_SOB_VAL = 0x7fff` (thunk UAPI).

⇒ the wrap-at-32768 half of the sheet's row is **confirmed**; the "default 32762" is **not found**
(no such constant in the corpus, and `lookup_constant value=32762` returns nothing). A `long_sob` mode
exists and was not investigated.

Related, also confirmed: `HL_MAX_JOBS_PER_CS = 512` (public UAPI header
`/usr/include/drm/habanalabs_accel.h`), `GAUDI2_MAX_PENDING_CS = 64`, and the 261-entry physical-queue
array is visible in `WorkCompletionManager`'s own template signatures (`std::array<…, 261>`).

**Consequence worth carrying into the manual:** with a 15-bit counter and the manual's measured
"TPC bank advances 32 per activation" (§2.2), a TPC sync bank wraps after **1024 activations** —
a wrap hazard for long-lived monitors that nothing in the manual currently records. **INFERRED**,
not tested.

## 4. PARTLY REPRODUCED — the MME cycle law

Law as stated: `cycles = ceil(N/1024)·K + (K/64)·ROIs·(2^ceil(log2 M) − 1)`, claimed exact on fp8.

Bench: one `torch.compile(fullgraph=True, dynamic=False)` region, one node, warm 5, n = 30 launches,
**merged-union device occupancy per launch** (the manual's statistic, not the marker median). fp8 via
`torch.ops.hpu.fp8_gemm_v2` (A `[M,K]`, B `[N,K]`, `trans_B=True`), bf16 via `matmul`.

| arm | measured occupancy | law cycles | law @1800 MHz | meas/law |
|---|---:|---:|---:|---:|
| fp8 M=1 K=5120 **N=1280** (the sheet's calibration point) | **5.905 µs** | 10320 | 5.733 µs | **1.030** |
| fp8 M=1 K=5120 N=1024 | 4.240 µs | 5200 | 2.889 µs | 1.47 |
| fp8 M=1 K=5120 N=2048 | 6.474 µs | 10320 | 5.733 µs | 1.13 |
| bf16 M=1 K=5120 N=1024 | 6.403 µs | — | — | — |
| bf16 M=1 K=5120 N=1280 | 9.062 µs | — | — | — |
| bf16 M=1 K=5120 N=2048 | 11.130 µs | — | — | — |
| bf16 M=1 K=5120 N=4096 | 19.885 µs | — | — | — |
| bf16 M=2/4/8, K=5120, N=1024 | 6.475 / 6.452 / 6.566 µs | — | — | — |

**What reproduces:** at the sheet's own calibration shape the law lands within **3.0 %** of our measured
occupancy at its own 1800 MHz clock (implied clock 10320 cyc / 5.905 µs = **1748 MHz**). The sheet's
"5.2–5.5 µs" is a little optimistic against our 5.905, but the same quantity.

**What does not reproduce, and is why this is not "exact":**
1. **The N quantisation fails.** N=1280 and N=2048 both give `ceil(N/1024) = 2` and therefore the *same*
   predicted cycle count, yet they measure **9.6 % apart** (5.905 vs 6.474 µs).
2. **Small N carries a fixed term the law omits** — at N=1024 the measurement is **1.35 µs above** the
   law@1800 prediction, which is the same size as the manual's independently measured per-node
   intercept (§4.2.10, 1.2–1.7 µs). So the law is a *cycle* model, not a node-time model.
3. **No measurable M term for bf16 at M ≤ 8** (6.40 → 6.57 µs across M=1/2/4/8, 2.6 % spread), where the
   law's `(2^ceil(log2 M) − 1)` term would predict +560 cycles at M=8 (+0.31 µs at 1800) — the same
   order as the observed spread, so this dimension is **inconclusive**, not refuted.
4. **Structural difference by dtype:** a bf16 GEMM emits **2 kernel events per MME tid per launch**
   (8 events/launch over 4 tids), an fp8 GEMM **1** (4 events/launch) — the node's ROI/execution split
   differs by dtype, which is exactly the `ROIs` term the law carries. Untested as a term.

**Verdict: the law is a useful first-order predictor on fp8 at its calibration point (3 %), and is not
exact in the dimensions we varied. It is not a node-time model.**

## 5. VERIFIED — M=1 GEMV rate at the calibration shape

fp8, M=1, K=5120, N=1280: 6,553,600 B in **5.905 µs** ⇒ **1.11 TB/s** (sheet: 1.2 TB/s, 8 % high).
bf16 same shape: 13.1 MB in 9.062 µs ⇒ 1.45 TB/s. Both sit inside the manual's §3.2.1 small-tile
cliff band (1.14 TB/s at 4.2 MB → 1.94 at 16.8 MB), so this **corroborates** the manual rather than
adding a new fact. The per-node floor row (1.2–1.7 µs) is **consistent** with our data via item 4.2
above but was not measured directly: a trivial 8×8 relu node's *kernel* span is **0.076 µs**, so the
floor is not kernel execution and needs the node-occupancy instrument (§4.2.10's `span_min` fit).

## 6. REFUTED / UNRECONCILED

- **"MME cores 24 (2/DCORE)"** — refuted on both numbers. The compiler's HAL says **1 MME engine per
  DCORE** (⇒ 4 per card, matching our trace's 4 MME tids and the driver's `GAUDI2_NUM_MME = 4`) and
  **2 MME cores per engine** (⇒ 8 cores). Neither is 24, and 2/DCORE × 4 would be 8 anyway. The 24
  appears to be the **TPC** count leaking into the MME row.
- **"DMA 10 ext + 5 int @ 970 GB/s"** — the numbers exist but mean something narrower than the row
  implies: 970 GB/s is the compiler's **per-engine** DMA constant, 10 is the compiler's DMA engine
  count, and the trace shows 8 tids of which 5 are the HBM↔SRAM lanes. Three numberings of one
  subsystem; do not quote any of them as "the DMA bandwidth".
- **Per-launch cost** — our harness measures a single-node compiled region's host enqueue at
  **37.9 µs** median (`enqueueWithExternalEventsExt`), against the sheet's 11–15 µs "trivial launch".
  Different harness and different layer (torch bridge vs raw Synapse), so **not comparable**; the
  sheet's launch-path decomposition remains **unverified**.
- **Everything else in the sheet** (compile times, recipe-cache sizes, dynamic-recipe premiums,
  LaunchTracker scan costs, RMW caps, arena packing) is **untouched** by this pass.

## 7. What this changes in the manual (proposed, not yet edited)

1. **§1.7 / §1.12 items 1–2** — close them: TPC cost-model clock is 1800 MHz (GAUDI2), 1700 for
   GAUDI1/default; the BW getters are hard-wired 1700 on every device; the 2000 MHz general reader is
   **not** what the TPC cycle model uses. Add the 1700-vs-1800 inconsistency as a new, smaller caveat.
2. **§1.5** — add the per-TPC i-cache (64 KiB) and D-cache (32 lines) as `source read` rows, and
   qualify "no L1/L2" as "no cache serving HBM streams" rather than "no caches".
3. **§1.7** — add the compiler's own bandwidth constants (HBM 1945, SRAM 6554, DMA 970/engine) as the
   *planning* constants, distinct from our measured envelopes.
4. **§4.2.10** — the i-cache hypothesis for the sublinear per-launch term now has a bound (64 KiB).
5. **§2.2 / §3.9** — record the 15-bit sync-object counter and the resulting wrap hazard for a
   TPC bank advancing 32 per activation.
6. **§7** — the SRAM 6554 GB/s figure is the compiler's planning constant, which sets the ceiling an
   MME-from-SRAM experiment would be chasing; it is not a measurement.
7. **§1.1** — keep "4 MME" as measured; the "24 MME cores" claim is refuted and should go to the
   dead-end register only if someone is likely to re-import it (it is an external claim, so a line in
   §3.9's register is the right home).
