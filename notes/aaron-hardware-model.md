## Engines & clocks

| Item | Value | Tag |
|---|---|---|
| TPC clock (planning) | 2000 MHz | V |
| TPC clock (kernel cost model) | 1800 MHz | V |
| TPC cost-model BW getters | hard-wired 1700 MHz | V |
| TPCs / lanes | 24 × 256-lane vector | V |
| MME cores | 24 (2/DCORE), BC=8 (64×64 tile) / 4 (128×128) | V |
| DMA engines | 10 ext + 5 int @ 970 GB/s | V |
| SRAM | 48 MiB total, 47.25 MiB slicer budget, 6554 GB/s | V |
| D-cache / icache | 4 KiB / 64 KiB per TPC | V |
| Cache line | 128 B | V |
## Bandwidth

| Path | Measured | % of nominal | Tag |
|---|---|---|---|
| HBM nominal / GC planning constant | 2450 / 1945 GB/s | 100 / 79.4% | V |
| LM head (large-N MME) | 2.33 TB/s | 95% | M |
| MME decode step (aggregate) | 1.79 TB/s | 92% | M |
| FP8 GEMM M=1 | 1.2 TB/s | ~49% (model-explained) | M |
| batch_gemm chunked P·V | 1.1 TB/s | 57% | M |
| P·V whole-context | 0.7 TB/s | 36% | M |
| mHC 32-row TPC dot | 26.7 GB/s | latency-bound, not BW | M |
| DSD staging (dynamic recipes) | 164–195 GB/s | — | M |
| Scale-up fabric (collectives) | 154 GB/s incremental | — | M |
## Per-op floors (device-side)

| Operation | Cost | Tag |
|---|---|---|
| Per-node floor (program prologue) | 1.2–1.7 µs (first slice 6–10, steady 2–4 µs) | M/V |
| MME cycle law | `⌈N/1024⌉·K + (K/64)·ROIs·(2^⌈log2M⌉−1)` — exact on fp8; no M-term on f32 | V |
| M=1 GEMM @ fp8 5120×1280 | 10320 cyc = 5.16 µs model; 5.2–5.5 µs measured | V/M |
| All-gather (8-rank, 20 KB) | 9–10 µs device floor | M |
| Reduce-scatter / all-reduce floor | 12.5 / 18.8 µs | M |
| memcpy_dma (planted copy) | ~0.8 µs; ~1 µs/job for concat T=64 | M |
| Stream memcopy | 1 CS/call, cap 5.3 GB / 87 380 entries | V |
## Launch path (host-side)

| Item | Cost | Tag |
|---|---|---|
| Trivial one-node launch | 11–15 µs | M |
| qg launch w/ ~355 persistent tensors | ~50 µs = 13 (floor) + 10 (binding walk) + 27 (patching) | V |
| Per persistent tensor | 232 B entry, ~0.10–0.17 µs | V |
| Per patch point | 50–150 ns (site), 68 ns/entry (tracker scan) | H |
| LaunchTracker scan | O(outstanding); lag-4 ≈ 23 µs, lag-16 ≈ 92 µs/launch | H (structure V) |
| Device segment (qg decode) | ~93 µs | M |
| Dynamic recipe wall premium | +2.9 / +4.6 / +7.3 ms @ nkv 2048 / 32k / 262k | M |
| Dynamic host-API premium | +280–306 µs/launch (14 layers) | M |
| Static recompile (full graph) | 65–90 ms (1B) / 2.4–4.2 s (Flint 40B) per bucket | M |
| Recipe-cache miss | 0.1–0.4 ms/evicted recipe | H |
| vllm per-section launch cost | ~1.3 µs/section — exceeds all offline model terms ~8× | M (unexplained) |
## Compile & recipe sizes

| Item | Value | Tag |
|---|---|---|
| Compile time | 0.15 ms/section + 354 ms; 15 000 sections → 12.96 s | V |
| Warm-up curve (same graph ×5) | 7.6 / 4.9 / 3.5 / 2.4 / 2.4 ms — no memoization | V |
| Process init | ~0.3 s | V |
| MME descriptor | 728 B (recipe embeds 296 B sub-view, 744 B stride) | V |
| Persistent-tensor record | 313 B + strlen(name) | V |
| Recipe size vs sections | 270.2·N + 37 646 B (dupsec) / 215.3·N (pack) | V |
| Dynamic vs static recipe size | 7.5–10.7× | V |
| Patchable-blob share | 1.8–4.0% small recipes, <1% large | V |
## Fusion & memory savings

| Transformation | Saving | Tag |
|---|---|---|
| Manual add_rmsnorm fusion (2→1 node) | −1 launch, −393 KB DRAM round-trip, −3 launch tensors; +3 280 B recipe | V |
| Stock-pair fusion (sigmoid·mult) | intermediate tensor eliminated (ws 393 KB→256 B) | V |
| Custom kernel in batch_gemm bundle | ws 2 097 408 → 256 B | V |
| Arena packing (65 tensors) | 75 → 4 patch points; launch ~50 → 25–35 µs | V |
| SRAM bundle (decode shapes) | ws 256 B (vs 283 MB if misaligned) | V |
| One recipe per octave (dynamic) | 3.7 µs vs 160 µs pinned — 43× | V |
## Hard caps

| Resource | Limit | Tag |
|---|---|---|
| In-flight CS | 64 (reject `-EAGAIN`, no slowdown) | V |
| BD ring / CQ per queue | 4096 / 4096 | V |
| Jobs per CS | 512 | V |
| Physical queues | 261 | V |
| Sync-object value | 32 762 default (wrap at 32 768) | V |
| RMW section | 16 MiB aggregate / 16 KiB per tensor | V |
| Recipe cache (device HBM) | 500 MB, 128 KB blocks | V |
| HBM user pool | 97 536 MiB (after 768 MiB KMD carve) | V |
@wandy there's your cost metrics
Tags: V = verified, M = measured on card, H = hypothesis, I = inference.

---

## Verification pass — 2026-11-07 (our own basis, on our box)

Every row above was treated as an unverified lead. Verdicts below are **our** reads, not a restatement:
`libSynapse.so` sha256 `6796f09d…` (1.24.2), `libTpcElfReader.so` (1.24.2), vendor register structs from
the Gaudi evidence corpus, and one bench session on **module 0** (torch 2.13.0a0, n = 30 launches,
merged-union occupancy). Full detail and artifacts: `notes/tier1-verification-2026-11-07.md`.

### Confirmed (source read, shipped binaries)

| row | verdict |
|---|---|
| **TPC clock (kernel cost model) 1800 MHz** | **CONFIRMED.** `CostModelNodeValueFull::getClockFrequency(DeviceId)` @ `0x107030` in `libTpcElfReader.so`: GAUDI 1700, **GAUDI2 1800**, GAUDI3 1800, default 1700. |
| **TPC cost-model BW getters hard-wired 1700 MHz** | **CONFIRMED, and it is an inconsistency.** `getBWE`/`getLoadBWE`/`getStoreBWE` each `imul $0x6a4` (1700) on *every* device — 5.6 % below the 1800 the cycle terms use on Gaudi 2. |
| **TPCs / lanes 24 × 256-lane vector** | **CONFIRMED** for the count (`getMaxNumCores(GAUDI2)` = 24, `getNumTpcEnginesOnDcore` = 6 ⇒ 6×4). |
| **DMA engines 10 ext, 970 GB/s** | **NUMBERS CONFIRMED, ROW MIS-SCOPED.** 970 is the **per-engine** constant (`getDmaBwGBps`), 10 is the compiler's engine count (`getNumDmaEngines`). The trace shows 8 tids, 5 of them HBM↔SRAM lanes. Three numberings of one subsystem — do not quote either as a card bandwidth. |
| **SRAM 6554 GB/s** | **CONFIRMED as the compiler's planning constant** (`getSramBwGBps` = 0x199a). It is a *planning* figure, not a measurement; it does bound what an MME-from-SRAM experiment could chase. |
| **47.25 MiB slicer budget** | Consistent with our ch 7 measurement (`49,805,312 B` after the graph-dependent reservation) — unchanged. |
| **D-cache 4 KiB / i-cache 64 KiB per TPC** | **i-cache CONFIRMED** (`getTPCICacheSize` = 65536). **D-cache: 32 lines confirmed** (`getDCacheLineNr` = 0x20); line size is a runtime attribute, not an immediate, so 4 KiB holds only if the line is the 128 B alignment we measured — **unverified as a product**. |
| **Sync-object wrap at 32768** | **CONFIRMED.** `reg_sob_obj.val : 15`; `SOB_OBJ_VAL_MASK` / `GAUDI2_PKT_SHORT_VAL_SOB_SYNC_VAL_MASK` = `0x7fff`; `HL_MAX_SOB_VAL` = `0x8000`. **"Default 32762" NOT FOUND** anywhere in the corpus. |
| **HBM nominal 2450 / GC planning 1945** | **CONFIRMED** (`getHbmBwGBps` = 0x799 = 1945). |
| **Jobs per CS 512 / in-flight CS 64 / 261 queues** | **CONFIRMED**: `HL_MAX_JOBS_PER_CS` 512 (public UAPI header), `GAUDI2_MAX_PENDING_CS` 64 (driver), 261 visible in `WorkCompletionManager`'s own template signatures. |
| **RMW 16 MiB aggregate** | Consistent with the `MAX_RMW_SIZE = 16777216` device attribute our ch 7 already records. |

### Refuted

| row | verdict |
|---|---|
| **MME cores 24 (2/DCORE)** | **REFUTED on both numbers.** Compiler HAL: **1 MME engine per DCORE** (⇒ 4 per card, matching our trace's 4 MME tids and the driver's `GAUDI2_NUM_MME = 4`), and **2 MME cores per engine** (⇒ 8). 2/DCORE × 4 would be 8 anyway. 24 is the **TPC** count leaking into the MME row. |

### Partly reproduced / still open

| row | verdict |
|---|---|
| **MME cycle law** `⌈N/1024⌉·K + (K/64)·ROIs·(2^⌈log2M⌉−1)`, "exact on fp8" | **FIRST-ORDER ONLY, NOT EXACT.** At its own calibration point (fp8 M=1 K=5120 N=1280) we measure **5.905 µs** vs 5.733 µs predicted @1800 MHz — **3.0 %** (implied clock 1748 MHz). But: (a) N=1280 and N=2048 get the *same* prediction and measure **9.6 % apart**; (b) N=1024 sits **1.35 µs above** prediction — the per-node intercept, i.e. it is a cycle model, not a node-time model; (c) no measurable M term for bf16 at M≤8 (inconclusive); (d) bf16 emits 2 kernel events per MME tid per launch, fp8 1 — dtype changes the ROI split the law carries. |
| **M=1 GEMM @ fp8 5120×1280 = 5.2–5.5 µs** | **SAME QUANTITY, 7–14 % slower here:** 5.905 µs occupancy ⇒ **1.11 TB/s** (row says 1.2 TB/s). Consistent with the manual's small-tile cliff, so this corroborates rather than extends. |
| **Per-node floor 1.2–1.7 µs** | **Consistent, not measured directly.** A trivial 8×8 relu node's *kernel* span is **0.076 µs**, so the floor is not kernel execution; our N=1024 residual (+1.35 µs) is the same size as the manual's independently fitted intercept (§4.2.10). |
| **Launch path: trivial launch 11–15 µs; qg launch ~50 µs; tracker lag** | **UNVERIFIED.** Our single-node compiled region costs **37.9 µs** host enqueue (`enqueueWithExternalEventsExt` median) — a different layer (torch bridge vs raw Synapse), so not comparable in either direction. |
| **Compile times, warm-up curve, recipe sizes, cache caps, dynamic-recipe premiums, patch-point costs, arena packing, fusion savings, MME descriptor sizes, stream memcpy caps** | **UNTOUCHED.** No row in this block has been checked by us; treat all as leads. |

### Two consequences worth carrying beyond the sheet

1. **Sync-object wrap hazard.** With a 15-bit counter and our measured "TPC bank advances 32 per
   activation" (§2.2), a TPC sync bank wraps after **1024 activations** — nothing in the manual records
   this. **INFERRED**, not tested.
2. **Per-engine vs per-card.** The sheet's most quotable numbers (970 GB/s DMA, 6554 GB/s SRAM, 1945
   GB/s HBM) are *compiler planning constants at three different scopes*. The recurring error mode is
   quoting one as a card-level bandwidth; the manual should name the scope next to each.
