# Recipe residency and the device memory budget

**Status:** working notes for our own team, not a manual chapter. Written 2026-11-13 from measured
probes on the runtime we actually execute on (1.24.2 / Gaudi2) plus a read of an old Synapse source
snapshot. The measurements are cheap and repeatable — §6 has the recipe.

**Provenance.**

- **Runtime:** `/usr/lib/habanalabs/libSynapse.so`, package `habanalabs-graph 1.24.2-481`, Synapse
  build `79e284b` (matches the `_syn1.24.2.79e284b.recipe` filenames we generate). Driver
  `1.24.2-52a555d`, torch `2.13.0a0+git2469d18`, `habana-torch-plugin 1.24.2`.
- **Is it the clean vendor binary?** `dpkg -S` says the package owns it, `dpkg -V habanalabs-graph`
  reports **no modified files** (two man pages missing; the library itself is untouched), and `nm -D`
  shows **no** `nativeCompute*` symbols.
- **Does a normal process really load it?** `LD_LIBRARY_PATH` puts `/usr/lib/habanalabs` ahead of any
  `RUNPATH`; verified by reading `/proc/self/maps` from inside a `python` + `torch_hpu` process.
- **Source available for reading:** an **old 1.16 snapshot** (base commit `6de1f66`), from a
  third-party mirror of the vendor tree. Extract a clean copy with:

  ```bash
  git archive 6de1f66 | tar -x -C <dir>
  ```

  The working tree at `~/Envs/onecat/src/synapse` is that base **plus a heavily patched research
  tree** — `git diff --shortstat` there reads **80 files changed, 4211 insertions(+), 863
  deletions(-)**, with **17 untracked files** (16 of them source under `synapse/`, plus one build
  stub), including the `native_*` API: `native_compute_graph.hpp`, `native_recipe_memory.hpp`,
  `native_program_window.h`, `native_compute_program.hpp`, `native_bounded_selection.h`,
  `native_compute_graph_handle.hpp`. **It must not be cited.** Everything tagged *source read* below
  is from the clean extraction of `6de1f66`.
- **Path convention.** Synapse paths are relative to the **root of that extraction** — `synapse/` is
  the library subtree, `swtools_sdk/` its sibling. Bridge paths are relative to the bridge source
  tree, as in [`recipe-launch-notes.md`](recipe-launch-notes.md). A bare filename means the directory
  was given in full earlier in the same section.

**Basis tags.** **MEASURED** (ran here, artefact named) · **source read** (a named file and line;
1.16 unless stated) · **ASSUMED** · **INFERRED** (derivation stated) · **UNKNOWN**. The tag
vocabulary is the one defined in the [README](../README.md).

**Version caveat.** The source snapshot is **1.16**; the runtime is **1.24.2**. Source reads tell us
structure and mechanism, and each claim that matters is corroborated against the shipped binary
(`nm -D`, `strings -a`) or against a measurement. Where a number comes from the 1.16 source rather
than from this host, it says so.

Two traps that cost us time and belong in the "fast iteration" chapter:

1. **The onecat tree is patched.** It carries an uncommitted `synNativeComputeGraph*` "research API"
   whose comment claims it "skips recipe preparation, address patching, PDMA download and per-recipe
   synLaunch", and a `NativeRecipeMemory` / `m_nativeRecipeMemory` / `arcCoreAddress()`
   device-resident-program path. **Neither exists in vendor Synapse** — zero in the clean tree, zero
   `nativeCompute`/`NativeRecipeMemory` symbols in the 1.24.2 binary. Two different residency models
   live in this workspace; do not mix them. Size it by the counting rule
   `grep -rho 'synNative[A-Za-z0-9_]*' synapse/include synapse/src` over that tree: **210 occurrences
   in 14 files, 29 distinct identifiers** (21 `synNativeComputeGraph*` entry points and types, 8
   `synNativeGraph*` state enums). None of it is in the base commit.
2. **`hl_logger` levels are inverted**: `TRACE=0 DEBUG=1 INFO=2 WARN=3 ERROR=4 CRITICAL=5 OFF=6`.
   `LOG_LEVEL_ALL=5` is *near-silence*, not maximum verbosity. `LOG_LEVEL_ALL=0` is loudest.
   (**source read**, 1.16: `swtools_sdk/hl_logger/include/hl_logger/hllog_core.hpp:27-33`.) This
   cost three wasted runs.

---

## 1. The headline: on Gaudi2 a recipe is *never* resident in HBM

The mental model "compiled recipe ⇒ HBM occupied" is wrong, and so is "there is a recipe cache in HBM
holding my compiled graphs". Measured and read:

* **`synGraphCompile` allocates no device memory.** It produces host-side buffers: every blob the
  device later walks is `mmap`ed host pages mapped into the device address space
  (`RecipeAllocator` → `MemoryAllocatorUtils::alloc_memory_to_be_mapped_to_device`, with
  `madvise(MADV_DONTFORK)` and the comment "in order to prevent physical pages mapped to device from
  being swapped, thus causing MMU mapping problems" — which is proof they are host pages).
  (**source read**, 1.16: `synapse/src/runtime/common/recipe/recipe_allocator.cpp:15-30`,
  `synapse/src/runtime/common/memory_utils/memory_allocator_utils.cpp:12-40`.) The buffers allocated this way
  are exactly the ones the device walks: `execution_blobs_buffer`, `patching_blobs_buffer`,
  `dynamic_blobs_buffer`, both ECB command buffers, and `program_data_blobs_buffer`
  (`synapse/src/runtime/common/recipe/recipe_serializer.cpp:496/504/514/599/607/643`; `synapse/src/graph_compiler/recipe_blob.cpp:452/457/462`;
  `synapse/src/graph_compiler/recipe_ecb.cpp:418/434`).
* **Every `synLaunch` downloads the program again.** `RecipeLauncher::launch()` calls
  `prepareForDownload()` then `downloadToDev()` unconditionally — no "already resident" branch. Its
  own doc comment lists the steps: *"Download to device HBM (including patching) … enqueue to device
  (sends packets with recipe location)"*. `downloadToDev` does `memcpyToMapped` → patch →
  **`pdmaDownload`** (mapped host → HBM, by CpDma commands) → fence wait → `scalEnqueue`.
* **The destination is a rotating ring, not a cache.** `ArcHbmMemMgr` / `HbmGlblMemMgr` derive from
  `NBuffAllocator`, which splits a fixed window into `NUM_BUFF` chunks and hands out the next one each
  time, returning *"an offset and the long-sync-object to wait for"*; `prepareForDownload` then does
  `longSoWaitOnDevice(...)` — *wait until this slot is free on the device*. (**source read**, 1.16:
  `NUM_BUFF = 16` at `synapse/src/runtime/scal/common/recipe_launcher/nbuff_allocator.hpp:28`; chunk size is
  `size / NUM_BUFF` at `synapse/src/runtime/scal/common/recipe_launcher/hbm_global_mem_mgr.cpp:39-43`; on
  1.24.2 the count measures **64** on the **global** window, §2.)
* The vendor says it in as many words (both in `synapse/src/runtime/scal/common/recipe_launcher/`:
  `arc_hbm_mem_mgr.hpp`, repeated verbatim in `hbm_global_mem_mgr.hpp`):

  > This class is used to allocate memory of the Arc-HBM for running a recipe. It is currently
  > implemented in a very simple way, **assuming no caching**. It splits the given memory into chunks
  > (downloadedRecipes) and each time assigns a new chunk. … **In the future, we can consider adding
  > caching here to avoid the pdma**

* **`synRecipeDestroy` frees host memory only.** `recipeDestroy` → `notifyRecipeRemoval` →
  `removeRecipeHandle` → `destroyRecipeHandle`; no device free anywhere, because the recipe never
  owned device memory. **An unlaunched compiled recipe holds zero HBM.** (**source read**, 1.16:
  `synapse/src/runtime/common/recipe/recipe_handle_impl.cpp:18-24` frees only the `RecipeAllocator`.)
* The bridge's `RecipeCacheLRU` (`HABANA_PGM_LRU_MAX`, defaults 100 000 eager / 30 000 lazy) holds
  `shared_ptr<RecipeHolder>` around live `synRecipeHandle`s and evicts by **host** memory
  (`get_recipe_host_mem_size()`, threshold `PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT`), calling
  `synRecipeDestroy` on the GC thread. It bounds **host** memory. It has nothing to do with HBM.
* The workspace is not per-recipe either: `synLaunch` takes `pWorkspace` from the caller, and the
  bridge keeps **one device-wide** workspace buffer. (**source read**, 1.16: the workspace is an
  `[in]` argument of `synLaunch`, `synapse/include/synapse_api.h:612-617`.)

So the honest one-liner for the manual: **a graph costs one program download per launch, not one HBM
allocation per compile.**

---

## 2. The numbers, measured on 1.24.2 / Gaudi2

Read straight out of the runtime with `LOG_LEVEL_SYN_PROG_DWNLD=0` (printed once, at device acquire):

```
ArcHbmMemMgr: buffer addrCore given/actual 52000000/52000000
               addrDev given/actual 1001600152000000/1001600152000000
               size given/actual 7f80000/7f80000   chunkSize 1f8000
HbmGlblMemMgr: buffer addr given/actual 100160015a000000/100160015a000000
               size given/actual c000000/c000000   chunkSize 300000
```

| quantity | 1.24.2 measured | 1.16 source | note |
|---|---|---|---|
| chunks per **global** window | **64** — **INFERRED** | `NUM_BUFF = 16` | one method at three settings of one key: 192/3, 768/12 and 3072/48 MiB all give 64. **Global window only** — the arc chunk does not divide its window (§2a) |
| arc ("shared") window | `0x7F80000` = **127.5 MiB**, chunk `0x1F8000` ≈ 1.97 MiB | 32 MiB, chunk size/16 | window not moved by any key we found; chunk does not divide it (§2a) |
| global window | `0xC000000` = **192 MiB**, chunk `0x300000` = 3 MiB | `HBM_GLOBAL_MEM_SIZE_MEGAS` × 1 MiB | see scaling below |
| compute streams | 3 (`MaxNumOfComputeStreams`, unchanged) | 3 | the pre-allocation is `Size × 3` |
| max single-recipe program | the whole window (16 chunks in 1.16; 64 now) — beyond it the launch **throws**, it does not spill | | |

### 2a. The arc window's chunk does not divide the arc window

127.5 MiB ÷ 64 = 1.992 MiB = `0x1FE000`, and that is already 128-aligned, so
`round_down(chunkSize, ALIGN)` leaves it alone. The measured chunk is `0x1F8000` — smaller by
0.023 MiB — so 64 chunks hold **126 MiB** and **1.5 MiB** of the arc window is never handed out
(`64 × 0x1F8000 = 0x7E00000` against `0x7F80000`). **The 64 therefore holds for the global window
only**, and the arc row of the table above is a window with a chunk in it, not 64 of them.

Where the 1.5 MiB went. The 1.16 source has two terms, and only one of them fits:

* `getChunkSize()` divides by `NUM_BUFF` and rounds **down** to `ALIGN = 128`
  (`synapse/src/runtime/scal/common/recipe_launcher/arc_hbm_mem_mgr.cpp:44-48`; `ALIGN` at
  `arc_hbm_mem_mgr.hpp:36`). On its own this gives `0x1FE000`, not the measured `0x1F8000`.
* `HbmGlblMemMgr::init` chunks `m_size = size - (m_addr - addr)` — the size **after** the base address
  is rounded up (`synapse/src/runtime/scal/common/recipe_launcher/hbm_global_mem_mgr.cpp:24-27`). If
  the 1.24.2 arc manager does the same and lost 1.5 MiB to base alignment, then
  `0x7E00000 / 64 = 0x1F8000` **exactly**. **INFERRED**, and the log cannot settle it: it prints
  `size given/actual 7f80000/7f80000`, and the 1.16 global line it is modelled on prints the *given*
  value in both fields (`addr, addr` where the second should be `m_addr`,
  `synapse/src/runtime/scal/common/recipe_launcher/hbm_global_mem_mgr.cpp:30-36`). The mechanism is
  **UNKNOWN**; the two numbers above are measured.

**What the budget uses is the 127.5 MiB window, not the chunk**, so nothing downstream depends on
this. What would settle it: a 1.24.2 source drop, or a log line that prints `m_size`.

**Source corroboration for the 1.16 column** (all *source read*, clean `6de1f66`):
`MaxNumOfComputeStreams = 3` (`synapse/src/runtime/scal/common/entities/scal_dev.hpp:58`);
`hbmSharedSize = 32 * 1024 * 1024` (`synapse/src/runtime/scal/common/device_scal.hpp:126`);
`HBM_GLOBAL_MEM_SIZE_MEGAS` default **64** (`synapse/src/graph_compiler/habana_global_conf.cpp:594-598`);
the pools are allocated as `Size × MaxNumOfComputeStreams` (`synapse/src/runtime/scal/common/device_scal.cpp:242-247`).

**`HBM_GLOBAL_MEM_SIZE_MEGAS` scales the global window — but it is a PRIVATE key, and it is
per-stream.** Measured, with the *device total* read from `torch.hpu.mem_get_info()` in the same run
(so the last column is what it actually costs you):

| env | logged global window | device total reported | delta vs default |
|---|---|---|---|
| (unset, default 64) | `0xC000000` = 192 MiB | 96895.2 MiB | — |
| `HBM_GLOBAL_MEM_SIZE_MEGAS=32` alone | 192 MiB — **silently ignored** | 96895.2 MiB | 0 |
| `ENABLE_EXPERIMENTAL_FLAGS=true …=256` | `0x30000000` = 768 MiB | 96319.2 MiB | **−576 MiB** = the window delta, exactly |
| `ENABLE_EXPERIMENTAL_FLAGS=true …=1024` | `0xC0000000` = 3072 MiB | 91359.2 MiB | **−5536 MiB** ≠ window delta (2880) |

Readings that follow:

* The key is **per compute stream** and the reservation is **3 × the key** (256 → 768 MiB logged, and
  the device total fell by exactly that 576 MiB delta). Default 64 ⇒ **192 MiB**.
* Private keys are honoured from the environment only when `ENABLE_EXPERIMENTAL_FLAGS=true`
  (`hl_gcfg`: `if (m_isPublic || enableExperimental)`; `synConfigurationSet` passes
  `enableExperimental=false`, so the API cannot set them at all — environment only, before start).
* **Do not raise this key casually.** At 1024 the cost was 5536 MiB, not 2880: the larger allocation
  crossed a 4 GB address boundary and triggered the allocator's *"not aligned, moving to next 4GB
  alignment, allocation start … allocation end …"* workaround (**source read**, 1.16:
  `device_scal.cpp:266-290`), which throws away the difference. **A size knob here is not linear.**

**The only knob is a size, never a policy.** Bigger window ⇒ bigger maximum single-recipe program and
fewer ring-waits; it does **not** make anything resident.

---

## 2b. The memory budget: what to subtract from the 96 GB before sizing the KV cache

**All MEASURED** on 1.24.2 / Gaudi2, one card. **Three "total HBM" figures are in circulation and they
are three views of the same card** — quote the one you measured, with its instrument:

| view | figure | instrument |
|---|---|---|
| **driver** | **98304 MiB = 96.00 GiB** | `hl-smi -Q memory.total -f csv,nounits,noheader` |
| **Synapse** | **97376 MiB = 95.09 GiB** | `synDeviceGetMemoryInfo(dev, &free, &total)` — take `total` |
| **torch** | **96895.2 MiB = 94.62 GiB** | `torch.hpu.mem_get_info()[1]` — **bytes**, divide by 2²⁰ |
| **driver `memory.used` at idle** | **768 MiB** | `hl-smi -Q memory.used …`, no process holding the card |

The driver's total is **928 MiB** larger than Synapse's: firmware and driver reservation, which Synapse
never sees and you can never allocate. Synapse's is **481 MiB** larger than torch's: the program
windows, carved out at device acquire. And the **768 MiB** the driver reports as *used* at idle is the
idle baseline this manual already publishes (`01-basics.md:394`: *"an idle baseline of ~768 MiB.
**MEASURED**"*) — a *usage* figure, not a fourth total, and **not** something to subtract on top of
the 481 MiB.

| item | size | when | scales with recipe count? |
|---|---|---|---|
| card HBM, Synapse view — **the budget base** | **97376 MiB** = 95.09 GiB | — | no |
| ↳ global program window | **192 MiB** = 3 × `HBM_GLOBAL_MEM_SIZE_MEGAS`(64) | device acquire, before any tensor | **no** |
| ↳ arc program window | **127.5 MiB** (`0x7F80000`) | device acquire | **no** |
| ↳ streams, monitors, TDR, profiler | **~161 MiB** (remainder) | device acquire | no |
| **total reserved before your first tensor** | **~481 MiB (0.47 GiB)** — which is exactly why torch reports 96895.2 | | **no** |
| your workspace | `synWorkspaceGetSize`, **one** device-wide buffer in the bridge | first launch | no (sized by the max, not the count) |
| your tensors (weights, KV cache, activations) | the rest | you | you |

**The recipe count does not appear in this table, and that is the measurement.** 8 distinct recipes,
each compiled *and launched*, moved free HBM by **0.0 MiB** (per-step readings identical to the
0-recipe baseline, `96889.7 MiB` free before and after). Compiling more graphs costs host memory and
disk, not HBM.

Practical rule for the manual:

> Budget = **97376 MiB − 481 MiB** − workspace − weights.
> **Do not subtract a per-recipe term.** If HBM is filling up as the graph count grows, the recipes
> are not what is filling it — look at persistent/const sections, the workspace being sized by the
> largest recipe, or tensors retained by the host-side cache.

Two footnotes that matter:

* `synDeviceGetMemoryInfo()` on this build returns **`free = 0`** (status `synSuccess`, `total`
  correct). It is not usable for headroom; `torch.hpu.mem_get_info()` is. Note that the torch call
  returns **bytes**, not MiB — a 2²⁰ error waiting to happen, and the reason the figures above are
  quoted to one decimal place.
* The reported `total` **falls** as you raise `HBM_GLOBAL_MEM_SIZE_MEGAS` — Synapse carves the window
  out of the pool, so you see the cost in `total`, not in `free`. Measure the budget by differencing
  `total`, not `free`.

### 2c. "What if we execute a million recipes?" — the device is the cheap resource

Measured, 24 distinct recipes each compiled **and launched** (0 → 24):

| resource | 0 recipes | 24 recipes | growth | where the bound comes from |
|---|---|---|---|---|
| device memory (free) | 96889.7 MiB | 96889.7 MiB | **+0.0 MiB** | the 64-chunk ring; constant per stream |
| host RSS | 2104.6 MiB | 2215.2 MiB | **+110.6 MiB** (~4.6 MiB/recipe) | one `recipe_t` + its mmap'd program buffers **per live recipe** |
| disk (`.recipe` files) | 0 | 0.4 MiB / 9 files | grows with distinct recipes | `PT_HPU_RECIPE_CACHE_CONFIG` `max_dir_size_mb` (default 1024) |

So the inversion to document:

> **A million recipes costs the same HBM as one recipe, and is impossible in host memory.** The device
> holds a fixed window; the *host* holds one program image per live recipe, in `mmap`ed pages that are
> mapped into the device address space and marked `MADV_DONTFORK` (they cannot be swapped out). That is
> what the bridge LRU is for: it evicts on a **host** memory threshold
> (`PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT`), in **512 MiB** chunks (`512LL * 1024 * 1024`,
> `backend/kernel/hpu_habana_launch_op_pt.cpp:4211` — the code comment says "512MB", the constant is
> MiB), calling `synRecipeDestroy` on the GC thread.

**The threshold's default is a two-level story, and a grep misleads.** The gcfg default for
`PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT` is **0**, and 0 **disables the check entirely**
(`scripts/env_flags.yaml:428-430`; `pytorch_helpers/habana_helpers/misc_utils.cpp:31-34` tests
`!= 0U`). The familiar **90 %** comes from the *bridge*, which substitutes its own default the first
time a dynamic bucket is created: `RecipeCacheLRU::SetHostMemoryThreshold()` writes
`default_host_memory_threshold = 90` into the flag if it is unset
(`backend/kernel/hpu_recipe_cache.h:130`, `backend/kernel/hpu_recipe_cache.cpp:244-249`, called from
`CreateFirstDynamicBucket`). So "default 90 %" is true of a process that compiled something and false
of the gcfg — and the eviction is only *attempted* when a graph is built or a recipe inserted
(`hpu_habana_launch_op_pt.cpp:6305`, `backend/kernel/hpu_recipe_cache.cpp:80`), never on a timer.

Two corollaries:

* **A "recipe count" HBM problem is almost certainly a host-memory problem wearing HBM clothes.** When
  the LRU evicts under host pressure, the evicted recipes *recompile* on their next launch — which
  presents as a rising `compileGraph` count and a slower steady state, easily misread as device
  pressure.
* **Executing a million recipes is compile-bound, not memory-bound.** First execution of a distinct
  recipe measured ~50 ms dynamo + ~52 ms bridge/Synapse ≈ 100 ms. A million of them is ~28 hours of
  compilation, and each launch still pays a program download.

---

## 3. Is there an API for residency? No — and this is now proven three ways

1. **Names and docs.** 145 public entry points in the clean `synapse_api.h`; a case-insensitive grep
   for `residen|evict|pin|prefetch` over the whole header (names *and* doc comments) returns nothing.
2. **The shipped binary.** 155 exported `syn*` symbols in the 1.24.2 library; the same grep over
   `nm -D` output returns nothing. The word `resident` appears in the whole 1.16 tree only as
   `m_baseRegsCache` — "cache resident **section IDs**" for base-registers caching, an unrelated
   mechanism.
3. **The complete inventory.** Every public call that touches device memory or recipe lifetime, and
   why it is not residency control:

| call | what it does | why not |
|---|---|---|
| `synGraphCompile` / `V2` | builds a recipe; **no device allocation** | nothing to pin |
| `synRecipeDestroy` | frees the host-side recipe | recipe never owned HBM |
| `synRecipeSerialize` / `DeSerialize` | recipe ⇄ host file/blob | bytes, not device state |
| `synRecipeGetAttribute` | scalars from the handle (`WORKSPACE_SIZE`, `DEVICE_MEMORY_SIZE`, `HOST_MEM_SIZE`, `CONST_SECTIONS_SIZE`, `PERSISTENT_TENSORS_SIZE`) | **read-only — but this is the attribution tool we lacked** |
| `synRecipeSectionGetProp` | a **const** section's size and its *host* pointer | no device address; const sections only |
| `synWorkspaceGetSize` | scratch bytes for one recipe | workspace is the caller's |
| `synDeviceMalloc` / `Free` | raw device memory | no notion of a recipe |
| `synSectionSetPersistent` / `SetConst` / `SetRMW` (+ getters) | flags on **tensor data** sections | data, not program |
| `synSectionDestroy`, `synRecipeSectionHostBuffersClear` | host-side section lifecycle | host only |
| `synDeviceGetMemoryInfo` | device-wide free/total | **no attribution** |
| `synUpdateMemoryConsumption` | **new in 1.24.2**, absent from the 1.16 source | unknown semantics; not an allocation control |
| `synConfigurationSet/Get` | gcfg gateway | **refuses private keys** (`enableExperimental=false`) |
| `synLaunch` / `Ext` / `WithExternalEvents` | launch | this is what *causes* the download |

There is no load/unload, no pin, no evict, no residency set, no "make resident". **You cannot keep a
chosen set of recipes resident in Synapse at any version we can inspect.**

**Source corroboration** (*source read*, clean `6de1f66`): `synRecipeSectionGetProp` is
`synapse/include/synapse_api.h:470`, implemented at `synapse/src/runtime/common/syn_singleton_section.cpp:206-270`,
with props `SECTION_SIZE` / `SECTION_DATA` / `IS_CONST` (`synapse/include/synapse_api_types.h:49-52`). It
returns the const section's size and a **host** pointer (`constSections->data`), never a device
address, and for any non-const section id `SECTION_SIZE` returns `synFail` with `0`. The attribution
calls are `RECIPE_ATTRIBUTE_*` at `synapse/src/runtime/common/recipe/device_agnostic_recipe_processor.cpp:252-338` — and note that
`WORKSPACE_SIZE` there is only region 0 of three (`workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]`,
`device_agnostic_recipe_processor.cpp:213-236`), so even that pair under-reports a Gaudi1-style
workspace layout.

---

## 4. The Gaudi1 caveat that keeps getting misread as a Gaudi2 fact

There **is** a genuine device-HBM recipe cache in the tree — and it is the **Gaudi1 / qman** path, not
ours:

* `GCFG_RECIPE_CACHE_SIZE` (default **500 MiB** — `500 * 1024` KB, `MakePrivate`),
  `GCFG_RECIPE_CACHE_BLOCK_SIZE`
  (default **128 KB**, private), `RecipeCacheManager` with an LRU block allocator and per-recipe
  refcounts (`synapse/src/runtime/qman/common/recipe_cache_manager.cpp:43`: "Cache invalidation policy is LRU based").
* Allocated at device acquire from `qman/gaudi/device_gaudi.cpp` (`DeviceGaudi`, the Gaudi1 class), and
  the size check `RecipeVerification::verifyRecipeCacheSize` is gated on `devType == synDeviceGaudi`.
* The SCAL/Gaudi2 path (`DeviceScal`) references it **nowhere** — grep over
  `synapse/src/runtime/scal` for `RecipeCacheManager|RECIPE_CACHE` is empty. Consistent with what we
  measure: setting `ENABLE_EXPERIMENTAL_FLAGS=true RECIPE_CACHE_SIZE=262144` produces **no**
  recipe-cache log line and changes neither window on this Gaudi2 box.

**Source corroboration** (*source read*, clean `6de1f66`): the cache buffer is one device allocation
of `GCFG_RECIPE_CACHE_SIZE × 1024` bytes plus alignment, wrapped in a `RecipeCacheManager` with
`GCFG_RECIPE_CACHE_BLOCK_SIZE` blocks (`synapse/src/runtime/qman/common/device_recipe_addresses_generator.cpp:142-167`),
acquired per `(recipeId, allocationId)` with reference counts (`:219-260`) and released with
`setRecipeNotInUse` (`:90-110`). The defaults are `500 * 1024` KB and `128` KB
(`synapse/src/graph_compiler/habana_global_conf.cpp:1121-1133`). Same path also caches **per recipe handle**:
`m_recipeToMemoryMap[recipeHandle]` in `DeviceGaudi::allocateRecipeMemory`
(`synapse/src/runtime/qman/gaudi/device_gaudi.cpp:631-724`).

So: if anyone writes "set `RECIPE_CACHE_SIZE` to cache recipes in HBM", that is a **Gaudi1** statement.
On Gaudi2 there is no cache to size.

**And it is not `PT_HPU_RECIPE_CACHE_CONFIG`.** They are different caches, at different layers, on
different media:

* **`GCFG_RECIPE_CACHE_*` is Synapse's on-device cache**, of the *static part of a recipe*, and it is
  **device HBM**: `recipe_cache_manager.cpp:40` calls it *"on-device cache that stores recently used
  static part of recipes on pre allocated memory"*, and the allocation is
  `allocateMemory(cache_size + MANDATORY_KERNEL_ALIGNMENT, synMemFlags::synMemDevice, …)` at
  `synapse/src/runtime/qman/common/device_recipe_addresses_generator.cpp:139-160`. Gaudi1/qman only.
* **`PT_HPU_RECIPE_CACHE_CONFIG` is the bridge's disk cache** of serialised `.recipe` files — a
  directory, a `max_dir_size_mb`, and the `libs_env_hash` key §7–§8 are about.

The cache-layer table in [`recipe-launch-notes.md`](recipe-launch-notes.md) §4 listed the Synapse row
as "disk"; it is device HBM, and that row now says so.

---

## 5. What this means for the manual's performance model

1. **Program download is a per-launch cost.** It is why graph breaks hurt beyond the CPU side: two
   one-node graphs pay two downloads, two long-SO ring waits, two enqueues; one two-node graph pays
   one. **Eager mode is the limit case of this, not an escape from it**: `PT_HPU_LAZY_MODE=0` still
   compiles *every op* to its own one-node recipe — there is no region recipe to be resident — so N ops
   is N recipes, N submissions and N program downloads. Eager is **worse** for the residency story,
   not cheaper. (**source read**: `graph::compile()` always calls `synGraphCompile`, and `eager_mode_`
   only chooses `synGraphCreateEager` over `synGraphCreate` and otherwise touches bookkeeping and the
   recipe name — `backend/synapse_helpers/graph.cpp:140-146,643,646,685`. **MEASURED**: 768 TPC kernel
   events for one eager `y.copy_(x)`, `02-regions.md:1198`.) The *first* call is size-independent
   (103.6 ms at 1 K elements and at 1 M) and the steady-state call is ~0.5–0.7 ms — the per-launch
   cost is dominated by the fixed path, not by program bytes, until the program gets large.
   (**MEASURED**, `notes/recipe-launch-notes.md` §0, `experiments/01_one_region.py`.)
2. **Bigger regions amortise better, up to the window.** Keep a region inside the window; past it the
   launch fails outright.
3. **Never count on "it's compiled, so it's warm in HBM".** The only thing "compiled" buys is that the
   program exists on the host side to be downloaded.
4. **Attribute device memory with `synRecipeGetAttribute`**, not by guessing — it reports
   `DEVICE_MEMORY_SIZE` / `HOST_MEM_SIZE` / `WORKSPACE_SIZE` per recipe.
5. **Two knobs, two resources, never conflate:** the bridge LRU (`HABANA_PGM_LRU_MAX`) bounds *host*
   memory; `HBM_GLOBAL_MEM_SIZE_MEGAS` (private, env-only) bounds the *device* program window.

---

## 6. Repro / verification recipe (for the "fast iteration" chapter)

```bash
# See the program windows of the runtime you are actually on. Printed once, at device acquire.
HABANA_LOGS=logs \
LOG_LEVEL_ALL=6 LOG_LEVEL_SYN_PROG_DWNLD=0 LOG_LEVEL_SYN_DEVICE=0 \
  python your_probe.py
grep -h 'ArcHbmMemMgr\|HbmGlblMemMgr' logs/* | sed 's/.*\] //'
```

Provenance habits worth keeping:
`dpkg -V habanalabs-graph` (am I on the packaged runtime?) ·
`nm -D --defined-only /usr/lib/habanalabs/libSynapse.so | grep <sym>` (does the mechanism still
exist?) · `strings -a … | grep '<format string>'` (cheap existence proof for a mechanism) ·
`/proc/self/maps` from inside the process (which library did I *actually* load?).

---

## 7. One `@torch.compile` function → how many graphs, how many recipes? (MEASURED)

Cold recipe cache per case, so the `.recipe` count is unambiguous. torch 2.13 / bridge 1.24.2.

| case | body | dynamo graphs | breaks | ops (all graphs) | **recipes** |
|---|---|---|---|---|---|
| A | `t + 1` | 1 | 0 | 1 | **1** |
| B | `(t + 1) * 2 - 3` | 1 | 0 | 3 | **1** |
| C | `y = t + 1; s = y.sum().item(); return y + s` | **2** | **1** | 3 | **2** |
| D | `y = t + 1; return y[: t.shape[0] // 2] * 2` | 1 | 0 | 3 | **1** |

The `ops` column counts every op in the function **across all its graphs** (case C is `add`, `sum`,
`add` — the `.item()` that causes the break is a transfer, not an op), which is why B, C and D are all
3 and only C needs two recipes.

Readings:

* **A `@torch.compile` function does not create one graph.** It creates one FX graph per dynamo
  *frame*, and a graph break splits that frame into several graphs. Case C is the minimal
  demonstration: one decorated function, two graphs, **two recipes**.
* **Ops do not multiply recipes.** B and D are three ops each and still one recipe. A recipe is one
  per maximal contiguous run of *clusterable* nodes — so the count tracks boundaries, not op count.
* The same function called at a new shape or with a new traced `int` is a new frame → new graphs →
  new recipes. That is the specialization arithmetic, seen at the recipe level.
* The recipe filename is `<RecipeArgumentSpec hash>_<libs_env_hash>_syn<driver version>`
  (`backend/kernel/hpu_recipe_cache.cpp:294-308`; the store that makes it the cache key is `:317`).
  The leading field is the graph itself. The trailing
  `_syn1.24.2.79e284b` is the **driver** version string from `synDriverGetVersion`, so a driver or
  Synapse upgrade does change every filename. The **middle field is not a hash of the libraries or the
  environment** — it is a constant. That is §8, and it is the one finding here you have to act on.

### The chain, named

`@torch.compile` → **dynamo** (frame → FX graph(s), one per frame, split at breaks) →
`hpu_backend(graph_module, example_inputs)` (registered in
`python_packages/habana_frameworks/torch/dynamo/compile_backend/backends.py`) → **`aot_autograd`**
with `partition_fn=hpu_partition` and `inference_compiler=hpu_inference_compiler` → **bridge FX
passes** (`python_packages/habana_frameworks/torch/dynamo/compile_backend/_passes/`) → per-node
**placement** (`hpu_cluster` / `eager`) → for each
maximal clusterable run: a Synapse graph → `synGraphCompile` → **one recipe** → each execution is one
`synLaunch` → **one submission**, which re-downloads the program into the 64-chunk ring (§1–§2).

So the three counts a reader must keep apart: **graphs** (dynamo) ≥ **recipes** (clusterable runs) ≥
**submissions** (launches). The tldr's "a recipe/handle count is not a submission count" is the same
distinction one level down.

---

## 8. What the recipe cache cannot see: `libs_env_hash` is dead code

The disk cache's filename **is** its key — `recipe_cache_.store(hashCode + cache_id_suffix_, …)` at
`backend/kernel/hpu_recipe_cache.cpp:317` — and the suffix is built at `:294-308`:

```
<RecipeArgumentSpec::hashCode()>_<CacheVersion::libs_env_hash()>_syn<synDriverGetVersion()>
```

The middle field is meant to change when the **kernel library** or a compiler flag changes, so that a
rebuild misses the cache. **It does not.** Both halves of it are dead, for two independent reasons.

**The file half.** `hash64_file_content()`
(`pytorch_helpers/habana_helpers/habana_serialization/cache_version.cpp:100-139`) declares
`size_t hashRes{0}` at `:107` and then a **second, shadowing** `size_t hashRes{0}` inside the `mmap`
branch at `:123`. The loop at `:126-127` fills the **inner** one; `:138` returns the **outer** one. It
therefore returns **0 for every file, whatever its size or contents** — the `mmap` and the whole
per-byte loop are wasted work. (**source read**:
`pytorch_helpers/habana_helpers/habana_serialization/cache_version.cpp:107,123,126-127,138`.)

**The env half.** `check_env_fo_hashing()` (`cache_version.cpp:141-172`) decides with `env_var.find(v)`
(`:170`) whether an environment entry is on its list. The needles are **space-padded to a column** so
they line up in the source — `"ENABLE_RAGGED_SOFTMAX_OPT             "` (`:150`) — and the haystack is a
`"NAME=VALUE"` string taken from `environ` (`:208-215`). `"NAME=VALUE"` contains no run of spaces, so
**no needle ever matches**: no environment variable has ever been hashed. (**source read**:
`pytorch_helpers/habana_helpers/habana_serialization/cache_version.cpp:141-172,150,165,170,208-215`.)

So `libs_env_hash()` (`cache_version.cpp:174-222`) is `at::hash_combine` applied N times to 0. It
varies with the *number* of files it was asked to hash and with nothing else. (That number is 2 or 3 —
`habana_device_path()`, the Synapse library, and one `GC_KERNEL_PATH` entry. The multi-path branch at
`:186-199` splits on `:`, but a colon-list `GC_KERNEL_PATH` is **rejected at device init** —
`FATAL ERROR … Path is not a regular file`, exit 134 — so the branch is unreachable in practice.)

### 8.1 Measured: three arms, one filename

One graph (`x * 2.0 + 1.0`, `fullgraph=True`, `dynamic=False`), a **fresh cache directory per arm**,
`PT_HPU_RECIPE_CACHE_CONFIG=<dir>,true,1024,false`. All three arms produced the **identical** file:

| arm | `GC_KERNEL_PATH` | environment | recipe file |
|---|---|---|---|
| 1 | the stock `libtpc_kernels.so` | — | `4757981558864380105_a16fb581eee_syn1.24.2.79e284b.recipe` |
| 2 | a copy of the same library with **one byte appended** | — | **identical** |
| 3 | the stock library | `DISABLE_TENSORS_PINNING=true` — a name on the hashed list (`:165`) | **identical** |

`_a16fb581eee_` is also the field in every filename we have ever produced, including
`recipe-launch-notes.md` §0's. A constant is exactly what a hash of "0, 0, 0" looks like. **MEASURED.**

### 8.2 The consequence is the opposite of the obvious one

The cache cannot see a kernel edit or a configuration change. So:

1. **Editing a TPC kernel and rebuilding will serve the stale recipe.** The name is unchanged, the file
   is there, the graph is the same graph. When you iterate on a kernel, **clear the cache directory**
   — or set the config's fourth field, `delete_on_init`, to `true` — as part of the rebuild.
2. **A cache directory shared across two configurations silently mixes recipes** built under
   different compiler flags: two builds of one kernel library, one directory, one key.
3. **Maintain your own key: one cache directory per *(build, configuration)* pair.** Nothing in the
   filename provides that separation, so it has to come from you. `delete_on_init=true` is the
   alternative if you would rather pay a rebuild every process.
4. **`PT_RECIPE_CACHE_IGNORE_VERSION=1`** drops the `_<hash>_syn<version>` suffix altogether
   (`hpu_recipe_cache.cpp:299-301`). It exists, it makes the sharing worse, and it is a reminder that
   this suffix is the cache's only version guard.

**What does still invalidate:** `RecipeArgumentSpec::hashCode()` — the graph, i.e. ops, shapes and
attributes — and the driver version string. A **Synapse or driver upgrade** changes every filename; a
**kernel-library rebuild does not**. Two different things are called "the build", and only one of them
is in the key.

### 8.3 What we could not show, and why it matters

The stale-serving half is **INFERRED**, not measured end to end. In this bridge configuration the disk
cache did not *hit* even on four identical repeated runs: each run recompiled (first call 218–230 ms,
against ~140 ms cold and ~77 ms with a recipe already on disk, `recipe-launch-notes.md` §0) and rewrote
the `.recipe` — whose size varied between runs (3366 B, 3144 B). So the hazard is a property of a cache
that *does* hit, and the launch notes record one that does (`recipe-launch-notes.md` §4: "0 new recipes
on a same-config reboot"). What is unconditional, and what this section establishes, is that **the key
cannot change when the kernel library or a hashed environment variable changes**. Whether a given
configuration then serves the old bytes is the disk cache's business — and its key contains nothing
else. **UNKNOWN:** why it did not hit here; a per-process component inside the serialised
`RecipeArgumentSpec` is the obvious suspect.

---

## 9. Dynamic shapes: the constraint is the **bridge**, not the hardware (MEASURED)

The manual's first fact is *"there are no dynamic shapes"*. That is true of the stack we ship and
**false as a statement about the device stack**. Everything below is measured at the **Synapse API
level** — below the bridge and below dynamo.

### 9.1 A dynamic recipe is genuinely size-parametric, including for GEMM

Graph: `C = A[M,K] @ B[K,N]` (MME) → `Y = C + bias` (TPC), every dimension declared
`DATA_TENSOR_DYNAMIC` with a `[min,max]` bound, one recipe handle.

| arm | result |
|---|---|
| vary M (512→64→8), N and K fixed | exact; the write extent tracks M |
| vary N (512→64→8), M and K fixed | exact; the write extent tracks N |
| **vary K (512→256→64→8)** | **exact** — `Y[0] − 1 == 8` at K=8, so the K-fold sum really ran over 8 terms |
| rectangular 64×100×32, 17×17×17, 1×1×1 | exact |
| 7 shapes on one handle, one process | all exact; a rejected launch does not poison the handle |
| outside the declared `[min,max]` | **rejected, `synStatus 35`** — not silently computed |

**The control that makes this mean something.** The same graph compiled **statically** and launched at
M=64 (compiled for 512) returned **`synSuccess`** and wrote **262,144** elements — 229,376 sentinel
violations. A static recipe silently ignores the launch size, and the harness demonstrably *can* see
that. The PASS rows are therefore real, not a harness that cannot fail.

**Evidence the program is re-parameterised rather than masked:** `m_mappedSizePatch` is **263,264 B
(87 % of the image)** for the DSD recipe against **208 B (0.07 %)** for the static one; 10 of 112 blobs
patched, against 9 of 23.

### 9.2 Exactness is not a schedule

Nothing above says the MME is **re-tiled** for the smaller shape rather than **re-parameterised at the
tiling chosen for the compiled maximum**. A 64×64×64 GEMM run at a 512×512×512 tiling would be exact
and slow. **So "dynamic shapes" may be true for correctness and still be "one big bucket" for
throughput.** That is **UNKNOWN**, and it is the question that decides usability: measure per-size
efficiency of a DSD recipe against static-per-shape before recommending it.

### 9.3 The trap: a user-section intermediate breaks DSD

Put the MME output that feeds the TPC node into an **application section** — persistent or not — and the
recipe fails at the *exact compiled shape* with `Failed to perform DSD SIF patching` / status 35. Each
node alone patches fine; the trigger is specifically the **MME→TPC edge into a user section**. **Keep
intermediate tensors compiler-owned.**

### 9.4 "No buckets, just dynamic sizes" — achieved, and the catch

One serialised recipe (48,476 B), deserialised in a fresh process, ran `add1_fwd_f32_gaudi2` at **16
sizes from N=1 to N=1024** bit-exactly, writing exactly N elements each time — including 1, 3, 511 and
513. Declared `min=1, max=1024`, so there is no meaningful lower bound.

| arm | outcome |
|---|---|
| `min=1, max=1024`, 16 sizes | all PASS; `wrote == N`; sentinel intact at `Y[N]` |
| static control (`DATA_TENSOR`, max 1024) | writes **1024 at every N** — silently ignores `tensorSize` |
| `min=256` | 1, 3 and 255 rejected (`35`); 256 and above pass |
| above max (1025, 2048) | rejected (`35`) |
| **`ENABLE_WIDE_BUCKET` alone** | **refused by the config loader** — internal key, needs `ENABLE_EXPERIMENTAL_FLAGS` |
| **`ENABLE_WIDE_BUCKET` + `ENABLE_EXPERIMENTAL_FLAGS`** | **bound removed and SILENTLY WRONG** — 138 elements written at N=1 |
| `SHAPE_TENSOR` input | compiles and serialises, but cannot carry N; launch → `30` |
| `DEVICE_SHAPE_TENSOR` input | `synNodeCreate` → `1` |
| base kernel library (no DS bit) | `synNodeCreate` → `26`, *"TPC kernel with guid … doesn't support DS"* |
| DS bit set, no SIF | `synGraphCompile` → `26`, *"shape inference … inconsistent result, inferred sizes [0]"* |

Four findings:

1. **DSD is a kernel-library declaration, not a caller option.** `GetKernelGuids` must set
   `supportsDynamicShapes = 1` **and** a non-zero `nameHash.hashValue` (the SIF id), and
   `GetShapeInference` must actually resize the output. **All three are mandatory.** The upstream
   `add1` kernel has none of them, so "make add1 shape-parametric" means **editing the kernel
   library**, not changing how it is called.
2. **`ENABLE_WIDE_BUCKET` is a footgun.** It removes the lower-bound *check* without making the recipe
   correct: at N=1 it wrote 138 elements and returned success. **Declare `min=1` instead. Never relax
   the bound.**
3. **The maximum is structural.** It sizes descriptors, the index space, data chunks and the workspace
   (8448 B for the dynamic recipe, 256 B static). "Dynamic up to a declared capacity" is the strongest
   guarantee available.
4. **The tail is exact without masking.** The kernel has no tail mask (`depthStep = 64`) yet writes
   exactly 511 and 513. The static control shows the bound comes from the per-launch descriptor, not
   from the kernel. Mechanism **INFERRED**, not isolated.

### 9.5 Where each constraint lives

| claim | layer | status |
|---|---|---|
| M, N and K are all patchable on the MME | **Synapse** | **MEASURED**, with a static control |
| a static recipe ignores the launch size | **Synapse** | **MEASURED** |
| a DSD recipe is exact inside its declared bound and rejects outside it | **Synapse** | **MEASURED** |
| what we ship is one static recipe per shape | **bridge** | **MEASURED** at the recipe level (§7) |
| "`dynamic=True` is turned into static compilation" | **bridge** | **ASSUMED (vendor)** — that is how `01-basics.md` §1.4.1 already tags it, and it is the right tag: it is the standard account, not something we instrumented |
| the bridge *has* a dynamic path at all | **bridge** | **source read** — `get_dynamic_config_value()` reads dynamo's `dynamic=True` (`python_packages/habana_frameworks/torch/dynamo/compile_backend/passes.py:509-522`, the read is `:520`) and the DS/SIF path is gated on `optim_output_sif_ds` (`passes.py:2216-2222`). We have never seen it produce a dynamic recipe on the default path |
| dynamo can emit symbolic graphs | **dynamo** | **source read** — it is `torch._dynamo.config.assume_static_by_default` that the bridge reads (`passes.py:520`) |
| the HPU itself | — | **UNKNOWN** — nothing here is below Synapse |

**Read the row, not the headline.** "There are no dynamic shapes" is a **bridge** statement, tagged
ASSUMED as a mechanism and MEASURED as a recipe count. It is the right operating rule — the bridge will
not hand you a dynamic recipe on any path we have seen, so every new shape is a new program — and the
wrong conclusion about the device. Do not infer a hardware limit from a layer above it; and do not let
this section replace the fixed-shape discipline until §9.2 is measured.

**Open:** per-size efficiency (§9.2); DSD launch cost against static; why the user-section case fails
SIF; generality to other dtypes, ops, ranks and multiple bounds; whether an *unbounded* maximum is
possible (expected no — the compiler must size descriptors for the largest shape).

---

## 10. Open questions

1. Recover the per-stream vs total split of the logged window sizes (the log prints the pre-allocation,
   which is `Size × 3`). One probe with a single compute stream would settle it.
2. Confirm `NUM_BUFF = 64` in 1.24.2 by disassembly or a 1.24.2 source drop. It is **INFERRED** from
   the **global** window at three settings of `HBM_GLOBAL_MEM_SIZE_MEGAS` — one method, one window —
   and 1.16 is `16` (`synapse/src/runtime/scal/common/recipe_launcher/nbuff_allocator.hpp:28`).
3. What sizes the **arc** window (127.5 MiB, unaffected by `HBM_GLOBAL_MEM_SIZE_MEGAS`), and where does
   the 1.5 MiB between 64 chunks and the window go (§2a)? A 1.24.2 source drop, or a log line that
   prints `m_size`, settles both.
4. `synUpdateMemoryConsumption` (new in 1.24.2) — semantics unknown; could be the missing attribution
   hook.
5. Re-read the jchen `seqs=8` "dies on a 24 MB allocation" failure against §2 — 24 MB is the same order
   as one 3 MiB global chunk × 8, or a few arc chunks.
6. Measure steady-state launch cost for a *multi-MB* program vs a tiny one, to split "ring walk +
   long-SO" from "PDMA bytes". Everything so far is on programs small enough to be free.
7. **MME descriptors have no separate allocation we can find** in the 1.16 tree: the only
   program-data registrations are TPC kernels (`synapse/src/graph_compiler/habana_nodes/tpc_node.cpp:582,593`), the TPC printf buffer
   (`synapse/src/graph_compiler/tpc_kernel_loader.cpp:419`) and H2D/static tensors (`synapse/src/graph_compiler/passes/allocate_tensors.cpp:306,330`). **INFERRED:**
   they ride the ECB/dynamic-blob path and therefore live in ARC HBM. Whether 1.24.2 allocates a
   descriptor buffer of its own is **UNKNOWN**.
8. Why the disk recipe cache did not hit across four identical runs (§8.3) — a per-process field in the
   serialised `RecipeArgumentSpec`, or something else? This is the difference between "the stale-recipe
   hazard is live" and "the disk cache never serves anything on this configuration".

---

## 11. The `00-tldr.md` change applied alongside this note

`AGENTS.md` requires the TLDR to define *"lazy vs. eager vs. compiled, graphs vs. recipes"*, and it
defined none of them. The same branch adds those rows, the layer tag on the static-shape fact and the
`eager` / `eager placement` disambiguation to [`00-tldr.md`](../00-tldr.md). **Read the TLDR, not this
section** — this note is the evidence behind it, and a change log here would rot the moment the TLDR
moved.
