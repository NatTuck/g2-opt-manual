# Recipe residency and the device memory budget

**Status:** working notes for our own team, not a manual chapter. Written 2026-11-13 from measured
probes on the runtime we actually execute on (1.24.2 / Gaudi2) plus a read of an old Synapse source
snapshot. The measurements are cheap and repeatable — §6 has the recipe.

**Provenance.**

- **Runtime:** `/usr/lib/habanalabs/libSynapse.so`, package `habanalabs-graph 1.24.2-481`, Synapse
  build `79e284b` (matches the `_syn1.24.2.79e284b.recipe` filenames we generate). Driver
  `1.24.2-52a555d`, torch `2.13.0a0+git2469d18`, `habana-torch-plugin 1.24.2`.
- **Is it the clean vendor binary?** `dpkg -S` says the package owns it, `dpkg -V habanalabs-graph`
  reports no modification, and `nm -D` shows **no** `nativeCompute*` symbols.
- **Does a normal process really load it?** `LD_LIBRARY_PATH` puts `/usr/lib/habanalabs` ahead of any
  `RUNPATH`; verified by reading `/proc/self/maps` from inside a `python` + `torch_hpu` process.
- **Source available for reading:** an **old 1.16 snapshot** (base commit `6de1f66`), from a
  third-party mirror of the vendor tree. Extract a clean copy with:

  ```bash
  git archive 6de1f66 | tar -x -C <dir>
  ```

  The working tree at `~/Envs/onecat/src/synapse` is that base **plus three local patches** and must
  not be cited. Everything tagged *source read* below is from the clean extraction of `6de1f66`.

**Basis tags.** **MEASURED** (ran here, artefact named) · **source read** (a named file and line;
1.16 unless stated) · **ASSUMED** · **INFERRED** (derivation stated) · **UNKNOWN**. The tag
vocabulary is the one defined in the [README](../README.md).

**Version caveat.** The source snapshot is **1.16**; the runtime is **1.24.2**. Source reads tell us
structure and mechanism, and each claim that matters is corroborated against the shipped binary
(`nm -D`, `strings -a`) or against a measurement. Where a number comes from the 1.16 source rather
than from this host, it says so.

Two traps that cost us time and belong in the "fast iteration" chapter:

1. **The onecat tree is patched.** It carries an uncommitted `synNativeComputeGraph*` "research API"
   (24 occurrences) whose comment claims it "skips recipe preparation, address patching, PDMA
   download and per-recipe synLaunch", and a `NativeRecipeMemory` / `m_nativeRecipeMemory` /
   `arcCoreAddress()` device-resident-program path. **Neither exists in vendor Synapse** — zero in
   the clean tree, zero `nativeCompute`/`NativeRecipeMemory` symbols in the 1.24.2 binary. Two
   different residency models live in this workspace; do not mix them.
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
  (**source read**, 1.16: `src/runtime/common/recipe/recipe_allocator.cpp:15-30`,
  `src/runtime/common/memory_utils/memory_allocator_utils.cpp:12-40`.) The buffers allocated this way
  are exactly the ones the device walks: `execution_blobs_buffer`, `patching_blobs_buffer`,
  `dynamic_blobs_buffer`, both ECB command buffers, and `program_data_blobs_buffer`
  (`recipe_serializer.cpp:496/504/514/599/607/643`; `recipe_blob.cpp:452/457/462`;
  `recipe_ecb.cpp:418/434`).
* **Every `synLaunch` downloads the program again.** `RecipeLauncher::launch()` calls
  `prepareForDownload()` then `downloadToDev()` unconditionally — no "already resident" branch. Its
  own doc comment lists the steps: *"Download to device HBM (including patching) … enqueue to device
  (sends packets with recipe location)"*. `downloadToDev` does `memcpyToMapped` → patch →
  **`pdmaDownload`** (mapped host → HBM, by CpDma commands) → fence wait → `scalEnqueue`.
* **The destination is a rotating ring, not a cache.** `ArcHbmMemMgr` / `HbmGlblMemMgr` derive from
  `NBuffAllocator`, which splits a fixed window into `NUM_BUFF` chunks and hands out the next one each
  time, returning *"an offset and the long-sync-object to wait for"*; `prepareForDownload` then does
  `longSoWaitOnDevice(...)` — *wait until this slot is free on the device*. (**source read**, 1.16:
  `NUM_BUFF = 16` at `src/runtime/scal/common/recipe_launcher/nbuff_allocator.hpp:28`; chunk size is
  `size / NUM_BUFF` at `hbm_global_mem_mgr.cpp:39-43`. On 1.24.2 the count measures **64** — §2.)
* The vendor says it in as many words (`arc_hbm_mem_mgr.hpp`, repeated verbatim in
  `hbm_global_mem_mgr.hpp`):

  > This class is used to allocate memory of the Arc-HBM for running a recipe. It is currently
  > implemented in a very simple way, **assuming no caching**. It splits the given memory into chunks
  > (downloadedRecipes) and each time assigns a new chunk. … **In the future, we can consider adding
  > caching here to avoid the pdma**

* **`synRecipeDestroy` frees host memory only.** `recipeDestroy` → `notifyRecipeRemoval` →
  `removeRecipeHandle` → `destroyRecipeHandle`; no device free anywhere, because the recipe never
  owned device memory. **An unlaunched compiled recipe holds zero HBM.** (**source read**, 1.16:
  `src/runtime/common/recipe/recipe_handle_impl.cpp:18-24` frees only the `RecipeAllocator`.)
* The bridge's `RecipeCacheLRU` (`HABANA_PGM_LRU_MAX`, defaults 100 000 eager / 30 000 lazy) holds
  `shared_ptr<RecipeHolder>` around live `synRecipeHandle`s and evicts by **host** memory
  (`get_recipe_host_mem_size()`, threshold `PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT`), calling
  `synRecipeDestroy` on the GC thread. It bounds **host** memory. It has nothing to do with HBM.
* The workspace is not per-recipe either: `synLaunch` takes `pWorkspace` from the caller, and the
  bridge keeps **one device-wide** workspace buffer. (**source read**, 1.16: the workspace is an
  `[in]` argument of `synLaunch`, `include/synapse_api.h:612-617`.)

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
| chunks per window | **64** | `NUM_BUFF = 16` | confirmed three independent ways: 192/3, 96/1.5, 48/0.75 MiB |
| arc ("shared") window | `0x7F80000` = **127.5 MiB**, chunk `0x1F8000` ≈ 1.97 MiB | 32 MiB, chunk size/16 | not moved by any key we found |
| global window | `0xC000000` = **192 MiB**, chunk `0x300000` = 3 MiB | `HBM_GLOBAL_MEM_SIZE_MEGAS` × 1 MiB | see scaling below |
| compute streams | 3 (`MaxNumOfComputeStreams`, unchanged) | 3 | the pre-allocation is `Size × 3` |
| max single-recipe program | the whole window (16 chunks in 1.16; 64 now) — beyond it the launch **throws**, it does not spill | | |

**Source corroboration for the 1.16 column** (all *source read*, clean `6de1f66`):
`MaxNumOfComputeStreams = 3` (`src/runtime/scal/common/entities/scal_dev.hpp:58`);
`hbmSharedSize = 32 * 1024 * 1024` (`src/runtime/scal/common/device_scal.hpp:126`);
`HBM_GLOBAL_MEM_SIZE_MEGAS` default **64** (`src/graph_compiler/habana_global_conf.cpp:594-598`);
the pools are allocated as `Size × MaxNumOfComputeStreams` (`device_scal.cpp:242-247`).

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

**All MEASURED** on 1.24.2 / Gaudi2, one card.

| item | size | when | scales with recipe count? |
|---|---|---|---|
| card HBM (as Synapse reports it) | 97376 MiB = 95.09 GiB | — | no |
| card HBM (as torch reports it) | 96895.2 MiB = **94.62 GiB** | after acquire | no |
| ↳ global program window | **192 MiB** = 3 × `HBM_GLOBAL_MEM_SIZE_MEGAS`(64) | device acquire, before any tensor | **no** |
| ↳ arc program window | **127.5 MiB** (`0x7F80000`) | device acquire | **no** |
| ↳ streams, monitors, TDR, profiler | **~161 MiB** (remainder) | device acquire | no |
| **total reserved before your first tensor** | **~481 MiB (0.47 GiB)** | | **no** |
| your workspace | `synWorkspaceGetSize`, **one** device-wide buffer in the bridge | first launch | no (sized by the max, not the count) |
| your tensors (weights, KV cache, activations) | the rest | you | you |

**The recipe count does not appear in this table, and that is the measurement.** 8 distinct recipes,
each compiled *and launched*, moved free HBM by **0.0 MiB** (per-step readings identical to the
0-recipe baseline, `96889.7 MiB` free before and after). Compiling more graphs costs host memory and
disk, not HBM.

Practical rule for the manual:

> Budget = 96 GB − ~0.5 GB (fixed runtime reservation) − workspace − weights.
> **Do not subtract a per-recipe term.** If HBM is filling up as the graph count grows, the recipes
> are not what is filling it — look at persistent/const sections, the workspace being sized by the
> largest recipe, or tensors retained by the host-side cache.

Two footnotes that matter:

* `synDeviceGetMemoryInfo()` on this build returns **`free = 0`** (status synSuccess, total correct).
  It is not usable for this; `torch.hpu.mem_get_info()` is.
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
> (`PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT`, default 90 %), in 512 MB chunks, calling `synRecipeDestroy`
> on the GC thread.

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
`include/synapse_api.h:470`, implemented at `src/runtime/common/syn_singleton_section.cpp:206-270`,
with props `SECTION_SIZE` / `SECTION_DATA` / `IS_CONST` (`include/synapse_api_types.h:49-52`). It
returns the const section's size and a **host** pointer (`constSections->data`), never a device
address, and for any non-const section id `SECTION_SIZE` returns `synFail` with `0`. The attribution
calls are `RECIPE_ATTRIBUTE_*` at `device_agnostic_recipe_processor.cpp:252-338` — and note that
`WORKSPACE_SIZE` there is only region 0 of three (`workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE]`,
`device_agnostic_recipe_processor.cpp:213-236`), so even that pair under-reports a Gaudi1-style
workspace layout.

---

## 4. The Gaudi1 caveat that keeps getting misread as a Gaudi2 fact

There **is** a genuine device-HBM recipe cache in the tree — and it is the **Gaudi1 / qman** path, not
ours:

* `GCFG_RECIPE_CACHE_SIZE` (default **500 MB**, `MakePrivate`), `GCFG_RECIPE_CACHE_BLOCK_SIZE`
  (default **128 KB**, private), `RecipeCacheManager` with an LRU block allocator and per-recipe
  refcounts (`recipe_cache_manager.cpp:43`: "Cache invalidation policy is LRU based").
* Allocated at device acquire from `qman/gaudi/device_gaudi.cpp` (`DeviceGaudi`, the Gaudi1 class), and
  the size check `RecipeVerification::verifyRecipeCacheSize` is gated on `devType == synDeviceGaudi`.
* The SCAL/Gaudi2 path (`DeviceScal`) references it **nowhere** — grep over
  `synapse/src/runtime/scal` for `RecipeCacheManager|RECIPE_CACHE` is empty. Consistent with what we
  measure: setting `ENABLE_EXPERIMENTAL_FLAGS=true RECIPE_CACHE_SIZE=262144` produces **no**
  recipe-cache log line and changes neither window on this Gaudi2 box.

**Source corroboration** (*source read*, clean `6de1f66`): the cache buffer is one device allocation
of `GCFG_RECIPE_CACHE_SIZE × 1024` bytes plus alignment, wrapped in a `RecipeCacheManager` with
`GCFG_RECIPE_CACHE_BLOCK_SIZE` blocks (`src/runtime/qman/common/device_recipe_addresses_generator.cpp:142-167`),
acquired per `(recipeId, allocationId)` with reference counts (`:219-260`) and released with
`setRecipeNotInUse` (`:90-110`). The defaults are `500 * 1024` KB and `128` KB
(`src/graph_compiler/habana_global_conf.cpp:1121-1133`). Same path also caches **per recipe handle**:
`m_recipeToMemoryMap[recipeHandle]` in `DeviceGaudi::allocateRecipeMemory`
(`src/runtime/qman/gaudi/device_gaudi.cpp:631-724`).

So: if anyone writes "set `RECIPE_CACHE_SIZE` to cache recipes in HBM", that is a **Gaudi1** statement.
On Gaudi2 there is no cache to size.

---

## 5. What this means for the manual's performance model

1. **Program download is a per-launch cost.** It is why graph breaks hurt beyond the CPU side: two
   one-node graphs pay two downloads, two long-SO ring waits, two enqueues; one two-node graph pays
   one. Our earlier probe showed the *first* call is size-independent (103.6 ms at 1 K and at 1 M
   elements) and the steady-state call is ~0.5–0.7 ms — the per-launch cost is dominated by the fixed
   path, not by program bytes, until the program gets large.
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
HABANA_LOGS=/tmp/logs \
LOG_LEVEL_ALL=6 LOG_LEVEL_SYN_PROG_DWNLD=0 LOG_LEVEL_SYN_DEVICE=0 \
  python your_probe.py
grep -h 'ArcHbmMemMgr\|HbmGlblMemMgr' /tmp/logs/* | sed 's/.*\] //'
```

Provenance habits worth keeping:
`dpkg -V habanalabs-graph` (am I on the packaged runtime?) ·
`nm -D --defined-only /usr/lib/habanalabs/libSynapse.so | grep <sym>` (does the mechanism still
exist?) · `strings -a … | grep '<format string>'` (cheap existence proof for a mechanism) ·
`/proc/self/maps` from inside the process (which library did I *actually* load?).

---

## 7. One `@torch.compile` function → how many graphs, how many recipes? (MEASURED)

Cold recipe cache per case, so the `.recipe` count is unambiguous. torch 2.13 / bridge 1.24.2.

| case | body | dynamo graphs | breaks | ops | **recipes** |
|---|---|---|---|---|---|
| A | `t + 1` | 1 | 0 | 1 | **1** |
| B | `(t + 1) * 2 - 3` | 1 | 0 | 3 | **1** |
| C | `y = t + 1; s = y.sum().item(); return y + s` | **2** | **1** | 2 | **2** |
| D | `y = t + 1; return y[: t.shape[0] // 2] * 2` | 1 | 0 | 3 | **1** |

Readings:

* **A `@torch.compile` function does not create one graph.** It creates one FX graph per dynamo
  *frame*, and a graph break splits that frame into several graphs. Case C is the minimal
  demonstration: one decorated function, two graphs, **two recipes**.
* **Ops do not multiply recipes.** B and D are three ops each and still one recipe. A recipe is one
  per maximal contiguous run of *clusterable* nodes — so the count tracks boundaries, not op count.
* The same function called at a new shape or with a new traced `int` is a new frame → new graphs →
  new recipes. That is the specialization arithmetic, seen at the recipe level.
* The recipe filename carries `_<libs_env_hash>_syn1.24.2.79e284b` — the environment hash and the
  Synapse build. A different build or a changed `GC_KERNEL_PATH` changes every filename.

### The chain, named

`@torch.compile` → **dynamo** (frame → FX graph(s), one per frame, split at breaks) →
`hpu_backend(graph_module, example_inputs)` (registered in
`habana_frameworks/torch/dynamo/compile_backend/backends.py`) → **`aot_autograd`** with
`partition_fn=hpu_partition` and `inference_compiler=hpu_inference_compiler` → **bridge FX passes**
(`dynamo/compile_backend/_passes/`) → per-node **placement** (`hpu_cluster` / `eager`) → for each
maximal clusterable run: a Synapse graph → `synGraphCompile` → **one recipe** → each execution is one
`synLaunch` → **one submission**, which re-downloads the program into the 64-chunk ring (§1–§2).

So the three counts a reader must keep apart: **graphs** (dynamo) ≥ **recipes** (clusterable runs) ≥
**submissions** (launches). The tldr's "a recipe/handle count is not a submission count" is the same
distinction one level down.

---

## 8. Open questions

1. Recover the per-stream vs total split of the logged window sizes (the log prints the pre-allocation,
   which is `Size × 3`). One probe with a single compute stream would settle it.
2. Confirm `NUM_BUFF = 64` in 1.24.2 by disassembly or by a 1.24.2 source drop (currently inferred from
   three consistent chunk-size ratios; 1.16 is `16`, `nbuff_allocator.hpp:28`).
3. What sizes the **arc** window (127.5 MiB, unaffected by `HBM_GLOBAL_MEM_SIZE_MEGAS`)?
4. `synUpdateMemoryConsumption` (new in 1.24.2) — semantics unknown; could be the missing attribution
   hook.
5. Re-read the jchen `seqs=8` "dies on a 24 MB allocation" failure against §2 — 24 MB is the same order
   as one 3 MiB global chunk × 8, or a few arc chunks.
6. Measure steady-state launch cost for a *multi-MB* program vs a tiny one, to split "ring walk +
   long-SO" from "PDMA bytes". Everything so far is on programs small enough to be free.
7. **MME descriptors have no separate allocation we can find** in the 1.16 tree: the only
   program-data registrations are TPC kernels (`tpc_node.cpp:582,593`), the TPC printf buffer
   (`tpc_kernel_loader.cpp:419`) and H2D/static tensors (`allocate_tensors.cpp:306,330`). **INFERRED:**
   they ride the ECB/dynamic-blob path and therefore live in ARC HBM. Whether 1.24.2 allocates a
   descriptor buffer of its own is **UNKNOWN**.

---

## 9. The `00-tldr.md` change applied alongside this note

`AGENTS.md` requires the TLDR to define *"lazy vs. eager vs. compiled, graphs vs. recipes"*. It
defined **none** of them, and it used the word **eager** in a *second, unrelated* sense (placement).
The same branch that adds this note applies the following to [`00-tldr.md`](../00-tldr.md); this
section is the change log, so a reader of the TLDR can find the reasoning.

**1. Four rows inserted into the Terms table**, immediately after the `node` row:

- `eager placement` — an op the partitioner will not cluster; runs outside every recipe. **Not** the
  same thing as eager *mode*.
- `lazy mode` — `PT_HPU_LAZY_MODE=1`, the legacy execution model: ops accumulate into an implicit
  graph closed at a mark-step. **Deprecated; not a target.**
- `eager mode` — `PT_HPU_LAZY_MODE=0`, ops execute as issued, no recipe. A reference point, never an
  option (measured ~400×, ch 6).
- `compiled mode` — `torch.compile(fn, backend="hpu_backend")`, **the only supported target**. Modern
  vLLM Gaudi is `torch.compile`-only.

**2. One paragraph inserted directly under the Terms table**, next to the existing
"An `eager`-placed node runs **outside** every recipe…" note, disambiguating the overloaded word and
recording that **lazy mode is deprecated** (modern vLLM Gaudi is `torch.compile`-only; neither eager
nor lazy is ever a performance proposal — both are reference points). It carries one *source read*:
the shipped bridge disables autoload *only* when `PT_HPU_LAZY_MODE=1`, i.e. the default is non-lazy,
so the `bridge_config.py` comment claiming "the default mode is lazy" is **stale** — an instance of the
"stale claim to watch for" pattern the TLDR already documents.

**3. One line added under the five facts:** *"A `torch.compile` boundary is not a recipe boundary."*
The backend is `aot_autograd(partition_fn=hpu_partition)`: the region is partitioned first, and each
maximal contiguous clusterable run becomes its own recipe. This is what makes fact 5's "verify
submissions, not regions" actionable, and it is the same statement §1 of this note relies on (a recipe is a *run*, not a region —
so the thing that gets downloaded is a run), and §7 measures it: one decorated function,
two graphs, **two recipes**.

**Size check.** `00-tldr.md` was 23.6 KB / ~3.8 k words before this change; the three insertions add
~120 words, far inside the 10 000-token cap `AGENTS.md` sets.

**What this note does *not* change:** `01-basics.md`, `04-execution-model.md` and `05-profiling.md`.
The residency model belongs in the execution-model chapter and the budget in the profiling/iteration
chapters, but that is a separate pass.
