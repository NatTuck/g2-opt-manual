# 8. How PyTorch code compiles and runs

**Status: draft.** Tags follow the manual's convention — **MEASURED** / **source read** / **ASSUMED** /
**INFERRED** / **UNKNOWN** ([README](README.md#how-to-read-this-manual)). Every number below is
restated from the named artifact beside it, not from memory, and each carries the conditions it is
valid under. The measured material is on the runtime this project executes on: Synapse `1.24.2-481`
(build `79e284b`), driver `1.24.2-52a555d`, torch `2.13.0a0+git2469d18`, `habana-torch-plugin`
`1.24.2`. The source reads are the bridge source tree at **1.24.2** — the version the running process
loads; its `passes.py` is bit-identical to the installed `python_packages/habana_frameworks` copy — and,
for Synapse internals, an old **1.16** snapshot (base commit `6de1f66`), which is flagged wherever it is
the only evidence. Bridge paths are relative to that source root; Synapse paths to the 1.16 extraction
root.

**What this chapter is.** The layer *above* the device: what `@torch.compile` actually produces, how
that reaches Synapse, what a recipe is, what a launch costs, and what any of it costs in memory. It
answers the three questions that decide whether a change to your Python is a change to one program or
to four hundred:

1. **What does `@torch.compile` produce, and how many of each thing?** Three counts are routinely
   conflated — graphs, recipes, submissions — and they are not the same number. (§8.1–§8.3)
2. **What is a recipe, and what does it cost to run one?** A recipe is a host-side program that holds
   no device memory; every launch re-downloads it into a rotating ring. (§8.4–§8.5)
3. **What does the graph count cost in memory, and which cache are you actually sizing?** The answer
   is host memory and disk, not HBM — and the one cache everyone reaches for does not exist on this
   card. (§8.6–§8.7)

It closes with the specialisation rule restated at the recipe level (§8.8), a table of **which layer
owns each constraint** (§8.9), the lazy/eager/compiled distinction and its stale vendor comment
(§8.10), how to see all of this yourself (§8.11), and its own UNKNOWN register (§8.12).

> **Why this is chapter 8 and not part of chapter 4.** Chapter 4 is *the TPC execution model* — the
> layer **under** the compiler, denominated in VLIW bundles. This material is the layer **above** it:
> dynamo, AOT Autograd, the bridge's passes, Synapse's recipes. It belongs next to chapters 1–2, which
> use the vocabulary, but the manual appends and does not renumber
> ([chapter 4's own preface](04-execution-model.md)), so it is chapter 8 and cross-references chapters
> 1, 2, 4 and 6 rather than restating them.

**A note on what is new here.** Chapters 1–3 use the words *region*, *recipe* and *submission* as
though their relationship were obvious. It is not: §8.2 measures one decorated function becoming two
recipes, and §8.6 measures 24 distinct recipes costing **0.0 MiB** of device memory. Both facts are
load-bearing for how you write code, and neither was in the manual before.

---

## 8.1 The chain in one page

One `@torch.compile` call walks this chain. Each stage produces something with a different lifetime,
and only one of them is device memory.

| # | stage | produces | counted as | evidence |
|---|---|---|---|---|
| 1 | **dynamo** traces the Python frame | one FX graph per frame; a graph break splits a frame | **graphs** | §8.2 (MEASURED) |
| 2 | **`hpu_backend`** hands the graph to AOT Autograd | a joint fwd/bwd graph, then a partitioned forward | — | `backends.py:52-79` (*source read*) |
| 3 | **bridge FX passes** | a graph whose every node carries `meta["placement"]` | — | `passes.py:1596` (*source read*) |
| 4 | **partitioner** groups `hpu_cluster` nodes | one fused submodule per maximal contiguous run | **recipes** | `partitioner.py:27-38`, `cluster_compiler.py:132` (*source read*) |
| 5 | **`get_callable_recipe`** compiles each submodule | one Synapse recipe (host-side program) | **recipes** | `recipe_compiler.py:361-372` (*source read*) |
| 6 | **each execution** | one `synLaunch` that re-downloads the program | **submissions** | §8.5 (MEASURED + *source read*) |

**The three counts, and why they are not each other:**

- **graphs** (dynamo) ≥ **recipes** (maximal clusterable runs) ≥ **submissions** (one launch per
  recipe per execution).
- A graph break **increases graphs**; it does not by itself increase recipes-per-graph — it ends one
  region and starts another, each of which still contributes its own runs (§8.2, case C).
- Ops **do not** multiply recipes. Three clusterable ops in one body are one recipe (§8.2, cases B and
  D).
- **`torch.compile`'s boundary is not a recipe boundary.** The backend is
  `aot_autograd(partition_fn=hpu_partition)` (`backends.py:72-79`): the region is partitioned *first*,
  and each maximal contiguous clusterable run becomes its own recipe.

The measured table that establishes *which* constructs split a run — placement, not op identity — is
[§2.4](02-regions.md), 24/24 arms against the live FX graph. **This chapter does not re-argue it.** What
this chapter adds is the machinery that produces those boundaries, and what each boundary costs.

---

## 8.2 What `@torch.compile` actually produces: frames, graphs, breaks

**A decorated function does not create one graph.** Dynamo traces *frames*, and a graph break splits
one frame into several graphs. Measured with a cold recipe cache per case, so the `.recipe` count is
unambiguous (**MEASURED**, [notes §7](notes/recipe-residency-and-memory-budget.md)):

| case | body | dynamo graphs | breaks | ops | **recipes** |
|---|---|---|---|---|---|
| A | `t + 1` | 1 | 0 | 1 | **1** |
| B | `(t + 1) * 2 - 3` | 1 | 0 | 3 | **1** |
| C | `y = t+1; s = y.sum().item(); return y + s` | **2** | **1** | 3 | **2** |
| D | `y = t+1; return y[:t.shape[0]//2] * 2` | 1 | 0 | 3 | **1** |

The `ops` column counts **every** op in the function **across all its graphs** — case C is `add`,
`sum`, `add`, and the `.item()` that causes the break is a transfer, not an op. That is why B, C and D
are all three and only C needs two recipes: the column exists to show that op count is not what moves
the recipe count.

Three readings, each of which changes how you write code:

1. **Case C is the minimal demonstration that a decorated function is not a program.** One function,
   two graphs, **two recipes** — because `y.sum().item()` is a host read, which is a graph break. A
   region that breaks is *two regions* for every purpose the rest of this manual cares about: two
   compilations, two sets of launches, two cache entries.
2. **Ops do not multiply recipes.** B and D are three ops each and still one recipe. A recipe is one
   per maximal contiguous run of *clusterable* nodes, so the count tracks **boundaries**, not op count
   ([§2.4](02-regions.md)).
3. **The recipe filename does *not* carry the environment, and that is a hazard.** The generated names
   are `<graph hash>_<libs_env_hash>_syn1.24.2.79e284b.recipe`. The **trailing** field is the *driver*
   version string, so a Synapse or driver upgrade does change every filename. The **middle** field looks
   like an environment hash and is a **constant** — `_a16fb581eee_` in every filename this project has
   ever produced — because both of its inputs are dead code (§8.7.3). **Rebuilding a TPC kernel or
   changing a compiler flag does not change the key.** What that costs you, and what to do about it, is
   §8.7.3; the file's contents are hashed to zero at `cache_version.cpp:107,123,138`. (**MEASURED**,
   notes §8.1.)

**What a graph break costs is not this chapter's subject.** It costs in two currencies — overlap and
host launch traffic — and [§2.5.1](02-regions.md) prices both. Here the point is narrower: the break
also changes the *unit of compilation*, and everything downstream is counted per unit.

---

## 8.3 The backend is AOT Autograd with a partitioner

`hpu_backend` is registered as a dynamo backend and is a thin wrapper around `aot_autograd`
(**source read**, `habana_frameworks/torch/dynamo/compile_backend/backends.py:52-79`):

```python
@register_backend
def hpu_backend(graph_module, example_inputs, **kwargs):          # :52-53
    ...
    inference_compiler = partial(hpu_inference_compiler, dyn_graph_module=graph_module)   # :62
    ...
    return aot_autograd(
        fw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_fw),
        bw_compiler=hpu_backend_config.patch(options)(hpu_training_compiler_bw),
        inference_compiler=hpu_backend_config.patch(options)(inference_compiler),
        decompositions=get_hpu_decompositions(),
        keep_inference_input_mutations=hpu_backend_config.keep_input_mutations,
        partition_fn=hpu_partition,                               # :78
    )(graph_module, example_inputs)                               # :79
```

Two things follow, and they are the reason this chapter exists:

- **The partitioner is a first-class stage, not a detail.** `partition_fn=hpu_partition` (`:78`) is
  the hook that decides how many recipes a region becomes.
- **`hpu_backend_config.inference is False`** swaps the inference compiler for the training
  backward compiler (`:66-71`) — i.e. training and inference are the *same* pipeline with a different
  compiler slot.

### 8.3.1 The partitioner, and what "maximal contiguous run" means mechanically

`hpu_partition` (**source read**, `partition_fn.py:253-291`) runs joint-graph passes and then the
default partitioner:

```python
def hpu_partition(joint_module, _joint_inputs, *, num_fwd_outputs, static_lifetime_input_indices=None):  # :253
    ...
    fw_module, bw_module = default_partition(joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs)  # :282
    ...
    except AssertionError:
        return min_cut_rematerialization_partition(...)           # :291
```

The *recipe* partitioner — the one that produces clusterable runs — is a capability-based partitioner
over the forward module, and its capability test is the placement metadata (**source read**,
`partitioner.py:27-38`):

```python
class HabanaClusterOperatorSupport(OperatorSupport):              # :27
    def is_node_supported(self, submodules, node):
        return node.meta["placement"] == "hpu_cluster" and "partition_assigned" not in node.meta   # :29

class HabanaPartitioner(CapabilityBasedPartitioner):              # :32
    def __init__(self, graph_module, sup_op=HabanaClusterOperatorSupport):
        super().__init__(graph_module, sup_op(), allows_single_node_partition=True)   # :34-38
```

**A node is clusterable exactly when its `placement` is `hpu_cluster`.** That is the whole rule; what
sets placement is [§2.4](02-regions.md)'s subject and the validator's, not this chapter's. The
placement field itself is written by the bridge's passes (**source read**, `passes.py:1592-1596`), and
the pass is strict about it:

```python
if "placement" in node.meta:
    logger.debug(f"Node {node} ... has had it's placement already set")
    if not node.meta["placement"] == placement:
        raise AssertionError("Placement mismatch")                # :1595
node.meta["placement"] = placement                                # :1596
```

`allows_single_node_partition=True` (`:37`) is why a **lone** clusterable node between two eager nodes
still becomes a recipe of its own — which is what makes the recipe count behave the way
[§2.4](02-regions.md)'s table reports.

### 8.3.2 From fused submodule to compiled recipe

After partitioning, each fused submodule is replaced by a compiled callable (**source read**,
`cluster_compiler.py:132-145`):

```python
syngraph_module = get_callable_recipe(jit_ir, submod, self.ctx.graph_name, ...)   # :132
self.ctx.graph_module.delete_submodule(target)                                    # :142
self.ctx.graph_module.add_submodule(target, syngraph_module)                      # :143
self.subgraph_cnt += 1                                                            # :145
```

and `get_callable_recipe`'s own docstring states the fallback honestly (**source read**,
`recipe_compiler.py:361-372`):

> *Calls backend to create compiled recipe or just returns unchanged module to run it eagerly depending
> on config.*

That docstring is the whole eager/compiled story in one line (§8.10): eager is not a mode you choose
for speed, it is the path taken when a node is not compiled.

The bridge reports the count it produced — `Number of subgraphs created: %s` (`cluster_compiler.py:158`)
— which is a directly observable proxy for **recipes**, and the cheapest instrument in this chapter
(§8.11).

### 8.3.3 The four compiler phases, in order

`hpu_compiler_inner` (**source read**, `compilers.py:132-181`) is the per-graph entry point, and its
order is the pipeline:

```python
optimize_pre_placement(graph_module, ...)          # :168
optimize_pre_partitioner(graph_module, ...)        # :171
partition_module(graph_module, ...)                # :174   <- the placement-aware split
optimize_post_partitioner(graph_module, ...)       # :177
```

with `hpu_inference_compiler` (`:205-214`) a passthrough that selects the freezing compiler when
`torch._inductor.config.freezing` is set, and otherwise calls `hpu_compiler_inner(..., is_training=False,
is_backward=False)`. The four phase names are the useful part: **placement is decided before the
partitioner runs** (`optimize_pre_placement` → `optimize_pre_partitioner` → `partition_module`), so a
pass that changes placement changes the recipe count, and a pass that runs after the partitioner
cannot.

---

## 8.4 What a recipe is, and where it lives

A compiled recipe is a **host-side program**. Everything the device will later walk is allocated as
host pages and *mapped* into the device address space, not allocated in HBM (**source read**, 1.16
`src/runtime/common/recipe/recipe_allocator.cpp:15-30` and
`src/runtime/common/memory_utils/memory_allocator_utils.cpp:12-40`):

```cpp
const bool mappedAlloc = !m_ignoreDeviceMapping && shouldBeMappedToDevice;
if (mappedAlloc) buf = (char*)MemoryAllocatorUtils::alloc_memory_to_be_mapped_to_device(sizeInBytes);
else             buf = new char[sizeInBytes];
```

and the allocator's comment says why the pages are pinned down:

> *in order to prevent physical pages mapped to device from being swapped, thus causing MMU mapping
> problems* — followed by `madvise(MADV_DONTFORK)`.

The buffers allocated this way are exactly the ones the device walks: the execution, patching and
dynamic blob buffers, both ECB command buffers, and the program-data blob buffer. Four consequences:

1. **`synGraphCompile` allocates no device memory.** A compiled, never-launched recipe holds **zero
   HBM**. (**MEASURED**: 8 distinct recipes, each compiled *and launched*, moved free HBM by
   **0.0 MiB** — [notes §2b](notes/recipe-residency-and-memory-budget.md).)
2. **`synRecipeDestroy` frees host memory only** (**source read**, 1.16
   `recipe_handle_impl.cpp:18-24`: it frees the `RecipeAllocator` and deletes the handle). There is no
   device free on that path because there is no device allocation to free.
3. **The workspace is not per-recipe.** `synLaunch` takes `pWorkspace` from the caller
   (`include/synapse_api.h:612-617`, *source read*, 1.16), and the bridge keeps **one device-wide**
   workspace buffer sized by the largest recipe. So the workspace is a *max*, not a sum.
4. **The attribution tools exist, and they are the ones to use.** `synRecipeGetAttribute` reports
   `WORKSPACE_SIZE`, `DEVICE_MEMORY_SIZE`, `HOST_MEM_SIZE`, `CONST_SECTIONS_SIZE` and
   `PERSISTENT_TENSORS_SIZE` per recipe; `synRecipeSectionGetProp` reports a **const** section's size
   and its *host* pointer, and nothing about device addresses. Do not guess a per-recipe device cost —
   query it. (§8.6 uses the same call set to build the budget.)

**The mental model to keep, and the two wrong ones to drop.** A recipe is a program image on the host
with a device-side execution window that is **shared and rotating**. "Compiled recipe ⇒ HBM occupied"
is wrong. "There is a recipe cache in HBM holding my compiled graphs" is also wrong on this card
(§8.7).

---

## 8.5 What every submission costs

**Every launch downloads the program again.** `RecipeLauncher::launch()` calls `prepareForDownload()`
then `downloadToDev()` unconditionally — there is no "already resident" branch — and `downloadToDev`
runs `memcpyToMapped` → patch → **`pdmaDownload`** (mapped host → HBM, by CpDma commands) → fence wait →
`scalEnqueue` (**source read**, 1.16 `src/runtime/scal/common/recipe_launcher/recipe_launcher.cpp`,
quoted in [notes §1](notes/recipe-residency-and-memory-budget.md); that file is one of the 80 the patched
tree modifies, so no clean line number can be quoted for it).

**The destination is a rotating ring, not a cache.** `ArcHbmMemMgr` and `HbmGlblMemMgr` derive from
`NBuffAllocator`, which splits a fixed window into `NUM_BUFF` chunks and hands out the next one,
returning *"an offset and the long-sync-object to wait for"*; `prepareForDownload` then waits on that
long-SO — *wait until this slot is free on the device*. The 1.16 source has `NUM_BUFF = 16`
(`src/runtime/scal/common/recipe_launcher/nbuff_allocator.hpp:28`) and chunk size `size / NUM_BUFF`
(`hbm_global_mem_mgr.cpp:39-43`); on 1.24.2 the measured count is **64** (§8.6).

The vendor states the design and its limitation in the header itself (**source read**, 1.16
`arc_hbm_mem_mgr.hpp:9,14`, repeated at `hbm_global_mem_mgr.hpp:8,13` — both headers are unmodified in
the patched tree, so these are clean 1.16 lines):

> This class is used to allocate memory of the Arc-HBM for running a recipe. It is currently
> implemented in a very simple way, **assuming no caching**. It splits the given memory into chunks
> (downloadedRecipes) and each time assigns a new chunk. … **In the future, we can consider adding
> caching here to avoid the pdma**

So the one-line summary for the manual:

> **A graph costs one program download per launch, not one HBM allocation per compile.**

**What that download costs in time.** The *first* call is size-independent — **103.6 ms at 1 K and at
1 M elements** — and the steady-state call is **~0.5–0.7 ms** (**MEASURED**,
[notes §5](notes/recipe-residency-and-memory-budget.md)). The per-launch cost is therefore dominated by
the fixed path (ring walk, long-SO wait, enqueue), not by program bytes, until the program gets large.
That is why graph breaks hurt beyond the CPU side: two one-node graphs pay two downloads, two ring
waits and two enqueues where one two-node graph pays one.

**When the ring is full, the launch does not spill — it throws.** A single recipe's program must fit
in one chunk; past the window the launch fails outright (**MEASURED**, notes §2).

---

## 8.6 The memory budget

This is the section to read before sizing a KV cache, and its headline is a negative result.

### 8.6.1 The fixed reservation, before your first tensor

**All MEASURED** on 1.24.2 / Gaudi2, one card ([notes §2b](notes/recipe-residency-and-memory-budget.md);
the driver and idle rows re-confirmed for this chapter with `hl-smi` on 2026-11-13). **Three "total HBM"
figures are in circulation and they are three views of one card** — quote the one you measured, with its
instrument:

| view | figure | instrument |
|---|---|---|
| **driver** | **98304 MiB = 96.00 GiB** | `hl-smi -Q memory.total -f csv,nounits,noheader` |
| **Synapse** — **the budget base** | **97376 MiB = 95.09 GiB** | `synDeviceGetMemoryInfo(dev, &free, &total)`, take `total` |
| **torch** | **96895.2 MiB = 94.62 GiB** | `torch.hpu.mem_get_info()[1]` — **bytes**, divide by 2²⁰ |
| **driver `memory.used` at idle** | **768 MiB** | `hl-smi -Q memory.used …`, no process holding the card |

The driver's total is **928 MiB** larger than Synapse's (firmware and driver reservation, which Synapse
never sees and you can never allocate). Synapse's is **481 MiB** larger than torch's (the program
windows, carved out at device acquire). The **768 MiB** is a *usage* figure, not a fourth total — it is
the idle baseline this manual already publishes ([§1.5](01-basics.md), `01-basics.md:394`: *"an idle
baseline of ~768 MiB. **MEASURED**"*) — and it is **not** something to subtract on top of the 481 MiB.

| item | size | when | scales with recipe count? |
|---|---|---|---|
| card HBM, Synapse view — **the budget base** | **97376 MiB** = 95.09 GiB | — | no |
| ↳ global program window | **192 MiB** = 3 × `HBM_GLOBAL_MEM_SIZE_MEGAS`(64) | device acquire, before any tensor | **no** |
| ↳ arc program window | **127.5 MiB** (`0x7F80000`) | device acquire | **no** |
| ↳ streams, monitors, TDR, profiler | **~161 MiB** (remainder) | device acquire | no |
| **total reserved before your first tensor** | **~481 MiB (0.47 GiB)** — which is exactly why torch reports 96895.2 | | **no** |
| your workspace | `synWorkspaceGetSize`, **one** device-wide buffer in the bridge | first launch | no (sized by the max, not the count) |
| your tensors (weights, KV cache, activations) | the rest | you | you |

The window figures come from the runtime itself, printed once at device acquire with
`LOG_LEVEL_SYN_PROG_DWNLD=0` (**MEASURED**):

```
ArcHbmMemMgr: buffer addrCore given/actual 52000000/52000000
               addrDev given/actual 1001600152000000/1001600152000000
               size given/actual 7f80000/7f80000   chunkSize 1f8000
HbmGlblMemMgr: buffer addr given/actual 100160015a000000/100160015a000000
               size given/actual c000000/c000000   chunkSize 300000
```

`0x7F80000` = 127.5 MiB with a `0x1F8000` (≈1.97 MiB) chunk; `0xC000000` = 192 MiB with a 3 MiB chunk.

**The 64-chunk count holds for the global window only, and it is INFERRED.** One method at three
settings of one key — 192/3, 768/12 and 3072/48 MiB — gives 64 every time, so the *global* window has
64 chunks and 1.16's `NUM_BUFF = 16` was raised. The **arc** window does not divide: 64 × `0x1F8000`
= `0x7E00000` = 126 MiB against a `0x7F80000` window, so **1.5 MiB is never handed out**. The 1.16
source has the two terms that could explain it — `round_down(chunkSize, ALIGN)` with `ALIGN = 128`
(`arc_hbm_mem_mgr.cpp:44-48`, `arc_hbm_mem_mgr.hpp:36`), and `m_size = size - (m_addr - addr)`
(`hbm_global_mem_mgr.cpp:24-27`), the size after the base address is rounded up — but which one applies
in 1.24.2 is **UNKNOWN** (the log prints the *given* size in both fields). The budget uses the 127.5 MiB
**window**, not the chunk, so nothing below depends on it. It is UNKNOWN #2.

**The practical rule:**

> Budget = **97376 MiB − 481 MiB** − workspace − weights.
> **Do not subtract a per-recipe term.** If HBM is filling up as the graph count grows, the recipes are
> not what is filling it — look at persistent/const sections, the workspace being sized by the largest
> recipe, or tensors retained by the host-side cache.

### 8.6.2 What *does* grow, measured

24 distinct recipes, each compiled **and launched** (**MEASURED**,
[notes §2c](notes/recipe-residency-and-memory-budget.md)):

| resource | 0 recipes | 24 recipes | growth | where the bound comes from |
|---|---|---|---|---|
| device memory (free) | 96889.7 MiB | 96889.7 MiB | **+0.0 MiB** | the 64-chunk ring; constant per stream |
| host RSS | 2104.6 MiB | 2215.2 MiB | **+110.6 MiB** (~4.6 MiB/recipe) | one `recipe_t` + its mmap'd program buffers **per live recipe** |
| disk (`.recipe` files) | 0 | 0.4 MiB / 9 files | grows with distinct recipes | `PT_HPU_RECIPE_CACHE_CONFIG`'s `max_dir_size_mb` (default 1024) |

So the inversion, which is the sentence to carry into a design review:

> **A million recipes costs the same HBM as one recipe, and is impossible in host memory.** The device
> holds a fixed window; the *host* holds one program image per live recipe, in `mmap`ed pages that are
> mapped into the device address space and marked `MADV_DONTFORK` — they cannot be swapped out.

Two corollaries that are worth more than the table:

- **A "recipe count" HBM problem is almost certainly a host-memory problem wearing HBM clothes.** When
  the bridge's LRU evicts under host pressure (§8.7), the evicted recipes *recompile* on their next
  launch — which presents as a rising `compileGraph` count and a slower steady state, easily misread as
  device pressure. [§2.5.3](02-regions.md) lists exactly this signature ("count rises and keeps rising
  in the steady state → cache-limit thrash").
- **Executing a million recipes is compile-bound, not memory-bound.** First execution of a distinct
  recipe measured **~50 ms dynamo + ~52 ms bridge/Synapse ≈ 100 ms**; a million of them is **~28 hours**
  of compilation, and each launch still pays a program download. (**MEASURED**, notes §2c.)

### 8.6.3 The one knob, and why it is not linear

`HBM_GLOBAL_MEM_SIZE_MEGAS` scales the global window — but it is **private** and **per-stream**
(**MEASURED**, notes §2):

| env | logged global window | device total reported | delta vs default |
|---|---|---|---|
| (unset, default 64) | `0xC000000` = 192 MiB | 96895.2 MiB | — |
| `HBM_GLOBAL_MEM_SIZE_MEGAS=32` alone | 192 MiB — **silently ignored** | 96895.2 MiB | 0 |
| `ENABLE_EXPERIMENTAL_FLAGS=true …=256` | `0x30000000` = 768 MiB | 96319.2 MiB | **−576 MiB** = the window delta, exactly |
| `ENABLE_EXPERIMENTAL_FLAGS=true …=1024` | `0xC0000000` = 3072 MiB | 91359.2 MiB | **−5536 MiB** ≠ window delta (2880) |

Read it as three separate lessons:

1. **Private keys need `ENABLE_EXPERIMENTAL_FLAGS=true`**, and the API cannot set them at all —
   `synConfigurationSet` passes `enableExperimental=false`. Environment only, before start.
2. **The reservation is 3 × the key**, because it is per compute stream and there are three
   (`MaxNumOfComputeStreams = 3`, *source read*, 1.16 `scal_dev.hpp:58`; the pools are allocated
   `Size × MaxNumOfComputeStreams`, `device_scal.cpp:242-247`).
3. **A size knob here is not linear.** At 1024 the device lost 5536 MiB rather than 2880: the larger
   allocation crossed a 4 GB address boundary and triggered the allocator's *"not aligned, moving to
   next 4GB alignment"* workaround (**source read**, 1.16 `device_scal.cpp:266-290`), which throws away
   the difference. Budget by differencing the **total**, not by arithmetic on the key.

**And it is a size, never a policy.** A bigger window means a bigger maximum single-recipe program and
fewer ring waits; it does **not** make anything resident.

**Two instrument traps.** `synDeviceGetMemoryInfo()` on this build returns **`free = 0`** (status
`synSuccess`, total correct). Use `torch.hpu.mem_get_info()`. And that call returns **bytes, not MiB** —
`101601996800` is 96895.2 MiB — so reading it as MiB is a **2²⁰×** error. (Re-measured for this chapter:
`free = 101596229504 B`, `total = 101601996800 B` = 96895.21 MiB.) The reported `total` *falls* as you
raise the window — Synapse carves it out of the pool — so measure by differencing `total`.

---

## 8.7 The four caches, and which one people confuse

Four different things are called "the cache". Only one is on the device, and it is not on this device.

| cache | what it holds | where it lives | how it is bounded | on Gaudi 2? |
|---|---|---|---|---|
| **dynamo's code cache** | traced frames and guards | host, per process | dynamo's own limits | yes |
| **bridge disk recipe cache** | serialized `.recipe` files | **disk** | `PT_HPU_RECIPE_CACHE_CONFIG`'s `max_dir_size_mb` (default **1024**) — and its key **cannot see a kernel edit** (§8.7.3) | yes |
| **bridge host LRU** | live `synRecipeHandle`s | **host memory** | `HABANA_PGM_LRU_MAX` (defaults **100 000** eager / **30 000** lazy); evicts on a host threshold | yes |
| **device recipe cache** | recipe programs in HBM, LRU blocks | **device HBM** | `RECIPE_CACHE_SIZE` (default **500 MiB**), block **128 KiB** | **no — Gaudi1 only** |

Two names that are *not* the same cache and are routinely conflated: `GCFG_RECIPE_CACHE_*` is
Synapse's **on-device** cache (Gaudi1), and `PT_HPU_RECIPE_CACHE_CONFIG` is the bridge's **disk** cache.
Different layer, different medium, different owner.

### 8.7.1 The one that does not exist here

There **is** a genuine device-HBM recipe cache in the tree — on the **Gaudi1 / qman** path
(**source read**, 1.16): `GCFG_RECIPE_CACHE_SIZE` (default **500 MiB** — `500 * 1024` KB, private) and
`GCFG_RECIPE_CACHE_BLOCK_SIZE` (**128 KiB**, private) size one device buffer, wrapped in a
`RecipeCacheManager` with per-recipe refcounts and an LRU invalidation policy
(`recipe_cache_manager.cpp:40` calls it an *"on-device cache that stores recently used static part of
recipes on pre allocated memory"*; the allocation is
`allocateMemory(cache_size + MANDATORY_KERNEL_ALIGNMENT, synMemFlags::synMemDevice, …)` at
`device_recipe_addresses_generator.cpp:139-160`, refcounts at `:219-260`; defaults at
`habana_global_conf.cpp:1121-1133`). **Device HBM, not disk.**

The SCAL/Gaudi2 path (`DeviceScal`) references it **nowhere** — a grep over
`synapse/src/runtime/scal` for `RecipeCacheManager|RECIPE_CACHE` is empty — and setting
`ENABLE_EXPERIMENTAL_FLAGS=true RECIPE_CACHE_SIZE=262144` on this Gaudi2 box produces **no** cache log
line and changes neither program window (**MEASURED**, notes §4).

> **If anyone writes "set `RECIPE_CACHE_SIZE` to cache recipes in HBM", that is a Gaudi1 statement.**
> On Gaudi 2 there is no cache to size; what exists is the rotating ring of §8.5.

### 8.7.2 The one that makes you think you have an HBM leak

The bridge's host LRU is the cache that actually evicts in a long run, and it evicts on **host** memory,
calling `synRecipeDestroy` on the GC thread. Because eviction destroys the recipe, the next launch of
that graph **recompiles it**. So the symptom of host pressure is a rising compile count and a slower
steady state — which reads as device pressure and is not. §8.6.2's corollary is the diagnosis; this is
the mechanism.

The threshold is a **two-level story, and a grep misleads**:

- The gcfg default for `PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT` is **0**, and 0 **disables the check
  entirely** (`scripts/env_flags.yaml:428-430`; `pytorch_helpers/habana_helpers/misc_utils.cpp:32-34`
  tests `!= 0U`).
- The familiar **90 %** comes from the *bridge*, which substitutes its own default the first time a
  dynamic bucket is created: `RecipeCacheLRU::default_host_memory_threshold = 90`
  (`backend/kernel/hpu_recipe_cache.h:130`), written into the flag if it is unset.

So "default 90 %" is true of a process that compiled something and false of the gcfg. Eviction runs in
**512 MiB** chunks (`512LL * 1024 * 1024`, `backend/kernel/hpu_habana_launch_op_pt.cpp:4211` — the code
comment says "512MB", the constant is MiB), and it is only *attempted* when a graph is built or a recipe
inserted, never on a timer.

### 8.7.3 The disk cache's key cannot see your build

The disk cache's **filename is its key** — `recipe_cache_.store(hashCode + cache_id_suffix_, …)` at
`backend/kernel/hpu_recipe_cache.cpp:317` — and the suffix is built at `:294-308`:

```
<RecipeArgumentSpec::hashCode()>_<CacheVersion::libs_env_hash()>_syn<synDriverGetVersion()>
```

The leading field is the graph (ops, shapes, attributes). The trailing field is the **driver** version
string, so a Synapse or driver upgrade changes every filename. The middle field is *meant* to change
when the **kernel library** or a compiler flag changes, so that a rebuild misses the cache. **It does
not.** Both halves of it are dead code, for two independent reasons.

**The file half.** `hash64_file_content()`
(`pytorch_helpers/habana_helpers/habana_serialization/cache_version.cpp:100-139`) declares
`size_t hashRes{0}` at `:107` and then a **second, shadowing** `size_t hashRes{0}` inside the `mmap`
branch at `:123`. The loop at `:126-127` fills the **inner** one; `:138` returns the **outer** one. It
therefore returns **0 for every file, whatever its size or contents** — the `mmap` and the per-byte loop
are wasted work. (**source read**: `cache_version.cpp:107,123,126-127,138`.)

**The env half.** `check_env_fo_hashing()` (`cache_version.cpp:141-172`) decides with `env_var.find(v)`
(`:170`) whether an environment entry is on its list. The needles are **space-padded to a column** so
they line up in the source — `"ENABLE_RAGGED_SOFTMAX_OPT             "` (`:150`) — and the haystack is a
`"NAME=VALUE"` string taken from `environ` (`:208-215`). `"NAME=VALUE"` contains no run of spaces, so
**no needle ever matches**: no environment variable has ever been hashed. (**source read**:
`cache_version.cpp:141-172,150,165,170,208-215`.)

So `libs_env_hash()` (`cache_version.cpp:174-222`) is `at::hash_combine` applied N times to 0. It varies
with the *number* of files it was asked to hash and with nothing else. (That number is 2 or 3 — the
habana device path, the Synapse library, and one `GC_KERNEL_PATH` entry. The multi-path branch at
`:184-198` splits on `:`, but a colon-list `GC_KERNEL_PATH` is **rejected at device init** — `FATAL
ERROR … Path is not a regular file`, exit 134 — so that branch is unreachable in practice.)

**Measured: three arms, one filename.** One graph (`x * 2.0 + 1.0`, `fullgraph=True`, `dynamic=False`), a
**fresh cache directory per arm**, `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,true,1024,false` — `true` is
`delete_on_init`, which is harmless *here* because each arm starts empty anyway (it is not harmless for
a cross-run test; §8.7.4). All three arms produced the **identical** file:

| arm | `GC_KERNEL_PATH` | environment | recipe file |
|---|---|---|---|
| 1 | the stock `libtpc_kernels.so` | — | `4757981558864380105_a16fb581eee_syn1.24.2.79e284b.recipe` |
| 2 | a copy of the same library with **one byte appended** | — | **identical** |
| 3 | the stock library | `DISABLE_TENSORS_PINNING=true` — a name on the hashed list (`:165`) | **identical** |

`_a16fb581eee_` is also the field in every filename this project has ever produced, including every
filename in §8.8's measurements. A constant is exactly what a hash of "0, 0, 0" looks like.
**MEASURED.**

**The consequence is the opposite of the obvious one.** The cache cannot see a kernel edit or a
configuration change, so:

1. **Editing a TPC kernel and rebuilding will be served the stale recipe wherever the cache hits** —
   and it does hit (§8.8.2 measures it). The name is unchanged, the file is there, the graph is the
   same graph. When you iterate on a kernel, **clear the cache directory** — or set the config's second
   field, `delete_on_init`, to `true` — as part of the rebuild. The key and the hit are MEASURED; the
   byte substitution after a rebuild is INFERRED (third tag, below).
2. **A cache directory shared across two configurations silently mixes recipes** built under different
   compiler flags: two builds of one kernel library, one directory, one key.
3. **Maintain your own key: one cache directory per *(build, configuration)* pair.** Nothing in the
   filename provides that separation, so it has to come from you.
4. **`PT_RECIPE_CACHE_IGNORE_VERSION=1`** drops the `_<hash>_syn<version>` suffix altogether
   (`hpu_recipe_cache.cpp:299-301`). It exists, it makes the sharing worse, and it is a reminder that
   this suffix is the cache's only version guard.

**What does still invalidate:** `RecipeArgumentSpec::hashCode()` — the graph, i.e. ops, shapes and
attributes — and the driver version string. A **Synapse or driver upgrade** changes every filename; a
**kernel-library rebuild does not**. Two different things are called "the build", and only one of them
is in the key.

**Three claims, three tags — and the third is the one to run.**

- **The key is constant — MEASURED.** The three arms above, and `_a16fb581eee_` in every filename this
  project has produced.
- **The cache hits — MEASURED.** With `delete_on_init=false`, a second process at an unseen integer
  value created **0 new recipes** and took an 83.7 ms disk hit (§8.8.2);
  [`recipe-launch-notes.md`](notes/recipe-launch-notes.md) §4 records the same on a same-config reboot.
- **A rebuilt kernel is then served the old bytes — INFERRED.** The key is constant and the cache hits,
  but nobody has rebuilt a kernel and inspected which recipe came back. That is UNKNOWN #8, and it is
  one experiment away.

**The instrument that hid this was ours.** The experiment that first concluded "the disk cache does not
hit" ran `PT_HPU_RECIPE_CACHE_CONFIG=<dir>,true,1024,false` — and the second field is `delete_on_init`,
which wipes the directory at **every** process start (§8.7.4). **A cache wiped at init cannot hit**, so
that experiment could not have observed a hit whatever the truth was: it recompiled and rewrote the
`.recipe` on all four runs (sizes 3366 B, 3144 B) because it was told to. An instrument that overwrites
its own evidence. The same `true` is harmless in the three arms above only because each arm starts from
a fresh directory; the cross-process arm in §8.8.2 needs `false`.

### 8.7.4 Two more traps in the same configuration string

1. **The second field is `delete_recipe_cache_on_init`, and the third is MB — not entries**
   (`bridge_config.py:71-86`: `path, delete_on_init, max_size_MB, allow_NFS`; the gcfg default is
   `",false,1024,false"` at `scripts/env_flags.yaml:96-97`, so with no path there is no disk cache at
   all). With `true`, **every
   process wipes the directory at init**, so `.recipe` counts are destroyed as evidence and
   cross-process reuse cannot be measured at all. Per-arm counts taken with `true` are fine for
   *within-process* counting and invalid for anything cross-process. The fourth field, NFS, makes the
   second and third be ignored entirely.
2. **The recipe filename does not determine the bytes.** The same graph compiled into two fresh cache
   directories produced the same filename at the same size — 26,270 B — and **different md5s**
   (`1ca32f92…` vs `a76dd872…`) (**MEASURED**: `md5sum` over the two `.recipe` files; reproduce with two
   runs of the `ints.py` listing in §8.13). The name keys graph **structure**, not content. Sharing a
   directory was measured safe for the integer case — the 0-d operand is an input *slot* bound at launch
   (§8.8.2) — but "same filename" is not "same recipe", and §8.7.3's whole hazard is that the name is
   the only thing the cache checks.

---

## 8.8 Why a new shape or a new `int` is a new program — and where that stops

The specialisation rule is stated once in [§1.4](01-basics.md) (static shapes, specialised integers)
and its counting consequence in [§2.5.3](02-regions.md) (the specialisation count as a first-class
metric). At the recipe level there are two edges that are easy to get backwards, and the second is where
the manual's own summary has been wrong twice.

### 8.8.1 A guard is not a node

- **A Python `int` in traced code joins the guard set.** It can produce a new compiled graph, but it
  does **not** split one compilation into more recipes, because the partitioner never sees it
  ([§2.4](02-regions.md)'s warning, explicitly).
- So the two counts move for different reasons: **recipes within one graph** are set by placement
  boundaries; **graphs per region** are set by the guard set (shape, dtype, int, and the container
  structure of the inputs).
- The same function called at a new shape or with a new traced `int` is a **new frame** → new graphs →
  new recipes, and each new recipe is a new program image on the host and a new ring slot to wait for
  (§8.5).

### 8.8.2 There *is* an escape hatch, and it is a 0-d tensor

Pass the varying integer as a **0-d tensor** and the value stops being part of the program. Measured —
one compiled function, `f(t, k) = t + k`, `k` a 0-d `int64` tensor on the device, `n = 1…10`
(**MEASURED**, the integer probe in §8.13):

| arm | dynamo graphs | recipes |
|---|---|---|
| `k` a 0-d `int64` tensor, `n = 1…10` | **1** | **1** |

The graph signature is `forward(self, L_x_, L_k_: "i64[][]hpu:0")` and the body is `add = l_x_ + l_k_` —
**the value is an operand, not a literal**. The recipe is written once and never rewritten; the first
call is 142 ms and every later value costs 0.3–0.5 ms.

**It crosses a process boundary as data.** With `delete_on_init=false` and a warm cache directory
(**MEASURED**):

| process | call | recipes after |
|---|---|---|
| 1: `n = 1` | 151.3 ms | **1** |
| 2 (cold process): `n = 42`, a value never seen | **83.7 ms** | **1** — **0 new recipes** |

That 83.7 ms is a real disk hit at an unseen value: nothing was compiled, the recipe was deserialised.
A Python int at the same point does not behave this way (§8.8.3).

**What it costs.** The graph takes an `i64` operand and produces an `f32` result, so the recipe carries a
cast — on the device, not a host sync, and not a second graph. Both `shape ()` and `shape (1,)` are
accepted; there is no rank-≥1 requirement.

### 8.8.3 A Python `int` costs **two** recipes on the default path — or one per value if you force static

This is the part the manual's summary gets wrong twice, so both halves are measured (**MEASURED**,
the integer probe in §8.13; torch 2.13 / bridge 1.24.2):

| arm | values | recipes |
|---|---|---|
| `k` a Python int, default (`dynamic=None`) | 1, 2, 3 | **2** |
| `k` a Python int, default (`dynamic=None`) | 1…10 | **2** — flat |
| `k` a Python int, `dynamic=True` | 1, 2, 3 | **2** |
| `k` a Python int, **`dynamic=False`** | 1, 2, 3 | **3** — one per value |
| `k` a Python int, `specialize_int=True` | 1, 2, 3 | **3** — one per value |

**The mechanism, from the graph dump.** The **first** value is baked as a constant —
`def forward(self, L_x_)` with `add = l_x_ + 1`. The **second** distinct value fails the guard and
recompiles to a graph that takes the int as a **symbolic input**:

```
def forward(self, L_x_: "f32[1024][1]hpu:0", L_k_: "Sym(s16)")
    add: "f32[1024][1]hpu:0" = l_x_ + l_k_
```

That second graph then serves every later value, which is why the count is **flat at two** from the
second value onward. Within a process a Python int is therefore bounded at two recipes — **not** one per
value.

**But forcing static compilation restores one-per-value.** With `dynamic=False` — which is what this
manual tells you to use for a reproducible measurement — `assume_static_by_default` is set, every
distinct int is specialised, and three values give three recipes (`specialize_int=True` is the same arm).
**If you benchmark with `dynamic=False`, budget one program per distinct integer.**

**The unbounded case is a cold process.** The two-recipe bound is *per process*. Process 1 at `n=1`
produces one recipe; a process 2 that meets `n=42` produces a **new** one and pays a full 186 ms
compile, because a cold process re-bakes whatever value it meets first (measured, same cache directory
as §8.8.2's table). So *value-per-process* — a server that restarts, a harness that forks per case —
grows the cache without bound. That, and not the in-process count, is the scaling problem.

### 8.8.4 Structural ints cannot be escaped

Where the integer decides the **shape**, a tensor cannot help — the value *is* the graph. `x[:n] * 2`
with a Python int (**MEASURED**, the integer probe in §8.13):

| values | dynamo graphs | recipes |
|---|---|---|
| `n = 1, 2, 3` | 2 | **3** |
| `n = 1…7` | 2 | **4** |

Growth with distinct values is real, and it is **not** one-per-value — in this run `n = 4, 5, 6` reused
`n = 3`'s recipe and only `n = 7` added one. That is shape **bucketing** on the sliced extent, and the
bucketing rule is **UNKNOWN** (UNKNOWN #9). `x.reshape(n, -1)` behaves the same way. The practical
reading: for a *data* integer — an offset or a count that changes no shape — a 0-d tensor removes the
program count entirely; for a *structural* integer it does not, and the only lever left is to make the
value set small and explicit.

### 8.8.5 What the counts cost

The arithmetic that makes a large count *expected* rather than a bug — O(layers × constants × buckets)
— is [§1.4.3](01-basics.md). This chapter adds only what the count costs: **host memory and disk per
live recipe, nothing per recipe on the device** (§8.6.2).

---

## 8.9 Where each constraint lives

Every rule in this manual is a rule *somewhere* in the stack, and the layers do not have the same
strength. The distinction matters because a policy is a thing you can argue with a flag, a capability
limit is not, and we have never measured the bottom layer at all.

| constraint | layer | status |
|---|---|---|
| traced frames are guarded; a break splits a frame into graphs | **dynamo** | **MEASURED** — §8.2's table, and `TORCH_LOGS=graph_code` prints each graph as it is produced |
| each op still compiles to its own one-node recipe in eager mode | **bridge** | **source read** — `graph::compile()` always calls `synGraphCompile` (`backend/synapse_helpers/graph.cpp:646`), and `eager_mode_` only chooses `synGraphCreateEager` over `synGraphCreate` (`:140-146`) and otherwise touches the recipe name (`:643`) and bookkeeping (`:685`). **MEASURED**: 768 TPC kernel events for one eager `y.copy_(x)` ([§2.4](02-regions.md)) |
| the default is **static** compilation: a new shape is a new recipe | **bridge** | **MEASURED** at the recipe level — §8.2's table, notes §7 |
| "`dynamic=True` is turned into static compilation" | **bridge** | **ASSUMED (vendor)** — the standard account, not something we instrumented ([§1.4.1](01-basics.md)) |
| the bridge **has** a dynamic path — this is a *default*, not a capability limit | **bridge** | **source read** — `get_dynamic_config_value()` reads dynamo's `assume_static_by_default` (`passes.py:543-556`, the read at `:554`) and the DS/SIF path is gated on `optim_output_sif_ds` (`passes.py:2260,2282`); `force_static_compile` short-circuits both (`:523,2278`). We have never seen it produce a dynamic recipe on the default path |
| a recipe **can** be declared size-parametric, and is exact | **Synapse** | **MEASURED** — `DATA_TENSOR_DYNAMIC` with a `[min,max]` bound: exact M, N **and K** down to 1, rejects outside the bound with `synStatus 35` (notes §9.1) |
| a **static** recipe silently ignores the launch size | **Synapse** | **MEASURED** — the control: compiled for M=512, launched at M=64, returned `synSuccess` and wrote 262,144 elements (229,376 sentinel violations) (notes §9.1) |
| the recipe is genuinely re-parameterised, not masked | **Synapse** | **MEASURED** — `m_mappedSizePatch` 263,264 B (87 % of the image) for the dynamic recipe against 208 B (0.07 %) static (notes §9.1) |
| the HPU itself | — | **UNKNOWN** — nothing we have run is below Synapse |

**Read the row, not the headline.** "There are no dynamic shapes" is a **bridge** statement — tagged
ASSUMED as a mechanism and MEASURED as a recipe count. It is the right operating rule, because the
bridge will not hand you a dynamic recipe on any path we have seen, so every new shape is a new program.
It is the **wrong** conclusion about the device. The bridge defaults to static compilation; it is not
incapable of anything else, and the measurements above show the layer beneath it is size-parametric. Do
not infer a hardware limit from a layer above it — and do not let this section replace the fixed-shape
discipline until the throughput question below is answered.

**What keeps this from being a recommendation.** Exactness is not a schedule. Nothing measured above
says the MME is **re-tiled** for a smaller shape rather than **re-parameterised at the tiling chosen for
the compiled maximum** — a 64×64×64 GEMM run at a 512×512×512 tiling would be exact and slow. So
"dynamic shapes" may be true for correctness and still be "one big bucket" for throughput. That is
UNKNOWN #10, and it is the question that decides usability: measure per-size efficiency of a dynamic
recipe against static-per-shape before recommending it.

**Two practical traps, both measured** (notes §9.3–§9.4):

- **A user-section intermediate breaks the dynamic path.** Put the MME output that feeds the TPC node
  into an **application section** — persistent or not — and the recipe fails at the *exact* compiled
  shape with `Failed to perform DSD SIF patching` / status 35. Each node alone patches fine; the trigger
  is specifically the MME→TPC edge into a user section. **Keep intermediate tensors compiler-owned.**
- **`ENABLE_WIDE_BUCKET` is a footgun.** It removes the lower-bound *check* without making the recipe
  correct: at N=1 it wrote 138 elements and returned success. **Declare `min=1` instead. Never relax the
  bound.**

**Why you cannot reach the dynamic path from a caller at all:** it is a **kernel-library
declaration**, not an option. `GetKernelGuids` must set `supportsDynamicShapes = 1`
**and** a non-zero `nameHash.hashValue` (the SIF id), and `GetShapeInference` must actually resize the
output — all three mandatory. The upstream `add1` kernel has none of them, so "make add1
shape-parametric" means **editing the kernel library**.

---

## 8.10 Lazy, eager, compiled: what each is, and which one is a target

| mode | what it is | status |
|---|---|---|
| **compiled** | `torch.compile(fn, backend="hpu_backend")` — the only supported target | **the target** |
| **eager mode** | `PT_HPU_LAZY_MODE=0` — ops execute as issued; **each op still compiles to its own one-node recipe** (there is no region recipe) | a **reference point**, never an option |
| **lazy mode** | `PT_HPU_LAZY_MODE=1` — ops accumulate into an implicit graph closed at a mark-step | **legacy; not the default; not a viable target** |
| **eager placement** | a *node* property: the partitioner will not cluster it; it runs outside every recipe | a property of a compiled region, not a mode |

**Why eager is a reference point and not an option:** the project has measured the gap at **81×** — an
expert kernel at **29 ms eager against 0.356 ms compiled** ([§6.6.1](06-fast-iteration.md)). An eager
number is not a lower bound on the compiled number and cannot stand in for it. **And eager is not
cheaper for anything this chapter measures:** one eager op is one one-node recipe, so N ops is N
recipes, N submissions and N program downloads (§8.5) — the limit case of the residency story, not an
escape from it. **Dynamo is not involved: in eager mode there is no traced frame to group, so the
compilation is the bridge's** — `graph::compile()` always calls `synGraphCompile`
(`backend/synapse_helpers/graph.cpp:646`), and `eager_mode_` only swaps `synGraphCreateEager` for
`synGraphCreate` (`:140-146`) (**source read**) — and the count bears it out: one eager `y.copy_(x)`
produced **768** TPC kernel events and zero DMA events (**MEASURED**, [§2.4](02-regions.md)).

**Why lazy is legacy rather than deprecated:** the evidence proves **not the default**, not a vendor
deprecation. `PT_HPU_LAZY_MODE` is declared `default: 0` (**source read**,
`scripts/env_flags.yaml:125-127`), and the shipped autoload reads `os.getenv("PT_HPU_LAZY_MODE", "0")`
(**source read**, `python_packages/habana_frameworks/autoload/__init__.py:30`). Modern vLLM Gaudi is
`torch.compile`-only, so lazy is not a viable target — but do not call it deprecated without a vendor
statement saying so. Meanwhile the comment in
`python_packages/habana_frameworks/torch/internal/bridge_config.py:529` still claims "the default mode
is lazy" (**source read**). That comment is a **stale claim**, and it is worth keeping as a specimen
because it is the same failure mode [the TLDR](00-tldr.md) already documents: a vendor comment that was
true two releases ago, read as documentation today.

**The fourth row is the one that trips people.** *Eager placement* and *eager mode* are unrelated: the
first is a node that splits a recipe inside a compiled region; the second is an execution model.
[§2.4](02-regions.md) measures the first; this section separates it from the second.

**One trap worth naming here, because it lives in the same files.** A patched working tree in this
workspace — the 1.16 base plus **80 modified files (+4211/−863) and 17 untracked**, a heavily patched
research tree — carried a `synNativeComputeGraph*` "research API" whose comment claims it "skips recipe
preparation, address patching, PDMA download and per-recipe synLaunch", plus a `NativeRecipeMemory`
device-resident-program path. Count it by `grep -rho 'synNative[A-Za-z0-9_]*' synapse/include
synapse/src`: **210 occurrences in 14 files, 29 distinct identifiers** (21 `synNativeComputeGraph*`
entry points and types, 8 `synNativeGraph*` state enums). **None of it exists in vendor Synapse**: zero
occurrences in a clean `6de1f66` extraction, and zero `nativeCompute`/`NativeRecipeMemory` symbols in
the installed 1.24.2 `libSynapse.so` (**MEASURED**, notes *Provenance*, trap 1). Two residency models
live in this workspace; do not mix them, and do not cite the patched tree.

---

## 8.11 How to see it yourself

Five instruments, cheapest first. None needs a profiler.

| want | do this | what you get |
|---|---|---|
| **recipe count** | run with a **cold** recipe cache, then count the `.recipe` files in it (exact command in §8.13) | the number §8.2's table is built from |
| **subgraph count** | bridge log line `Number of subgraphs created: N` (`cluster_compiler.py:158`) | recipes per compilation, without touching disk |
| **program windows** | `LOG_LEVEL_ALL=6 LOG_LEVEL_SYN_PROG_DWNLD=0` with `HABANA_LOGS=<dir>`, then grep that directory for `ArcHbmMemMgr` / `HbmGlblMemMgr` (exact command in §8.13) | the §8.6.1 block, printed once at device acquire |
| **the graph itself** | `DUMP_POST_GRAPHS=1` and read the dump ([§2.10](02-regions.md)) | per-node `engine`, recipe membership, modelled cost |
| **which library did I load?** | grep `/proc/self/maps` from inside the process; `dpkg -V habanalabs-graph`; `nm -D --defined-only` on `libSynapse.so` for a symbol | whether the runtime is the packaged one, and whether a mechanism still exists by name |

**The trap that costs the most time: `hl_logger` levels are inverted.** `TRACE=0 DEBUG=1 INFO=2 WARN=3
ERROR=4 CRITICAL=5 OFF=6` (**source read**, 1.16
`swtools_sdk/hl_logger/include/hl_logger/hllog_core.hpp:27-33`). `LOG_LEVEL_ALL=5` is **near-silence**,
not maximum verbosity; `LOG_LEVEL_ALL=0` is loudest. This cost three wasted runs on this project, and it
is the single most likely reason a "the log prints nothing" investigation goes nowhere.

---

## 8.12 What this chapter leaves UNKNOWN

Stated rather than omitted, per the manual's convention. Each entry says what would settle it.

| # | UNKNOWN | status | what would settle it |
|---|---|---|---|
| 1 | **Whether `NUM_BUFF` is 64 in 1.24.2** | **INFERRED** — 64 is one method at three settings of `HBM_GLOBAL_MEM_SIZE_MEGAS` on the **global** window (192/3, 768/12, 3072/48 MiB all give 64); 1.16 is `16`; the constant itself was not read in a 1.24.2 source drop | a 1.24.2 source drop, or disassembly of `getChunkSize` |
| 2 | **What sizes the *arc* window (127.5 MiB), and where its 1.5 MiB goes** | **UNKNOWN** — unaffected by `HBM_GLOBAL_MEM_SIZE_MEGAS`; no key found that moves it; 64 × `0x1F8000` = 126 MiB against a 127.5 MiB window, and the 1.16 terms that could explain the shortfall (`round_down(chunkSize, ALIGN)`; `m_size = size - (m_addr - addr)`) are not distinguishable from the log, which prints the *given* size in both fields | a 1.24.2 source drop, or a log line that prints `m_size` |
| 3 | **Whether the per-stream and total split of the logged sizes is exactly `Size × 3`** | **INFERRED** from the source's `Size × MaxNumOfComputeStreams` and the measured 3× scaling of the global window; not measured with a single compute stream | one probe restricted to one compute stream |
| 4 | **`synUpdateMemoryConsumption` (new in 1.24.2)** | **UNKNOWN** — absent from the 1.16 source; could be the missing per-recipe attribution hook | a 1.24.2 header, or a probe that calls it and reads the device total |
| 5 | **Steady-state launch cost for a multi-MB program** | **UNKNOWN** — every measured program is small enough that the fixed path dominates (103.6 ms first call, ~0.5–0.7 ms steady state, size-independent at 1 K vs 1 M elements) | one probe with a program large enough for PDMA bytes to show |
| 6 | **Whether MME descriptors have their own device allocation** | **INFERRED** that they ride the ECB/dynamic-blob path into ARC HBM: the only program-data registrations in 1.16 are TPC kernels, the TPC printf buffer and H2D/static tensors | a 1.24.2 source read, or a launch of an MME-heavy graph with the device total sampled before and after |
| 7 | **Whether `DeviceGaudi::releaseRecipeMemory` is reached at all** | **UNKNOWN** — the 1.16 tree has its definition and no caller; either it is reached through an interface not in that tree, or the Gaudi1 path leaks per-recipe memory until device teardown | a 1.24.2 source read, or a compile/launch/destroy loop watching the device total |
| 8 | **Whether a rebuilt kernel is served the *old* bytes** (§8.7.3) | **INFERRED** — the key is provably constant and the cache provably hits (§8.8.2), but nobody has rebuilt a kernel and inspected which recipe came back. The earlier "the cache does not hit" reading was an artifact of `delete_on_init=true`, which wipes the directory at every process start (§8.7.4) | rebuild the kernel with `delete_on_init=false`, launch the same graph, and inspect which `.recipe` is used and whether its bytes match the pre-rebuild file |
| 9 | **The bucketing rule for a structural `int`** (§8.8.4) | **UNKNOWN** — `x[:n] * 2` gave 3 recipes for `n = 1,2,3` and 4 for `n = 1…7`, with `n = 4,5,6` reusing `n = 3`; growth is real and not one-per-value | a sweep over a wide contiguous value range, recording which recipe each value lands on |
| 10 | **Whether a dynamic recipe is re-tiled, or only re-parameterised at the compiled maximum** (§8.9) | **UNKNOWN** — and it is the question that decides usability: a 64×64×64 GEMM run at a 512×512×512 tiling would be exact and slow | per-size efficiency of a dynamic recipe against static-per-shape |
| 11 | **Whether an *unbounded* dynamic maximum is possible** | **expected no** — the compiler must size descriptors, the index space, data chunks and the workspace for the largest shape (8448 B dynamic against 256 B static) | a source read of the descriptor sizing, or a probe that omits `max` |

---

## What to carry forward

1. **Three counts, three units.** Graphs (dynamo frames, split by breaks) ≥ recipes (maximal
   `hpu_cluster` runs) ≥ submissions (one launch per recipe per execution). A `torch.compile` boundary
   is not a recipe boundary.
2. **One decorated function is not one program.** Case C: two graphs, two recipes, from one function
   with one host read in it.
3. **Ops do not multiply recipes; boundaries do.** Three clusterable ops are one recipe; one eager node
   between them is three.
4. **A compiled recipe holds no HBM.** `synGraphCompile` allocates host pages mapped into the device;
   `synRecipeDestroy` frees host memory only; an unlaunched recipe costs zero device memory.
5. **Every launch downloads the program again**, into a rotating chunk of a fixed per-stream window —
   the vendor's own "assuming no caching … to avoid the pdma". Past one chunk the launch throws; it
   does not spill.
6. **Budget `97376 MiB − 481 MiB − workspace − weights`, then stop subtracting per-recipe terms.** 24
   recipes compiled and launched moved free HBM by **0.0 MiB**; host RSS moved ~4.6 MiB per recipe.
   Three totals are in circulation (98304 driver / 97376 Synapse / 96895.2 torch) and the 768 MiB idle
   baseline is a *usage*, not a fourth one.
7. **The graph count costs host memory and disk, not device memory.** If HBM grows with the graph
   count, look at sections, the workspace's max-sizing, or retained tensors — not at recipes.
8. **`RECIPE_CACHE_SIZE` is a Gaudi1 key.** On Gaudi 2 there is no device recipe cache; there is a
   rotating ring, and the cache that evicts is the bridge's **host** LRU (eviction ⇒ recompile).
9. **Only compiled is a target.** Eager is a reference point (**81×** off, [§6.6.1](06-fast-iteration.md))
   and compiles one one-node recipe per op; lazy is legacy, not deprecated; eager *placement* is
   neither.
10. **`hl_logger` levels are inverted.** `LOG_LEVEL_ALL=0` is loudest, `=5` is near-silence.
11. **The disk cache's key cannot see your build.** The filename's middle field is the constant
    `_a16fb581eee_`: `libs_env_hash()` hashes every file to zero and never matches an environment name,
    so a kernel rebuild or a flag change does not miss the cache. Keep one cache directory per
    *(build, configuration)* pair, or use `delete_on_init=true`.
12. **An `int` can be data or structure.** As a 0-d tensor it is a graph *input* — one recipe across
    values and across processes. As a Python int it is two recipes in a process (one per value if you
    force `dynamic=False`), and one per **cold process** — which is the unbounded case. A structural int
    (`x[:n]`) cannot be escaped at all.

---

## 8.13 Examples

The chapter adds no files, so these are inline and runnable. They are the smallest commands that
reproduce §8.2, §8.5, §8.6 and §8.8.

**Count recipes for a set of tiny cases** (cold cache each time — §8.2's table):

```bash
export PT_HPU_RECIPE_CACHE_CONFIG="$PWD/rc-cache",false,1024,false
for case in a b c d; do
  rm -rf "$PWD/rc-cache"    # cold, so the count is unambiguous
  python case_${case}.py    # one decorated function per case
  echo "$case: $(ls "$PWD"/rc-cache/*.recipe 2>/dev/null | wc -l) recipes"
done
```

**See the program windows** (§8.6.1):

```bash
HABANA_LOGS="$PWD/habana-logs" \
LOG_LEVEL_ALL=6 LOG_LEVEL_SYN_PROG_DWNLD=0 LOG_LEVEL_SYN_DEVICE=0 \
  python your_probe.py
grep -h 'ArcHbmMemMgr\|HbmGlblMemMgr' "$PWD"/habana-logs/* | sed 's/.*\] //'
```

**Watch a graph count fail to cost device memory** (§8.6.2):

```python
import torch, torch_hpu
MiB = 2 ** 20                      # mem_get_info returns BYTES
f = torch.compile(lambda t: t + 1, backend="hpu_backend")
free0, total0 = torch.hpu.mem_get_info()
handles = [f(torch.arange(1024, dtype=torch.float32, device="hpu")) for _ in range(24)]
free1, total1 = torch.hpu.mem_get_info()
print(f"free {free0 / MiB:.1f} -> {free1 / MiB:.1f} MiB, total {total0 / MiB:.1f} -> {total1 / MiB:.1f} MiB")
```

Read the **total**, not the free: `synDeviceGetMemoryInfo()` reports `free = 0` on this build, and the
window is carved out of the total (§8.6.3).

**Count the recipes an `int` costs** (§8.8). Save as `ints.py`; the cache directory is the last
argument and must persist between runs for the cross-process arm:

```python
# python ints.py tensor 1,2,3,4,5 ./rc    -> 1 recipe
# python ints.py int    1,2,3     ./rc    -> 2 recipes   (3 with dynamic=False)
import glob, os, sys, time
mode, values, cache = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3]
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{cache},false,1024,false"   # false: keep across runs
import torch, torch_hpu

def n_recipes():
    return len(glob.glob(os.path.join(cache, "*.recipe")))

# Compile ONCE. Compiling inside the loop resets dynamo's cache and measures nothing.
f = torch.compile(lambda x, k: x + k, backend="hpu_backend", fullgraph=True, dynamic=None)
t = torch.arange(1024, dtype=torch.float32, device="hpu")
for n in values:
    arg = torch.tensor(n, dtype=torch.int64, device="hpu") if mode == "tensor" else n
    t0 = time.perf_counter()
    f(t, arg)
    torch.hpu.synchronize()
    print(f"{mode} n={n} {(time.perf_counter() - t0) * 1e3:7.1f} ms  recipes={n_recipes()}")
```

Three arms settle §8.8: run `tensor 1,2,3,4,5` (one recipe), then `tensor 42` **in a new process**
against the same directory (**zero** new recipes), then `int 1,2,3` (two recipes — or three if you
change `dynamic=None` to `dynamic=False`). For the structural case, replace the body with
`lambda x, n: x[:n] * 2` and pass Python ints; the recipe count grows with the value set.

**Dump the graph and read placement** (§8.11, and [§2.10](02-regions.md) for the reader):

```bash
DUMP_POST_GRAPHS=1 python your_probe.py
ls .graph_dumps/ | head            # a dot-directory: a non-recursive ls of the parent hides it
```

**See the integer graph, not just the count** (§8.8.3): `TORCH_LOGS=graph_code` prints the two graphs —
the first with `add = l_x_ + 1`, the second with `L_k_: "Sym(s16)"`.

---

## Provenance of every figure

Every number in this chapter, with its source. "notes §n" is
[`notes/recipe-residency-and-memory-budget.md`](notes/recipe-residency-and-memory-budget.md), whose
numbering this table tracks; "notes *Provenance*" is that file's unnumbered preamble. `file:line` is a
source read of the bridge source tree at 1.24.2 — the version the running process loads; Python paths
are the installed `python_packages/habana_frameworks`, C++ paths are relative to the bridge source root
— or of the clean 1.16 Synapse extraction (base `6de1f66`), flagged as 1.16 where it matters. Rows
whose source is another chapter are cross-references, not restatements.

| figure | source |
|---|---|
| A–D table: graphs / breaks / ops / recipes — A 1/0/1/1, B 1/0/3/1, C 2/1/3/2, D 1/0/3/1 | MEASURED — notes §7 |
| recipe filename `<graph hash>_<libs_env_hash>_syn1.24.2.79e284b`; middle field the constant `_a16fb581eee_` | MEASURED — notes §8.1; the three arms below |
| `hash64_file_content()` returns 0 for every file (shadowed `hashRes`) | source read — `cache_version.cpp:107,123,126-127,138` |
| `check_env_fo_hashing()` never matches (space-padded needles vs `NAME=VALUE`) | source read — `cache_version.cpp:141-172,150,165,170,208-215` |
| `libs_env_hash()` is `hash_combine` over 0; colon-list branch unreachable (exit 134) | source read — `cache_version.cpp:174-222`, `:184-198` |
| filename is the cache key; `PT_RECIPE_CACHE_IGNORE_VERSION` drops the suffix | source read — `hpu_recipe_cache.cpp:294-308,317,299-301` |
| three arms (stock lib / one byte appended / `DISABLE_TENSORS_PINNING=true`) → one identical filename | MEASURED — notes §8.1 |
| same filename and size (26,270 B), different md5 (`1ca32f92…` vs `a76dd872…`) | MEASURED — `md5sum` over two fresh runs' `.recipe` files (2026-11-13) |
| 8 recipes compiled+launched → 0.0 MiB device change | MEASURED — notes §2b |
| 24 recipes: device 96889.7 → 96889.7 MiB; host RSS 2104.6 → 2215.2 MiB (+110.6); ~4.6 MiB/recipe | MEASURED — notes §2c |
| disk 0 → 0.4 MiB / 9 files | MEASURED — notes §2c |
| `PT_HPU_RECIPE_CACHE_CONFIG` fields `path, delete_on_init, max_size_MB, allow_NFS`; default `",false,1024,false"` — an empty path, so the disk cache is **off** until you name a directory | source read — `bridge_config.py:71-86` (the four fields); `scripts/env_flags.yaml:96-97` (the default; `1024` at `:114` is a doc-comment example, not the default) |
| card HBM 98304 MiB (driver) / 97376 MiB = 95.09 GiB (Synapse) / 96895.2 MiB = 94.62 GiB (torch) / 768 MiB used at idle | MEASURED — notes §2b, re-run for this chapter with `hl-smi` and `torch.hpu.mem_get_info()` on 2026-11-13 |
| `torch.hpu.mem_get_info()` returns **bytes** (101601996800 = 96895.2 MiB) | MEASURED — re-run for this chapter |
| idle baseline ~768 MiB | MEASURED — `01-basics.md:394` |
| global window 192 MiB = 3 × 64; arc window 127.5 MiB (`0x7F80000`); ~161 MiB remainder; ~481 MiB total | MEASURED — notes §2b |
| chunk sizes `0x1F8000`, `0x300000`; **64** chunks, **global window only** | MEASURED + INFERRED — notes §2, §2a |
| 64 × `0x1F8000` = `0x7E00000` = 126 MiB against `0x7F80000`; 1.5 MiB never handed out | MEASURED arithmetic — notes §2a |
| `round_down(chunkSize, ALIGN = 128)`; `m_size = size - (m_addr - addr)` | source read, 1.16 — `arc_hbm_mem_mgr.cpp:44-48`, `arc_hbm_mem_mgr.hpp:36`, `hbm_global_mem_mgr.cpp:24-27` |
| `NUM_BUFF = 16` (1.16) | source read, 1.16 — `nbuff_allocator.hpp:28`; chunk `size / NUM_BUFF` at `hbm_global_mem_mgr.cpp:39-43` |
| `HBM_GLOBAL_MEM_SIZE_MEGAS` default 64 | source read, 1.16 — `habana_global_conf.cpp:594-598` |
| key ignored without `ENABLE_EXPERIMENTAL_FLAGS`; 256→768 MiB, total 96319.2 (−576); 1024→3072 MiB, total 91359.2 (−5536 ≠ 2880) | MEASURED — notes §2 |
| 4 GB-alignment workaround | source read, 1.16 — `device_scal.cpp:266-290` |
| `MaxNumOfComputeStreams = 3`; pools allocated `Size × 3` | source read, 1.16 — `scal_dev.hpp:58`, `device_scal.cpp:242-247` |
| `hbmSharedSize = 32 MiB` (1.16) | source read, 1.16 — `device_scal.hpp:126` |
| 103.6 ms first call (size-independent at 1 K and 1 M elements); ~0.5–0.7 ms steady state | MEASURED — notes §5 |
| ~50 ms dynamo + ~52 ms bridge ≈ 100 ms per new recipe; ~28 hours for a million | MEASURED — notes §2c |
| eager 29 ms vs compiled 0.356 ms = **81×** | MEASURED — [§6.6.1](06-fast-iteration.md) |
| one eager `y.copy_(x)` → 768 TPC kernel events, zero DMA events | MEASURED — `02-regions.md:1198` |
| eager mode still compiles each op: `synGraphCompile` is unconditional; `eager_mode_` only picks `synGraphCreateEager` and the recipe name | source read — `backend/synapse_helpers/graph.cpp:140-146,643,646,685` |
| `HABANA_PGM_LRU_MAX` 100 000 eager / 30 000 lazy | source read — notes §1 (bridge) |
| host threshold: gcfg default 0 disables the check; the bridge substitutes 90 at the first dynamic bucket | source read — `scripts/env_flags.yaml:428-430`, `misc_utils.cpp:32-34`, `hpu_recipe_cache.h:130` |
| eviction in 512 MiB chunks (`512LL * 1024 * 1024`) | source read — `hpu_habana_launch_op_pt.cpp:4211` |
| `RECIPE_CACHE_SIZE` 500 MiB / `RECIPE_CACHE_BLOCK_SIZE` 128 KiB; device HBM; Gaudi1 only | source read, 1.16 — `habana_global_conf.cpp:1121-1133`, `device_recipe_addresses_generator.cpp:139-160,219-260`, `recipe_cache_manager.cpp:40` |
| SCAL/Gaudi2 references the recipe cache nowhere; `RECIPE_CACHE_SIZE=262144` changes nothing | source read + MEASURED — notes §4 |
| `synGraphCompile` allocates no device memory; `madvise(MADV_DONTFORK)` rationale | source read, 1.16 — `recipe_allocator.cpp:15-30`, `memory_allocator_utils.cpp:12-40` |
| `synRecipeDestroy` frees host memory only | source read, 1.16 — `recipe_handle_impl.cpp:18-24` |
| workspace is a caller argument (`pWorkspace`) | source read, 1.16 — `include/synapse_api.h:612-617` |
| `synRecipeGetAttribute` attribute set; `synRecipeSectionGetProp` const-only, host pointer | source read, 1.16 — `device_agnostic_recipe_processor.cpp:252-338`, `syn_singleton_section.cpp:206-270` |
| vendor "assuming no caching … to avoid the pdma" | source read, 1.16 — `arc_hbm_mem_mgr.hpp`, `hbm_global_mem_mgr.hpp` |
| `hpu_backend` → `aot_autograd(..., partition_fn=hpu_partition)` | source read — `backends.py:52-79` (lines 62, 66-71, 78) |
| `hpu_compiler_inner` four phases in order | source read — `compilers.py:132-181` (168, 171, 174, 177) |
| `hpu_inference_compiler` passthrough / freezing branch | source read — `compilers.py:205-214` |
| `hpu_partition` → `default_partition`, fallback `min_cut_rematerialization_partition` | source read — `partition_fn.py:253-291` (282, 291) |
| cluster membership = `placement == "hpu_cluster"` | source read — `partitioner.py:27-29` |
| `HabanaPartitioner(CapabilityBasedPartitioner)`, `allows_single_node_partition=True` | source read — `partitioner.py:32-38` |
| placement written by the pass; mismatch raises | source read — `passes.py:1592-1596` |
| one fused submodule → `get_callable_recipe`; `subgraph_cnt`; "Number of subgraphs created" | source read — `cluster_compiler.py:132,142-145,158` |
| `get_callable_recipe` docstring: "…or just returns unchanged module to run it eagerly…" | source read — `recipe_compiler.py:361-372` |
| the bridge's dynamic path: `get_dynamic_config_value()` reads `assume_static_by_default`; the DS/SIF gate; `force_static_compile` short-circuits both | source read — `passes.py:543-556` (read at `:554`), `:2260,2282`, `:523,2278` |
| `hl_logger` levels inverted `TRACE=0 … OFF=6` | source read, 1.16 — `hllog_core.hpp:27-33` |
| patched tree's `synNativeComputeGraph*`: 210 occurrences in 14 files, 29 identifiers; absent from the clean tree and from 1.24.2 symbols | MEASURED — notes *Provenance*, trap 1 |
| patched tree: 80 modified files (+4211/−863), 17 untracked | MEASURED — notes *Provenance* |
| `synDeviceGetMemoryInfo()` returns `free = 0`; measure the total | MEASURED — notes §2b |
| 145 public entry points in `synapse/include/synapse_api.h` / 155 exported `syn*` symbols; no residency API | source read + MEASURED — notes §3 |
| DSD: exact M, N **and K** down to 1; `synStatus 35` outside `[min,max]`; static control wrote 262,144 elements at M=64 | MEASURED — notes §9.1 |
| `m_mappedSizePatch` 263,264 B (87 %) dynamic vs 208 B (0.07 %) static; 10/112 blobs patched | MEASURED — notes §9.1 |
| DSD is a kernel-library declaration: `supportsDynamicShapes`, non-zero `nameHash`, real `GetShapeInference` | MEASURED — notes §9.4 |
| `ENABLE_WIDE_BUCKET` + experimental flags writes 138 elements at N=1 | MEASURED — notes §9.4 |
| workspace 8448 B dynamic vs 256 B static | MEASURED — notes §9.4 |
| a user-section intermediate → `Failed to perform DSD SIF patching` / status 35 | MEASURED — notes §9.3 |
| 0-d `int64` tensor, `n = 1…10`: 1 graph, 1 recipe; signature `L_k_: "i64[][]hpu:0"`; 142 ms first, 0.3–0.5 ms after | MEASURED — the `ints.py` listing in §8.13 |
| 0-d tensor across processes: 151.3 ms / 1 recipe, then 83.7 ms / **0 new recipes** at `n = 42` | MEASURED — the `ints.py` listing in §8.13 |
| Python int: 2 recipes at `dynamic=None` (flat to 10 values) and at `dynamic=True`; 3 for `dynamic=False` and for `specialize_int=True` | MEASURED — the `ints.py` listing in §8.13 |
| Python-int mechanism: first graph `add = l_x_ + 1`, second `L_k_: "Sym(s16)"` | MEASURED — `TORCH_LOGS=graph_code` |
| cold process at `n = 42` after a Python int: 186 ms, 2 recipes | MEASURED — the `ints.py` listing in §8.13 |
| structural int `x[:n] * 2`: 3 recipes for `n = 1,2,3`, 4 for `n = 1…7` | MEASURED — the `ints.py` listing in §8.13 |
| "one recipe per maximal contiguous run of `hpu_cluster` nodes"; 24/24 arms | MEASURED — [§2.4](02-regions.md) (cross-referenced, not restated) |
| specialisation count as a first-class metric; thrash signature | [§2.5.3](02-regions.md) (cross-referenced) |
| O(layers × constants × buckets) is expected | [§1.4.3](01-basics.md) (cross-referenced) |
| "`dynamic=True` is forced to static compilation" (ASSUMED, vendor) | [§1.4.1](01-basics.md) (cross-referenced) |
| `DUMP_POST_GRAPHS` reading; the dot-directory trap | [§2.10](02-regions.md) (cross-referenced) |
