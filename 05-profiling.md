# Chapter 5 — Profiling, and the identity chain

**Read this before you take a profile you intend to act on.** A trace that cannot be joined to
recipes and then to code is a *timing* instrument, not a *diagnostic* one — and the join is decided
**before the run**, not during analysis. This chapter is about that decision, the chain it enables,
and the traps that silently break it.

The statistical material lives where it always did: **§1.8** (what a duration means, markers, unions,
the gap decomposition), **§1.9** (the profiler shim), **§2.9** (ten rules of measurement discipline),
**§2.10** (`DUMP_POST_GRAPHS`). This chapter does not repeat them; it covers **how a profile is tied
to the code that produced it**, which none of them do.

---

## 5.1 The failure, measured

Two traces from the same campaign, same code family, both archived:

| run | region spans | distinct compile ids | resolvable to `file:line:function` | why |
|---|---:|---:|---:|---|
| **E4** | 196 | **49** | **49 of 49** | `TORCH_TRACE` was set at run time |
| **W0** | 1,228 | **121** | **0 of 121** | no `TORCH_TRACE` directory exists for it |

W0 is the trace behind §3.6.4's device-time decomposition and §3.6.5's KV-slab finding. It carries
1,228 region spans and 121 compile ids, and **not one of them can ever be named**, because the run
did not set one environment variable. The spans are in the trace; the map from compile id to source
line is *not* in the trace and is **unrecoverable afterwards** — it is written by dynamo during the
run, as a side effect.

**⇒ The rule this chapter exists to state: identity is a run-time decision. A profile taken without
the identity knobs cannot be repaired by any amount of analysis.**

---

## 5.2 The chain, and which legs are keys

```
device event --(recipeId)--> recipe --(recipeName)--> post-graph dump --> node types / engines
     ^
     |
 launch_recipe (inside the region span) --(pairing)--> enqueueWithExternalEventsExt
     |
     +--(containment)--> compile id --(TORCH_TRACE)--> file:line:function
```

| # | leg | key | availability |
|---|---|---|---|
| 1 | device event → **recipe** | `args["recipeId"]` / `recipeHandle` | **always present** on engine-labelled events; 1,604 of 1,604 enqueues carry it on W0, 110 distinct recipes |
| 2 | device event → **node** | event `name` = dump node `name` (`24_memcpy_116119_internal`) | **always present**; joins exactly, but only within one run (names embed per-run ids) |
| 3 | recipe → **dump** | `args["recipeName"]` = the dump's `graph_name`; or `recipeId` = the dump header's `recipe_debug_id` | present — but **`recipeName` is the empty string on 1,252,091 of 1,253,723 device events (99.87 %)**, so key on the id, not the name |
| 4 | launch → **region** | **containment** in a `Torch-Compiled Region` span | computable from the trace alone — **but only for the launch, never the device event** (§5.2.1) |
| 5 | region → **code** | `Torch-Compiled Region: <compile_id>` → `TORCH_TRACE` `compilation_metrics` → `co_filename:co_firstlineno:co_name` | **only if `TORCH_TRACE` was set before the run** |

Legs 1–3 are **keys**. Leg 4 is containment. Leg 5 is a key that only exists if you paid for it.
**The leg that breaks in practice is 5**, and it breaks before the profile is taken.

### 5.2.1 Attribute the LAUNCH, never the device event

A device event is **asynchronous**: it does not execute inside the host span that enqueued it.
Attributing device events to regions by timestamp containment therefore fails — **measured: 3,551 of
4,309 device events fell outside every region** on a 3-block program whose launches were **0 of 60**
outside. The object to attribute is the launch (`launch_recipe`), which runs on the same thread as
the region annotation.

### 5.2.2 The one leg with no key at all: region → recipe

`launch_recipe` carries **no args** — no handle, no id. `LaunchRecipe` (the runtime-thread
submission) also carries no args. So region → recipe has to be **paired**, and the pairing is only
sound when the two host streams are 1:1:

| pairing | 3-block program | real model (W0) |
|---|---|---|
| `launch_recipe[i]` ↔ `enqueueWithExternalEventsExt[i]` | **exact — 20/20/20, one recipe per compile id** | **fails — 1,576 launches vs 1,604 enqueues, 24.6 % modal share, 0 of 118 compile ids consistent** |
| window ("next enqueue after this launch") | ambiguous 35 of 60 | — |

**So on a real model this leg is currently unavailable, and a tool that produces a confident table
anyway is lying to you.** Emit `<ambiguous>`. Until the join exists, `attribute_chain.py` verifies
the 1:1 property and refuses when it does not hold.

**The identity is already in the process — it is being filtered out.** The bridge annotates the FX
graph name on the Python thread, inside the region span (`GraphExec::launch`,
`habana_eager/graph_exec.cpp:794`), and the profiler drops it because its `mandatory_events` list is
a regex set that no graph name matches (`backend/profiling/kineto_activity_profiler.cpp:517`). The
name in the post-graph dump is that same FX name with **one substitution** — `jit` → `syn`
(`backend/kernel/hpu_habana_launch_op_pt.cpp:247-248`) — plus a compile-time suffix
(`get_unique_recipe_name`, `backend/synapse_helpers/graph.cpp:643`). So the join is reachable with
two small edits and no new plumbing: let the annotation through the filter, and emit the
substituted name.

**Our fork carries this work:** <https://github.com/NatTuck/gaudi-pytorch-bridge>, branch
`gaudi2-profiling-identity` (off upstream `v1.24.2`). At the time of writing it carries the
`export_chrome_trace` fix from §5.7.1 **applied and verified**, and the recipe-identity change
**identified but not applied** — the local bridge build does not link here (cmake 4.4.x against a
patch set that needs 3.31.6), so nothing is claimed about its behaviour. `PATCHES.md` on that
branch carries the exact lines, the reasoning, and the two candidate strings that turned out to be
wrong.

---

## 5.3 The identity knobs — set before the run

```python
# BEFORE `import torch`. Both are read early and neither can be set afterwards.
os.environ["TORCH_TRACE"] = spec_dir          # compile id -> file:line:function
os.environ["DUMP_POST_GRAPHS"] = dump_dir     # one JSON per compiled graph, per-node engine field
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{cache_dir},false,8192,false"   # cold => graphs get built
```

- **`TORCH_TRACE` is the one people omit**, and its absence is the difference between the two rows in
  §5.1. It costs **disk, not time**: ~450 MB per arm measured (5.89 GB across 13 arms), so it is a
  data-management decision as well as a profiling one.
- **A cold recipe cache is a precondition, not a nicety.** A recipe deserialised from the cache is
  never re-dumped, so a warm cache gives an **empty dump** that reads as "nothing compiled".
- **`mkdir -p` the dump directory.** If it does not exist the compiler treats the path as a **file**
  — one run here turned it into an 18,114-byte file and the *next* run died on `makedirs`.
- **Per rank.** With TP > 1, rewrite `DUMP_POST_GRAPHS` per rank before the first compile; a shared
  directory reproduces the compiler's own race (`synStatus 26`).

**A run without `TORCH_TRACE` is discarded and re-run, not analysed.** That is a gate, not advice:
an unattributable trace costs more in wasted analysis than the run costs to repeat.

---

## 5.4 The activity mode is a decision with three answers

`ProfilerActivity.HPU` alone **drops every compile id**, because `Torch-Compiled Region` is a
CPU-side dynamo annotation. Measured on identical runs: HPU-only gives **0 region spans**; CPU+HPU
gives **3 compile ids, 60 spans**. The trace still times the device perfectly — it simply cannot be
attributed.

**And the cost of including CPU is smaller than the manual's own rule implies.** On the same two runs
of the same program:

| activity | merged union | TPC | MME | DMA |
|---|---:|---:|---:|---:|
| CPU+HPU | 0.593 ms | 0.102 | 0.152 | 0.381 |
| HPU-only | 0.598 ms | 0.101 | 0.153 | 0.385 |

**Device durations are unaffected** — 0.8 % on the merged union, ≤1 % per engine. What CPU collection
*does* corrupt is the **host span and queue depth** (the ~3× host-span inflation in §1.9, and the
`23.3 → 1.67` ops-outstanding collapse in §2.9 rule 6), which is what makes a device-bound arm look
host-bound.

⇒ **Split the old rule into three:**

| question | activity | why |
|---|---|---|
| device **durations / unions** | either — they agree | measured above |
| **host span, queue depth, overlap / concurrency** | **HPU-only** | CPU collection inflates these ~3× |
| **attribution to code** | **CPU+HPU, always** | the compile id is a CPU annotation |

For most work the answer is therefore **CPU+HPU**, and the "profile HPU-only" rule should be read as
"do not quote host-side numbers from a CPU+HPU trace", not "never collect CPU".

---

## 5.5 The clip you must declare

Every per-step figure in this manual depends on a **clip**, and the manual has never defined one.
State it in the artifact, because two counts of the same thing that differ by the step count are one
measurement, not two (§2.9 rule 8).

The convention used here:

```python
steps = count of `python_function` events whose name ends in "_execute_model_generic"   # or 1
```

and then every per-step figure is `total / steps` over an explicitly named window (steady state only
— exclude warm-up and compilation). **A per-step number without its clip is not a number.**

---

## 5.6 Collect

```python
from torch.profiler import ProfilerActivity, profile

acts = [ProfilerActivity.HPU, ProfilerActivity.CPU]      # CPU included => compile ids survive
prof = profile(activities=acts, record_shapes=False)
prof.start()
...                                                       # steady-state window only
torch.hpu.synchronize()
prof.stop()

# ONE workaround, required on torch 2.13 / Habana 1.24.2 (see §5.7.1):
prof.profiler.export_chrome_trace(path)                   # torch's own export passes 3 args; the shim takes 2
```

**Pre-flight every trace** with `devprobe.py` (§1.9): require `kernel` and `hpu_op` in the category
histogram, and — for an attribution trace — require **`Torch-Compiled Region` spans**. A trace that
has device events and no region spans is unattributable; say so and stop.

**And gate the arm before quoting it** (`pending ≥ 2`, `device_busy ≥ 80 %`, §1.8).

---

## 5.7 Three traps that cost runs

**5.7.1 An export bug on this build, and a second one that we suspected and then RETRACTED.**
(a) **MEASURED, reproducible, reported upstream** (`HabanaAI/gaudi-pytorch-bridge` issue 23):
`tensorboard_trace_handler` raises `TypeError: takes 2 positional arguments but 3 were given` —
torch 2.13 calls `export_chrome_trace(path, metadata)` while the Habana shim at
`activity_profiler.py:63` and `:195` defines `(self, path)`. It fires **at profiler exit**, after the
window has completed, so the run is lost. Call the wrapped profiler's method directly:
`prof.profiler.export_chrome_trace(path)`.

(b) **RETRACTED.** An earlier version of this section claimed a second defect: that the exporter
needs the output file to pre-exist, and otherwise emits a trace with `cpu_op` events and no device
events. **We could not reproduce it** — four attempts, three plain and one with `TORCH_TRACE` and
`DUMP_POST_GRAPHS` set, all exported correctly to a path that did not exist. The single observation
that prompted the claim is best explained by the output directory being removed by an external
cleaner *during* the run, which this sandbox does. **Do not add a pre-create-the-file step on the
strength of it.** *The claim is kept here, marked, rather than deleted: a retracted incantation is
worse than no incantation, because the next reader cannot tell whether it is load-bearing.*

If you *do* see `output_json.cpp: Failed to open … No such file or directory` followed by a
`json.exception.parse_error … empty input`, check first that the output **directory** still exists —
that is the cause we could reproduce, and it is a property of the environment, not the library.

**5.7.2 The dumps land in a dot-directory.** `DUMP_POST_GRAPHS=<dir>` writes to
`<dir>/.graph_dumps/`, so a non-recursive `listdir` reports **0 files** while the dumps sit one level
down, and `glob("**")` **skips hidden directories entirely**. Use `os.walk`. This is not hypothetical:
the first version of `attribute_chain.py` reported "0 recipes" against a directory holding 6.

**5.7.3 Sum versus union flips the answer.** On a 30-line program, summing per-engine durations made
a matmul block look DMA-bound (**DMA 0.673 ms > MME 0.595**); the unions give the correct **MME 0.152
> DMA 0.107**. The cause is per-TPC replication — 400 DMA events against 160 MME events. §2.9 rule 1,
reproduced end to end.

---

## 5.8 The worked example

`examples/05-profiling/` carries a program built so the answer is known in advance — three blocks,
one engine each, each its own `torch.compile` region, called from an **uncompiled** driver:

| source location | compile id | recipe | dump | node types | engine union |
|---|---|---|---:|---|---|
| `recipe_chain_demo.py:63:mme_block` | `0/0` | 64136 | `graph_0000_…` | `DmaMemcpy×2, GEMM×2` | **MME 0.152 ms**, DMA 0.107 |
| `recipe_chain_demo.py:68:tpc_block` | `1/0` | 63742 | `graph_0001_…` | `fused_kernel_…×1` | **TPC 0.057 ms** |
| `recipe_chain_demo.py:76:dma_block` | `2/0` | 61405 | `graph_0002_…` | `DmaMemcpy×1, add_fwd_bf16×1` | **DMA 0.069 ms**, TPC 0.045 |

`attribute_chain.py` prints that table and then **verifies it**: each block's dominant engine,
measured, is compared against the engine the code was written to use. It passes — and it refuses
(`AMBIGUOUS`) on the real model, where the 1:1 pairing does not hold.

The node types are the independent confirmation: the two GEMMs appear as two `GEMM` nodes, the four
elementwise ops **fuse to one** TPC kernel, and the copy block is one `DmaMemcpy` plus the `add`
broadcast. **Read the dump, not just the engine mix** — a fused kernel and four kernels are the same
engine and a different cost.

### 5.8.1 Two more keys available from the backend metrics

With `PT_HPU_METRICS_GC_DETAILS=1` and `PT_HPU_ENABLE_CACHE_METRICS=1` (set before the metrics module
is imported), `habana_frameworks.torch.hpu.metrics.metric_global(name).stats()` returns, per compile:

```
graph_compilation.recipe = [('.graph_dumps/graph_0000_…_3', 9083), ('.graph_dumps/graph_0001_…_4', 161102)]
recipe_cache  = RecipeHit {cache_hash: n}, RecipeMiss {cache_hash: n}
```

`graph_compilation.recipe` carries the **`graph_name`** — the same string as the trace's `recipeName`
— plus that graph's **compile duration in µs**, so "which recipe is expensive to compile" is a
lookup. `recipe_cache` keys on the **cache hash** (the `.recipe` filename in the cache directory),
which is a third identifier space. `stats()` returns a **list of pairs**, not a dict.

---

## 5.9 What this chapter does not establish

| # | open | what would settle it |
|---|---|---|
| 1 | **A key for region → recipe.** Currently a pairing, and it fails on a real model (§5.2.2). The identity is already annotated but filtered out, and the dump name is one `jit`→`syn` substitution away — so the fix is two small bridge edits, not new plumbing. **Our fork carries the patch, unverified** (it does not link here; §5.2.2). | Run a patched build on a real model and confirm the trace event name matches the dump's `graph_name`; then `attribute_chain.py`'s `AMBIGUOUS` path becomes dead code kept only for foreign traces. |
| 2 | **Whether the compile-time metric can carry the compile id.** `graph_compilation.recipe` carries the graph name and duration, not the compile id. | Read the metric per compile — reset and re-read around a single region — and check whether a compile id is derivable; or ask the vendor. |
| 3 | **`recipeId` == `recipe_debug_id` in general.** Established for this build and one run (three recipes, all matching). | A run where the two disagree, or the header documented. |
| 4 | **Attribution of eager-placed nodes.** A node placed `eager` runs outside every recipe and emits no `LaunchRecipe` (§2.1), so it has a device event and no recipe. | Count them explicitly and report the residual rather than folding it into a recipe. |
| 5 | **The cost of `TORCH_TRACE` in wall time** (as opposed to the ~450 MB/arm of disk). | A timed A/B with and without it on the same arm. |

---

**Tagging convention** as throughout: **MEASURED** / **ASSUMED** / **INFERRED** / **UNKNOWN**, see
`README.md`. The figures in this chapter are from one 4-AIP Gaudi 2 container, torch 2.13.0 /
Habana 1.24.2, driver 1.24.2 — and the export bug in §5.7.1 is **version-specific** (reported
upstream as `HabanaAI/gaudi-pytorch-bridge` issue 23); check it before assuming the workaround is
still needed. §5.7.1 also records a **retracted** claim, deliberately kept visible.
