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

**Our fork closes it, and it is verified.** <https://github.com/NatTuck/gaudi-pytorch-bridge>,
branch `gaudi2-profiling-identity` (off upstream `v1.24.2`). Two changes: let the graph-name
annotation through the filter, and emit the `jit`→`syn` form at `GraphExec::launch`. With those,
the leg becomes a **key join** — measured on a three-block program here, and on a 40-layer
MoE at 89.4 % coverage in §5.8.1:

```
=== region -> recipe: KEY JOIN via graph name (patched bridge)
recipe_chain_demo.py:63:mme_block  MME 0.038 ms   recipe 57369  dump graph_0000_…_syn_1000000_1000000_4.5.57369
recipe_chain_demo.py:76:dma_block  DMA 0.016 ms   recipe 61189  dump graph_0002_…_syn_1002000_1002000_6.5.61189
recipe_chain_demo.py:68:tpc_block  TPC 0.014 ms   recipe 62418  dump graph_0001_…_syn_1001000_1001000_5.5.62418
verification: PASS — engine attribution matches the code's intent
```

Three compile ids → three source lines → three recipes → three dumps, node types confirming the
intent, and **every `_syn_` event name mapped to exactly one dump**. The match is a **prefix**, not
an equality: the event carries `graph_0000_fused_0_syn_1000000` while the dump is
`…_syn_1000000_1000000_4`, because the recipe name appends a compile-time suffix. The dumper names
each file after the graph, so matching the event name against the *filename* is unambiguous — but
assert uniqueness rather than assuming it.

The same branch also carries a **build fix** the profiler work depends on: `cmake/Targets.cmake`
applied `-pie` to every target including shared libraries, which makes `ld` link a PIE *executable*
(`Scrt1.o: undefined reference to 'main'`, then every legitimately-undefined symbol becomes a hard
error). Guarding it on target type is what makes the backend link at all here. **The earlier belief
that this needed cmake 3.31.6 was wrong** — the flag was the cause, and 4.4.x builds it fine.

**On a stock, unpatched installation this leg is still unavailable**, so `attribute_chain.py` keeps
its fallbacks and still refuses (`AMBIGUOUS`) rather than guessing.

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

**Which API you use depends on whether the model runs in-process or under an engine that owns the
device.** Getting this wrong costs a full model load before it fails.

### 5.6.1 A standalone program (you own the device)

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

### 5.6.2 A vLLM run (the engine owns the device — the common case)

**Do not use the block above.** With vLLM's multiprocess engine the device is owned by the
`EngineCore` child, and any HPU-side call from the parent process dies:

```
RuntimeError: synStatus=8 [Device not found] Device acquire failed.
```

That is not specific to the profiler. It fires for `prof.start()`, for
`torch.hpu.mem_get_info()`, for `torch.hpu.synchronize()` — for anything that tries to acquire the
device in the parent. **Nothing HPU-side may run in the parent process.** Drive the profiler through
vLLM, which runs it in the worker that holds the device:

```python
import os
os.environ["VLLM_TORCH_PROFILER_DIR"] = prof_dir   # where the worker writes the trace
os.environ["VLLM_HPU_PROFILER_ACTIVITY"] = "1"     # device events; both names read by hpu_worker.py
...
llm = LLM(model=..., max_model_len=...)
llm.generate([prompt], sp)                          # warm-up: this is where compilation happens
llm.start_profile("identity")                       # runs in the worker
llm.generate([prompt], sp)                          # steady-state window
llm.stop_profile()
```

Two consequences worth stating:

- **`Torch-Compiled Region` spans still arrive**, because the worker collects CPU activity — so the
  attribution chain is intact. This is the same requirement as §5.4, satisfied by the worker's own
  activity configuration rather than by your `ProfilerActivity` list.
- **The warm-up generate must be outside the window.** With a cold recipe cache the first generate
  compiles the model; profiling it measures compilation.

**Pre-flight every trace** with `devprobe.py` (§1.9): require `kernel` and `hpu_op` in the category
histogram, and — for an attribution trace — require **`Torch-Compiled Region` spans**. A trace that
has device events and no region spans is unattributable; say so and stop.

**And gate the arm before quoting it** (`pending ≥ 2`, `device_busy ≥ 80 %`, §1.8).

---

## 5.7 Four traps that cost runs

**5.7.1 An export bug on this build, and a second one that we suspected and then RETRACTED.**
(a) **MEASURED, reproducible, reported upstream** (`HabanaAI/gaudi-pytorch-bridge` issue 23):
`tensorboard_trace_handler` raises `TypeError: takes 2 positional arguments but 3 were given` —
torch 2.13 calls `export_chrome_trace(path, metadata)` while the Habana shim at
`activity_profiler.py:63` and `:195` defines `(self, path)`. It fires **at profiler exit**, after the
window has completed, so the run is lost. Call the wrapped profiler's method directly:
`prof.profiler.export_chrome_trace(path)`.

**This is not a convenience-handler problem — it blocks profiling a vLLM run outright.** vLLM's own
`start_profile()` path calls the same method in the worker, so a `vllm` profiling run dies with the
same `TypeError` *after* a full model load (measured: 17–23 minutes on a 35 B model). Two ways out:
install the fix in the environment (our fork carries it, §5.2.2), or drive the profiler yourself in a
standalone process (§5.6.1). **The fix has to be in the interpreter that runs the worker**, not only
in a fork you have checked out — a fork does not change what `import` resolves to.

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

**And the same rule bites the arithmetic that *reports* coverage**, which is where we broke it. The
first version of the coverage line computed the unexplained residual by **subtraction**:

```
unexplained = total_union - union(mapped) - union(no recipe)     ->  -0.602 ms
```

A negative bucket. `union(A) + union(B) ≥ union(A ∪ B)`, because two disjoint *event sets* still
overlap in *time* — so subtracting sub-unions from a total is the sum-versus-union error wearing a
different hat. The unexplained bucket has to be **its own union**, computed over the events that
belong to it. The corrected line on the same trace reads 89.4 % mapped / 11.0 % no recipe / 1.4 %
unexplained, and states that the three do not sum to 100 % because they are unions.

**A coverage figure is the one number that makes an attribution table self-auditing** — it is what
tells you a table is missing a third of the step. So it has to be right, and it has to be reported
rather than implied.

**5.7.4 With a multiprocess engine, nothing HPU-side may run in the parent.** Three instances, all
measured, all the same root cause — the `EngineCore` child owns the device and the parent has no HPU
context:

| call in the parent | result |
|---|---|
| `torch.hpu.mem_get_info()` | `RuntimeError: synStatus=8 [Device not found] Device acquire failed.` |
| `torch.hpu.synchronize()` | same |
| `prof.start()` | same — the profiler acquires the device too |

So a profiling harness for a vLLM run cannot be "wrap `llm.generate()` in a `torch.profiler` block".
It must go through `llm.start_profile()` / `llm.stop_profile()` (§5.6.2), and any device statistics
you want must come from the worker's own reporting (or from `hl-smi` outside the process), never from
the parent. This is the same constraint the worker's own `mem_stats` documents: *the parent has no HPU
context, and initialising one there reserves the device and breaks worker init.*

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
measured, is compared against the engine the code was written to use. It passes.

The node types are the independent confirmation: the two GEMMs appear as two `GEMM` nodes, the four
elementwise ops **fuse to one** TPC kernel, and the copy block is one `DmaMemcpy` plus the `add`
broadcast. **Read the dump, not just the engine mix** — a fused kernel and four kernels are the same
engine and a different cost.

### 5.8.1 The real model — 40 layers, 256 experts, 89.4 % attributed

The toy proves the mechanism. This proves it at scale, and it is the run that makes the chain worth
having. **`Qwen/Qwen3.6-35B-A3B-FP8`** on one Gaudi 2: 35 GB of weights, 40 layers, 256 experts with
8 active, profiled through `llm.start_profile()` (§5.6.2) with `TORCH_TRACE` and
`DUMP_POST_GRAPHS` set and a cold recipe cache.

| | |
|---|---|
| region spans | **7,515** |
| compile ids | **33**, of which **23** mapped to a recipe |
| recipes | **23** |
| device union in the window | **121.7 ms** |
| **attributed to a source line** | **89.4 %** |
| in no recipe (eager / outside every recipe) | 11.0 % |
| unexplained (a recipe id in no mapping) | 1.4 % |

and the top of the table, which is the whole point:

| source location | engine union |
|---|---|
| `hpu_fused_moe.py:756:torch_dynamo_resume_in_patched_fused_moe_forward_at_756` | **TPC 27.7 ms, MME 21.2 ms** |
| `qwen3_next.py:549:forward` (the decoder layer) | TPC 13.3 ms, DMA 8.3 ms |
| `hpu_gdn_pytorch.py:841:hpu_chunk_gated_delta_rule` | MME 4.5 ms |

**Read those rows and check them against what the model is.** A 256-expert decode should spend most
of its time in the fused MoE, then in the decoder layer, then in the gated-delta-net chunk kernel —
and it does. That agreement is the validation; a join that mis-attributed would not produce it.

**Three honest caveats, because this is one run:**

- **The mapping was 1:1 in this window** — 23 compile ids, 23 recipes, **no** compile id launching
  more than one recipe and no recipe shared between compile ids. The tool handles the one-to-many
  case (it carries `{compile id → set of recipes}`, and prints the shape of the mapping precisely so
  a 1:1 run is distinguishable from a hidden many-to-many), but **this run did not exercise it**.
  A different batch or context length should make the specialisation diverge; that is the next run.
- **33 compile ids but only 23 mapped.** The other 10 launched no recipe the tool could match — worth
  understanding before the numbers are quoted as complete.
- **Timing is not comparable to anything.** Cold cache, graph dumping and CPU collection all distort
  it. This run is for identity.

### 5.8.2 Two more keys available from the backend metrics

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
| 1 | ~~**A key for region → recipe.**~~ **CLOSED on a patched bridge, OPEN on stock.** Two small edits make it a key join — verified on a three-block program **and on a 40-layer MoE** (§5.8.1). On a stock installation it is still a pairing that fails on a real model. | — **done** on the patched bridge. The stock path stays open, and is why the fallbacks remain in the tool. |
| 1a | **The one-to-many case is untested on a real model.** §5.8.1's run was 1:1 — 23 compile ids, 23 recipes; no compile id launched more than one recipe and no recipe was shared. The tool supports both, but a 1:1 run cannot distinguish *handled* from *hidden*. | A second run at a different batch size or context length, where specialisation should diverge, plus a check that the shape-of-mapping block reports it. |
| 1b | **33 compile ids, 23 mapped** in §5.8.1 — the other 10 launched no recipe the tool matched. | Per-compile-id inspection of those 10: the compile metric (§5.8.2) plus the dump set should say whether they compiled nothing, compiled to a recipe the dumper missed, or ran only eager work. |
| 1c | **The chain has not been run on DeepSeek V4 Flash** — the port this manual is mostly about. Qwen proves the mechanism; it does not prove it on the model that matters here. | The same run shape as §5.8.1, on DSv4, at a batch size where its 121 compile ids and 110 recipes actually diverge. |
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
