# 2. Optimizing small programs and regions

**Status: draft.** Every non-obvious claim carries a basis tag (**MEASURED** / **ASSUMED** /
**INFERRED** / **UNKNOWN**) as defined in the [README](README.md). Numbers are quoted with the
conditions they are valid under.

**This chapter assumes [chapter 1](01-basics.md)** — the engines and their tid windows, the DMA
queues, memory, bandwidth-with-its-size, the cost models and their error signs, markers, and device
pinning. Where this chapter needs one of those facts it restates only what the argument requires and
points at chapter 1 for the full treatment. The overlap, composition and measurement rules below are
this chapter's own.

This chapter is about **one `torch.compile` region on Gaudi 2** — a few to a few hundred ops that
you want to run concurrently. It answers four questions in order:

1. **When does cross-engine work actually overlap, and when is it exactly zero?** (§2.2–§2.4)
2. **How do you choose your regions, and what does a graph break cost?** (§2.5)
3. **When overlap happens, how much time do you save, and what destroys the saving?** (§2.6–§2.7)
4. **How do you measure any of this without fooling yourself?** (§2.8–§2.10)

The short version, so you can hold it while reading: **overlap on Gaudi 2 is not a hardware
property you get for free, and it is not a property of the *ops*. It is a property of the
*submission structure* your compiler produced.** Most "it didn't overlap" outcomes are structural
and fixable; most "it overlapped" claims are measurement artefacts until proven otherwise.

---

## 2.1 The objects, and the one word that matters

You need four nouns. They are nested:

| object | what it is | how you see it |
|---|---|---|
| **region** | one dynamo-traced FX graph from a `torch.compile` region | `Torch-Compiled Region: *` events; `TorchDynamo Cache Lookup` events (one per region execution) |
| **node** | one op in that graph, carrying a **placement** | the graph dump (`DUMP_POST_GRAPHS`, §2.10) or a registered backend pass |
| **recipe** | one compiled binary, produced from one **maximal contiguous run of `hpu_cluster`-placed nodes** | `LaunchRecipe` events; the recipe name in the dump |
| **submission** | one Synapse `Launch` — the thing the device actually schedules | the `Launch`/`enqueueWithExternalEventsExt` record |

**The word that matters is *submission*.** Not region. Not recipe. Not op. The Synapse scheduler
co-schedules **independent nodes inside one submission**; it does nothing across submissions.
Every rule in §2.2–§2.5 is a consequence of that sentence.

> **A submission has a *constructor* on this stack, not only a reader.** A hand-built Synapse C-API
> graph — nodes created with `synNodeCreate`, edges expressed as **shared tensor handles** (there is
> no explicit edge call; that is the normal Synapse idiom) — compiles to **one recipe** and dispatches
> with **one `synLaunch`**. A 4-node dependent DAG (`relu` → `relu` → `concat` → `gemm`) compiles and
> runs as one submission with every stage verified non-zero. **MEASURED.**
>
> **One lever that does *not* exist.** `synNodeSetUserProgrammability` **does** set a node's execution
> rank, and the GC honours it by emitting `blocking_nodes` — but it is **coupled to a dependency**.
> Applied to *independent* work it serialises the pair: forcing an independent DMA node to rank 0 made
> the span **worse in 3 of 3 rounds** (min-of-3 **0.798 → 0.890 ms**, +11 %) and landed within 11 % of
> the pure serial prediction `T(dma) + T(mme)`. **You cannot reorder independent work in this API
> without creating an edge to it** — so "reorder without changing the dependency structure" is not
> expressible, and any experiment that needs it must say so. **MEASURED.** *(Source:
> `Archive/by-date/2026-10-13/recipe-capability/`; `synNodeDependencySet` is accepted and **inert** in
> this configuration — an exported symbol returning `synSuccess` is not evidence the GC honours it.)*

> **Two counts that are easy to confuse, and only one of them is the submission count.**
> `LaunchRecipe` counts **recipe launches**, not device submissions. An `eager`-placed node runs
> *outside* every recipe and emits **no** `LaunchRecipe` — so a body consisting only of one eager
> node reports `LaunchRecipe/step = 0` while the op runs every step. **MEASURED** (a one-custom-op
> body: `LaunchRecipe = 0`, yet 9.6 kernel events/step). When you need the submission count, read
> `Launch`/`enqueueWithExternalEventsExt`, not `LaunchRecipe`.

---

## 2.2 The four-condition overlap law

Define the measurement first, because the law is stated in terms of it:

```
CROSS = Σ(unions of each engine family) − union(all families)      [ms/step]
```

`CROSS` is the literal time the engine families' execution windows intersect. `CROSS = 0` means
disjoint; `CROSS > 0` is real concurrency. Build each `union` by **merging intervals**, never by
summing durations (§2.9, rule 1). Family membership is by device kernel name: MME = `GEMM`/`gemm`,
DMA = `DmaTranspose`/`DmaMemcpy`/`transpose_dma`/…, TPC = everything else.

> ### The law
> **`CROSS > 0` iff ALL FOUR hold:**
> 1. **ONE Synapse submission** for the whole group (`Launch`/step == 1);
> 2. the groups are on **different engines**;
> 3. there is **no dataflow edge** between the groups — **and, for a same-engine elementwise
>    edge, no fusion** (see the split below);
> 4. the groups' **durations are comparable**.
>
> **Miss any one and `CROSS` is exactly `0.0000`.**

**MEASURED** (all four conditions were separated by matched arms). The four conditions are not
independent advice — they are a conjunction.

**Condition 3 is two conditions wearing one name**, and which one you have depends only on whether
the edge crosses an engine boundary. **MEASURED** (both forms, same harness, one recipe each).

| edge | what it lowers to | how to see it | what it costs |
|---|---|---|---|
| **cross-engine** | a **monitor on the producer engine's sync object** — the consumer's queue entry waits until that engine's counter reaches a threshold | the compiler's sync-scheme log prints, on the consumer, `Dependencies (monitors): Wait on type: <ENGINE> to reach (>=) sig-val: N`; an independent node prints `Dependencies (monitors): none` | **a runtime dependency** |
| **same-engine, fusable elementwise** | **fusion into ONE kernel** | the graph dump shows a single `fused_kernel_<hash>_<n>_<dtype>` node where the unfused body had two | **not a wait at all** — the win is a saved memory pass |

**The cross-engine form is structural, not an inference.** The `DUMP_POST_GRAPHS` node sets are
**identical** between the edge and no-edge arms (`dmatpc`: 1 `DmaMemcpy` + 1 `relu_fwd_f32` each) —
no fusion, no placement change, one recipe, one `synLaunch`. Only the monitor differs.

**The same-engine form, quantified.** Two `relu_fwd_f32` nodes joined by an edge compile to **one**
node (`fusedTPCNode_0_0`, `type=fused_kernel_0xF54BCFA8_1_f32`, `input=[pIn]`, `output=[pOut,
qOut]`) and measure **395.9 µs against 520.6 µs** at R=8192 for the same two nodes with the edge
removed. The fused kernel does read + 2 writes = **3** memory passes where the unfused pair does
2 reads + 2 writes = **4** — and 1.5× a single `relu` is **369.9 µs** predicted against **395.9**
observed. Nothing synchronises; the compiler merged the kernels and deleted a read. `memcpy_dma`
does **not** fuse this way (two chained copies stay two nodes) — the fuser is TPC-only.

**Why the split matters for a port.** A cross-engine edge is fixed by removing the dependency or by
accepting a wait. A same-engine elementwise edge is fixed by *code generation*: it is usually a
**win**, and "avoid the edge" is the wrong move.

**One refinement, because it changes how you use the list.** Conditions 1–3 have a **total**
failure mode: violating any of them gives exactly `0.0000`, not a reduced value, and the arms
separate each one cleanly. Condition 4 is **different in kind** — it is a **detectability**
condition, not an existence condition. A 5 %-duration group *does* hide, but the `CROSS` it
produces is small enough to be indistinguishable from a violation of 1–3; the 4096² pair in the
table below violates it and still reports `CROSS = 0.0248` (independent) vs `0.0159` (dependent),
i.e. **small and non-discriminating rather than zero**. So: read 1–3 as "if violated, there is no
overlap"; read 4 as "if violated, you have not measured anything either way". **MEASURED**, and it
is the reason condition 4 is stated as a requirement on the *experiment* as well as on the code.

**Label the observable, though, because condition 4 is dual.** The reading above is about
**`CROSS`**. For **wall-clock absorption** condition 4 is an **existence** condition, not a
detectability one: at TPC:MME = **1:12** the saving is **`0 %` to within the instrument's ±5 µs** —
indistinguishable from a violation of 1–3 — and it only plateaus at **≈ 45 %** once the TPC is about
a third of the MME.
Both readings hold at once (a pair can have intersecting windows and still save nothing, if both
legs are HBM-bound), which is exactly why they need separate labels. **MEASURED** — see the
2026-10-13 UPDATE below.

### The evidence

Every arm whose `CROSS` rises above the harness noise floor has `Launch`/step == 1. Every arm with
`Launch`/step ≥ 2 is exactly zero. The project's own tally from the pooled arm corpus is **7 arms
above `CROSS = 0.017` ms/step, every one `Launch`/step == 1**, and **22 of 22 multi-submission arms
exactly `0.0000`** (**MEASURED**). *(Counts depend on which harness tables are pooled — the
single-shot and model-op arms add more entries to the numerator, never a counterexample. The
pattern, not the count, is the result.)*

The arms below are the ones that make the pattern legible. All are graph-compiled, all asserted
their own precondition (§2.8), and all use **duration-matched** ops (§2.2 condition 4).

| arm | structure | `CROSS` (ms/step) | `Launch`/step | one submission? | verdict |
|---|---|---:|---:|:---:|---|
| TPC × MME, 2 ops | one region, independent | **0.2019** | 1.00 | yes | overlap — MME **entirely** hidden (kernel channel) |
| TPC × DMA, 2 ops | one region, independent | **0.2496** | 1.00 | yes | overlap — TPC **entirely** hidden |
| TPC × MME, 4 ops | one region, independent | **0.4021** | 1.00 | yes | overlap, and it **scales** |
| TPC × MME, 8 ops | one region, independent | **0.8080** | 1.00 | yes | overlap, whole MME family hidden (kernel channel) |
| TPC × TPC, 2/4 ops | one region, independent | 0.0000 | 1.00 | yes | **same engine** ⇒ serialised |
| MME × MME, 2/8 ops | one region, independent | 0.0000 | 1.00 | yes | **same engine** ⇒ serialised |
| TPC × MME | **uncompiled dispatch** (one host call per op) | 0.0000 | 2.00 | no | not one submission |
| TPC × MME | one region, **two recipes** | 0.0000 | 2.00 | no | not one submission |
| TPC × MME | **two HPU streams**, 36 ops/step | 0.0000 | 36.00 | no | not one submission |
| TPC × MME | one region, **dataflow edge** | 0.0000 | 1.00 | yes | **edge** ⇒ serialised |

*(all **MEASURED**)*

Read the TPC×MME rows across: in every overlapping arm `union(all) == union(TPC)` **exactly over
the kernel channel** (`TPC`/`MME`/`DMAk`) — the *whole* MME family disappears under the TPC, at 2, 4
and 8 ops. *(2026-10-13: putting the `hpu_op` DMA channel back — the MME's staging `DmaMemcpy`,
which `t2_census.py:99`'s `cat == "kernel"` filter drops — leaves the equality off by a constant
**+1.6–1.8 µs/step (0.2–0.8 %)**, one ~1–2 µs sliver per step, and leaves `ov`(TPC,MME) at exactly
`1.000`. Re-reduced: `Archive/by-date/2026-10-13/eocensus-t2-rereduce/`.)* The saving therefore **grows with
depth** (0.2019 → 0.4021 → 0.8080) instead of saturating at one op's worth, which is what you would
see if the scheduler were only hiding a fixed slice.

> ### UPDATE 2026-10-12 — the headline, re-measured on an independent harness, together with its negatives
>
> **The primary result, and the strongest statement in this chapter. MEASURED.**
>
> **In a single submission with independent branches, `union(all) == union(TPC)` EXACTLY over the
> kernel channel (`TPC`/`MME`/`DMAk`) — the entire MME family hides behind TPC, with `ov = 1.000`
> across 13 arms.** Every size, every TPC load, every orchestration that keeps one submission.
> **This is not partial overlap: the MME's `GEMM` contributes *nothing* to the union.** *(Scope added
> 2026-10-13 — the *family* is **99.2–99.8 %** hidden, not 100 %: its staging `DmaMemcpy` rides the
> **`hpu_op`** channel and adds a constant **+1.6–1.8 µs/step**; `ov` stays exactly `1.000`.
> `Archive/by-date/2026-10-13/eocensus-t2-rereduce/`.)* (`ov = (t_a + t_b − t_ab) / min(t_a, t_b)`,
> on **device unions**; the
> harness that produced it is 33 arm-runs, each a single `torch.compile(fullgraph=True,
> dynamic=False)` region — or a deliberately uncompiled control — replayed 20 steps after a 6-step
> warm-up at the correct shapes.)
>
> **The instrument was validated before the result was believed.** The reduction reproduces the
> archived `t2_graph_overlap` numbers **exactly** on the archived traces — `t2_02_tpc_mme_indep`
> 0.2659 / 0.2019 / CROSS 0.2019, `t2_08_tpc_mme_indep` 1.0387 / 0.8080 / 0.8080,
> `t2_02_tpc_dma_indep` 0.2496 / DMA 0.6730 / 0.2496, `t2_02_mme_mme_indep` 0.2697 / 0.0000 — and it
> does so with a *different* TPC op form from the archived harness (`torch.mul(x, s, out=o)` with a
> 0-dim tensor multiplier, instead of `torch.mul(x, 2.0)`). **And the headline reproduced within
> 1.1 % on a second independent run** — a separate process ~40 minutes later on the same module:
> `tri_tpc_mme_tx` CROSS 0.3831 → 0.3824.
>
> **The negatives — and each is now a *tested null*, not an assumed one.** Every arm below asserted
> and **passed** its own precondition (§2.8); none of them is an empty-queue arm.
>
> * **Two HPU streams, device-bound: `ov = 0.0000` exactly, both runs.** The precondition is asserted,
>   not assumed — **18.5 ops outstanding at `H_end`, 83.5 % device busy** (run 1) and **55.9 / 97.0 %**
>   (run 2) — and `union(all) = TPC + MME` to four decimals in both (the arm carries **zero**
>   `hpu_op` DMA events, so the channel scope above does not touch it). **A sentence of history, because
>   it is the reason this arm had to be re-run:** the E5 audit's two-stream arm (`07_streams_indep`)
>   was a **0/40-ops-in-flight null test** (host-bound, 10.9 % busy), which is what left the audit's
>   cases (d)/(e) open. **That null is not the arm §2.12 quotes, though** — §2.12's arm is
>   **`t1_02_streams_tpc_mme`** (36 ops/step, **119.1 ops outstanding, 97.8 % busy, both queues
>   provably loaded**), which was **device-bound** all along. **This one is a real negative — on a
>   third, independently device-bound stream arm.**
> * **Single vs double buffering: no effect on the overlap fraction.** In-place (`torch.mul(x, s,
>   out=x)`, so step *k+1* reads what step *k* wrote) against two independent `(in, out)` pairs:
>   `ov`(TPC,MME) = **1.000 both ways**, `ov`(TPC,DMAh) = **0.978 both**. The extra TPC op in the
>   double-buffered arm adds its full 0.185 ms to the union — same engine, so additive, exactly as
>   §2.6's rule predicts — and changes nothing else. **This is the first artifact anyone has produced
>   on this question**; §2.12 previously had none.
> * **Emission order: no effect.** Transpose emitted first vs TPC-first: CROSS **0.3831 vs 0.3826**.
>   The scheduler is not order-sensitive here, consistent with the archived finding that the scheduler
>   knobs are inert.
> * **A dataflow edge: the three TPC-involving pairs collapse to `0.000`; the rest survive.**
>   *(Corrected 2026-10-12 — this bullet previously read "**every** pairwise intersection collapses to
>   `0.000`" and named the residual `MME∥DMAh`. Both were wrong.)* CROSS falls 0.3831 → 0.0939, and that
>   residual is exactly **MME∥DMAk** — the arm's own record (`tri_tx_dep`) reads
>   `pairs = {TPC|MME 0.000, TPC|DMAk 0.000, TPC|DMAh 0.000, **MME|DMAk 0.0939**, MME|DMAh 0.0177,
>   DMAk|DMAh 0.0021}`, and `CROSS` here is `sum_kernel − union_kernel` over the **kernel** channels
>   only (`TPC`, `MME`, `DMAk`; `eocensus.py:222`), so `DMAh` — an `hpu_op` channel — is outside the
>   sum and cannot be its residual. §2.2 condition 3, reproduced with three engines instead of two.
>
> *(Source: `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §2, §3 and §6.
> Of the 33 arm-runs, 30 passed the precondition and
> **3 are recorded INCONCLUSIVE** — `solo_mme2048` and `sw_t0_m2048` (queue too shallow at the
> smallest matmul) and `ctl_sync_mid` (50.7 % busy, 0.7 ops outstanding; the sync drains the queue, so
> that arm cannot test anything, which is a design flaw and not a negative). Physical module 3.)*

### Condition 4 is not decoration — it is what makes the experiment a test

An overlap experiment in which one group is 5 % of the other's duration cannot detect anything:
even perfect overlap saves 5 %. This is the single most common way to run a *correct-looking*
experiment and get a **false zero**. Two matched pairs make the point (**MEASURED**):

| pair | TPC:MME ratio | independent `CROSS` | dependent `CROSS` | discriminating? |
|---|---|---:|---:|---|
| 4096² pair | ≈1:8 | 0.0248 | 0.0159 | **no** — both ≈ 62–66 % of a 40 µs TPC hidden |
| size-matched pair | ≈1:2 | **0.2715 ms/call** | **0.0000 ms/call** | **yes** |

Both pairs hold shape, ops and the single-submission structure fixed; the *only* difference within
each pair is whether the MME's first operand is the TPC's output (an edge) or an independent tensor
(no edge). The 4096² pair is reported for completeness and **must not be read as evidence about
dependencies**. Size-match your groups before you conclude anything.

### Condition 3, isolated

The size-matched pair above is the cleanest evidence for the dataflow condition, because
`torch.hpu.synchronize()` is called before **every** recipe call, so exactly one recipe is ever in
flight and cross-step pipelining is impossible. Independent body: `union(all) == union(MME)` — the
TPC is *entirely* hidden (the arm carries **zero** `hpu_op` DMA events, so the channel scope above
does not touch it). Dependent body: `union(all) == Σ(families)` **exactly over the kernel channels**
— and even with the `hpu_op` channel counted, that body's DMA union (**0.2457 ms/call**) adds only
**+7.9 µs/call** to the union, i.e. **97 %** of it was already inside. *(This arm is normalised per
**call**, not per step: 12 calls.)* Same ops, same
shapes, same submission count, one edge. **MEASURED.**

### What a dependency is enforced *on* — a byte range, not a node

Condition 3 says "no dataflow edge". It is worth knowing exactly what the compiler does when there
*is* one, because the granularity is **finer than the graph**.

**The mechanism, from the compiler's own sync-scheme log.** A cross-engine edge lowers to a **counted
semaphore** — a monotonic per-logical-engine counter (a "virtual sync object bank"; Gaudi 2 has four
logical engine queues, MME / TPC / DMA / ROT). The producer's activation **increments** its engine's
bank; the consumer's queue entry carries **at most one monitor per engine** — a 4-bit engine bitmap
plus a **delta threshold** relative to the last dependency that node's queue recorded — and is
released when the bank reaches it. On the runtime side it is a queue-manager fence + monitor. The
encoding is asymmetric: the TPC bank advances **32** per activation, MME/DMA/ROT advance **1**.
**MEASURED.**

**The dependency is computed from byte ranges, so it is per (consumer, *overlapping byte-range*) and
resolved to a producer activation** — not per node, and not per tensor. The consequence to carry is
that **partial consumption is real, and the compiler arranges it.** In one observed recipe a TPC node
sliced three ways is consumed by two consumers at two different granularities:

```
t0_bundle_0/op_0   (TPC)   pipeline 0/1/2 -> emits sig-vals 32 / 33 / 34
  .../memcpy/0     (DMA)   waits on TPC >= 32, then >= 33, then >= 34   <- per slice
  mm0_bundle_0/op_2 (MME)  waits on TPC >= 34                           <- the whole node
```

— the DMA copy pipelines against the producer's slices while the GEMM waits for all of it. **Same
producer, same recipe, two granularities, chosen by the ROI overlap. MEASURED.**

**And same-engine ordering costs nothing.** Two slices of one TPC node both read
`Dependencies (monitors): none`; within-engine order is **queue order**, and every monitor observed
was cross-engine. So the asymmetry that matters is **same-engine (free) vs cross-engine (a monitor)**,
not one engine versus another. **MEASURED.** *(Source:
`Archive/by-date/2026-10-13/dependency-semantics/` — `REPORT.md` §2, §4, §5.2.)*

**What an edge costs in wall clock: a measured negative, with a bound.** Across the **11 cross-engine
arm-pairs** of a 13-row matrix (the two `TPC→TPC` rows are the fusion case and are excluded; each arm
is one graph → one recipe → one `synLaunch`, `min` over 30–40 reps, physical module 3), every
`T(dep) − T(indep)` lies within **±8.0 µs** with **mixed signs** — the two cleanest large-node pairs
being **+3.8 µs on 540 µs (0.7 %)** and **+8.0 µs on 533 µs (1.5 %)**. The timer's own floor is
**27.5–33.3 µs** and its resolution on a difference of two arms is **≈ ±5 µs**, so the per-edge
handoff is **< ~8 µs and consistent with 0 — below this instrument's resolution.** And the
zero-absorption regime is visible in the ratio, not merely hidden under the floor: the *independent*
pair already measures **1.04–1.17× the sum** of its two solo times (four pairs — the two TPC↔MME rows
at ≈1:12 give **112 %** and **117 %**; the two DMA↔TPC rows at R=8192, matched within **1.11×**, give
**104 %** and **105 %**). **MEASURED negative, with a bound.** The practical rule: **do not price a
dependency in wall clock; price it in execution windows** (`CROSS`, above).

### How to check each condition

| condition | check |
|---|---|
| 1. one submission | `Launch`/step == 1 **and** all your ops carry the **same** `recipeHandle`. `LaunchRecipe` alone is not enough (§2.1). |
| 2. cross-engine | read the per-node `engine` field in the graph dump (§2.10). Do **not** infer the engine from an op name. |
| 3. no dataflow edge (and, same-engine, no fusion) | read the FX graph / JIT IR and confirm the nodes' operand sets are disjoint. Note that an edge can exist *only* through buffer reuse or aliasing — check for that too. Then read the compiler's **sync-scheme log** (`LOG_LEVEL_ALL=1`, category `SYNC_SCHEME`): a cross-engine edge shows up as `Dependencies (monitors): Wait on type: <ENGINE> to reach (>=) sig-val: N` on the consumer, and an independent node shows `none`. This is the structural check — it needs no profiler. Same-engine, look for a single `fused_kernel_*` node instead. |
| 4. comparable durations | measure each group **solo** in the same context (§2.7) and confirm the ratio is within ~2×. |

### A caution about "`union(all) == union(TPC)`"

That equality is *also* consistent with cross-step pipelining (step *k*'s MME hiding under step
*k+1*'s TPC). The interval structure rules it out — in the overlapping arm there are 20 merged TPC
blocks and 20 merged MME blocks and **each MME block *k* nests inside TPC block *k***, indices
0…19, **zero straddles**; under cross-step pipelining MME block *k* would straddle TPC blocks *k*
and *k+1* (**MEASURED**). Any future claim of intra-graph overlap should carry either the interval
nesting or an isolated single-shot run.

> ### UPDATE 2026-10-13 — the ratio is the variable, and an UNRESOLVED residual
>
> **MEASURED.** The two C-API rows are hand-built single-recipe graphs (one `synLaunch` each,
> physical module 3, `min` over 30 reps); the two `torch.compile` rows are single-recipe
> `torch.compile` arms on the same module:
>
> | TPC:MME | harness | absorbed |
> |---|---|---|
> | **1:12** | C-API | **0 %** *(to within ±5 µs)* |
> | ≈1:3 | C-API | **≈ 45 %** (plateau) |
> | ≈1:1.5 | `torch.compile` | **53–56 %** |
> | ≈1:1.1 | `torch.compile` | **90 %** |
>
> **The node count is not the lever.** Holding the TPC work fixed at 64 MB and the `gemm` fixed at
> 2048³, the total is **flat across K = 1/2/4/8** — **228.0 / 229.3 / 229.1 / 231.3 µs**.
>
> **Read the 0 % against the instrument.** This timer's floor (a one-node 8×8-`relu` recipe, `min`
> over 30–40 reps) is **27.5–33.3 µs** and its resolution on a *difference of two arms* is
> **≈ ±5 µs** — so the 1:12 row is **0 % to within ±5 µs**, not an exact zero.
>
> **⚠ UNRESOLVED, and this is the bound: no wall-clock arm reproduces the exact
> `union(all) == union(TPC)`.** The best absorption measured is **90 %** (torch) and **46 %**
> (C-API) — a ~10 % gap remains on the best arm, and the event-filter question stays open.
> **This block does not close it.**
>
> *(Source: `Archive/by-date/2026-10-13/dependency-semantics/`.)*

---

## 2.3 "Customness" is **not** the discriminator — and this trap is worth a whole section

A rule that circulated in our own project was: *"a custom op on either side of the pair forces
`CROSS = 0`."* It is **false**, and it is false in the most instructive way.

The experiment that killed it held **customness fixed** and changed **only the submission count**
(**MEASURED**). All three arms are one `torch.compile(fullgraph=True)` region per step, with the
same independent MME and solo durations matched within 1.18×:

| arm | TPC op | custom? | `CROSS` | `Launch`/step | one submission? |
|---|---|---:|---:|---:|:---:|
| **A** | `custom_op::fp4_dequant` (**the model's own op**) | yes | **0.2217** | **1.00** | **yes** |
| **B** | `kchain::alu` | yes | **0.0000** | **2.00** | no |
| **C** | `torch.add` | no | **0.4179** | **1.00** | **yes** |

**A and B are both custom ops and differ in exactly one measured respect — `Launch`/step 1.00 vs
2.00 — and the outcome flips.** In A and C, `union(all) == union(TPC)` exactly over the kernel
channel: the whole MME
family is hidden. Nesting confirms it: in A and C, MME block *k* nests inside TPC block *k* for all
20 with zero straddles; in B, all 20 **straddle**.

The mechanism is visible in the recipe artifact, not inferred from a launch count:

```
recipe graph_0000 (arm A): 11 nodes  TPC 1  MME 4  DMA 4  None 2   types: DmaMemcpy, GEMM, fp4_dequant_fwd_gaudi2
recipe graph_0001 (arm B): 10 nodes  TPC 0  MME 4  DMA 4  None 2   types: DmaMemcpy, GEMM
```

**In arm A the custom `fp4_dequant` node is inside the same recipe as the `GEMM`; in arm B the
custom `kchain_alu` node is not in the recipe at all — TPC count 0 — it runs as 20 separate
single-node recipes outside the compiled one.** Both ops use the *same* registration API, so the
difference is not registration style: it is **whether the backend lowers the op into the graph**.
Arm B's 21 distinct `recipeHandle`s are the tell. **MEASURED.**

> ### Why the trap is seductive, and what to do instead
> In a corpus that contains **only** `kchain`-style custom ops, "custom ⇒ multi-submission" holds
> **13/13**. Our own review concluded the rule from exactly that corpus. It broke the moment a
> *different* custom op (`fp4_dequant`, which lowers into the graph) was tested. **"Custom op" was
> a proxy for the submission count that happened to hold for one op family.**
>
> **Use the submission count, not the op's namespace.** If you find yourself writing a rule of the
> form "op X behaves differently", you have probably found a proxy. Find the mechanism.

---

## 2.4 Recipe boundaries are **placement**, not op identity

The previous section says the submission count is what matters. This section says what sets it.

> **There is one Synapse recipe per maximal contiguous run of `hpu_cluster`-placed nodes in the
> compiled region's FX graph.** A node that the HPU **shared-layer op validator** rejects is placed
> `eager`; an eager node is not clusterable and **splits the run** into two recipes.
>
> **Nothing else tested does this** — not op count, not dtype, not a Python-int guard, not a D2D
> copy, not `mark_step`.

**MEASURED**, verified **24/24** arms against the live FX graph, with the profiler trace and the
graph inspected independently.

| construct | placement of the inserted node | recipes/step | why |
|---|---|---:|---|
| 2 / 4 / 8 / 16 / 32 **native** adds in one body | `hpu_cluster` | **1** every time | growing a body with clusterable ops does not split |
| custom `kchain::alu` mid-body | **eager** | 1 → **2** | fails `shared_layer_validation` — *but read the correction below: that validator is only **reached** at all because the op is not in the `custom_op` namespace* |
| two custom ops, separated | eager ×2 | 1 → **3** | two splits |
| two custom ops, **adjacent** | eager ×2 | 1 → **2** | one eager *run* adds one split |
| custom op at the very **start or end** | eager | **1** (unchanged) | the eager node lies outside every clusterable run |
| custom op **alone** | eager | 1 → **0** | there is no clusterable run at all |
| `torch._dynamo.graph_break()` | — | 1 → **2** | two *regions*, one run each |
| `x.cpu().to('hpu')` (**native** D2H+H2D) | **eager** (`aten._to_copy` ×2) | 1 → **2** | non-HPU input ⇒ eager |
| `int(x.sum().item())` (**native** host read) | — | 1 → **2** | dynamo graph break ⇒ two regions |
| Python-int guard (extra int argument) | unchanged | **1** | guards are not nodes |
| `x * 3`, `x.clone()` (native) | `hpu_cluster` | **1** | a D2D copy **is** clusterable |
| `torch.hpu.mark_step()` | — | **does not compile** | lazy-mode-only API; raises inside a traced region |

> **One row in that table needs a warning, because it is easy to misread.** *"Python-int guard:
> recipes/step 1, unchanged"* means an int guard does **not** split a single compilation into more
> recipes — a guard is not a node, so the partitioner never sees it. It does **not** mean ints are
> free: an int used in traced code becomes part of the guard set and produces a **separate compiled
> graph per distinct value** ([Chapter 0](00-tldr.md), fact 2). Those are two different counts —
> recipes *within* one compiled graph, versus compiled graphs *per* region — and §2.5 is about the
> second one.

### The rule is "cannot be clustered", not "is custom"

Read the last few rows carefully. **`x.cpu().to('hpu')` is a native PyTorch construct and it splits
the recipe exactly like a custom op**, because `aten._to_copy` with a non-HPU source also fails the
shared-layer validation. Conversely a *custom* op (`fp4_dequant`, §2.3) lowers into the graph and
does **not** split.

**So the boundary condition is not the op's origin. It is whether the op can be placed in an HPU
cluster.** The placement decision is made once per node:

```
aten.add.Tensor     -> hpu_cluster   shared_layer_validation -> True
kchain.alu.default  -> eager         shared_layer_validation -> False   <-- the boundary
```

Both ops have `default_fallback=False` and both are absent from the hpu fallback list, so the
discriminator is the **shared-layer op-support table** — not the fallback list, and not the custom
namespace. **MEASURED** (read out of the backend, then reproduced from a registered optimization
pass that prints the placement and the reason per node).

> ### ⚠ CORRECTION 2026-10-12 — that discriminator is true for `aten` ops and **FALSE** for `custom_op`-namespace custom ops
>
> **The claim as it was made (kept above, not deleted):** *"the discriminator is the **shared-layer
> op-support table** — not the fallback list, and not the custom namespace."*
>
> **As stated it is wrong, and wrong in the direction that matters: it does not decide placement at
> all for a custom op.** **MEASURED**, in **one process holding both custom extensions**, by a
> registered `PRE_PARTITIONER` pass reading `node.meta["placement"]` per node:
>
> | node | namespace | `check_for_default_op_support` | `shared_layer_validation` | **placement** |
> |---|---|---:|---:|---|
> | `custom_op.fp4_dequant.default` | **`custom_op`** | **True** ("Graph support for user's CustomOp") | **False** | **`hpu_cluster`** |
> | `kchain.alu.default` | `kchain` | False | False | **`eager`** |
> | `aten.add.Tensor` | `aten` | False | **True** | `hpu_cluster` |
>
> **`custom_op::fp4_dequant` is clustered *despite* `shared_layer_validation = False`.** The
> validator is never consulted for it, because the backend **short-circuits on the namespace first**
> (`shared_layer.py:273`):
>
> ```python
> # Enable torch.compile for user's CustomOp API
> if hasattr(node.target, "namespace") and node.target.namespace == "custom_op":
>     return True, "Graph support for user's CustomOp"
> ```
>
> `check_for_default_op_support` returning `True` makes `is_eager_fallback_required` exit `False`
> **before** the `shared_layer_validation` call is reached — so the node is placed `hpu_cluster`,
> joins the clusterable run, and lands in the same Synapse submission as the GEMM.
>
> **So read the split precisely:**
>
> * **for an `aten` op** the original claim holds — `shared_layer_validation` decides
>   (`aten.add` → True → clustered);
> * **for an op in the `custom_op` namespace** it is false — placement is `hpu_cluster`
>   **regardless of what the validator would say**.
>
> The two nodes above with `shared_layer_validation = False` therefore land in **different** places,
> and the stated discriminator cannot separate them. **The real rule is the namespace
> short-circuit.** `kchain::alu` is `eager` because its namespace is not `custom_op` — nothing about
> its kernel makes it eager. *(Source: `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §5.1. Run on
> physical module 3.)*

> ### THE ENGINEERING RECIPE — how to get a hand-written `tpc-clang` kernel co-scheduled
>
> This is actionable and it is not obvious, so it is written as a procedure. To make a new
> `tpc-clang` kernel **co-schedule with MME/DMA in ONE submission**:
>
> 1. **Register it in the EXISTING `custom_op` library — do not create a new namespace.** Add its
>    `m.def` / `m.impl` lines to the existing `TORCH_LIBRARY(custom_op, …)` block and to the
>    `TORCH_LIBRARY_IMPL(custom_op, HPU/Meta, …)` blocks (in this project,
>    `src/custom_ops/fp4_dequant/hpu_custom_fp4_dequant.cpp`), rather than shipping a new
>    extension under a new namespace.
>    * **A defensive new namespace is exactly what costs you the overlap.** It is the natural thing
>      to do — it is what `kchain` did, and the reason recorded in its own header is mechanical:
>      `TORCH_LIBRARY` may be called only once per process per namespace, so a second extension
>      cannot re-declare `custom_op`, and loading two would be a hard `c10::Error`. **The namespace
>      short-circuit above is a better justification for the one-library design than that reason
>      was:** one extension, one `custom_op` namespace, **all** of its ops clusterable — whereas a
>      separate namespace buys isolation and **pays for it with the overlap**. *(That the `kchain`
>      namespace costs `kchain::alu` the overlap is **MEASURED**; that it does the same for the
>      other four `kchain` ops — `fp4_base`, `fp4_arith2`, `fp4_lutptr`, `ident` — follows from the
>      same measured namespace rule and is **INFERRED**, not separately measured.)*
> 2. **Keep the body a DAG with no dataflow edge to the MME/DMA branch.** A **cross-engine** edge
>    lowers to a monitor on the producer engine, and it drops every **TPC-involving** pairwise
>    intersection to exactly `0.000` (measured above with three engines);
>    the `MME∥DMAk` pair survives at 0.0939, which is what the residual `CROSS` becomes. *(A
>    **same-engine** TPC→TPC elementwise edge is the opposite case — the compiler fuses the two
>    nodes into one kernel; see §2.2.)*
> 3. **Duration-match** the branches to within ~2× (condition 4).
> 4. **Verify; do not assume.** Register a `PRE_PARTITIONER` pass and print
>    `node.meta["placement"]` — `hpu_cluster` means it is in the recipe. The whole check is ~100
>    lines (`Archive/by-date/2026-10-12/engine-overlap/eo_placement.py`). **Do not infer from the op's
>    name or its namespace** (§2.9 rule 3): read the placement. This is the check that produced the table
>    above; without it, the superseded claim in this section would still be standing.
> 5. **Confirm at the trace level** that `Launch`/step == 1 and the ops share one `recipeHandle` —
>    and read the `Launch` count, not the handle count (§2.1).

### The trap: `LaunchRecipe` is not a submission count

Restating §2.1 because it is load-bearing here. An `eager`-placed node runs **outside** the recipe
and emits **no** `LaunchRecipe`. In the one-custom-op-alone arm the recipe count is **0** while the
op executes every step. **A "recipes/step" figure must always be read as *recipe launches*.**
Likewise a `recipeHandle` **count** is not a recipe count — handles in a one-recipe arm ranged from
1 to 13 across runs as the recipe cache re-missed — while the **per-step recipe count was stable
20/20 identical**. **MEASURED.** Use the launch count; treat the handle count as diagnostic only.

### What this means for a large model

**INFERRED, clearly labelled.** The engine's per-region recipe multiplicity is *this* rule. In one
archived trace a region launches **5 recipes per execution** while another launches 1; by this rule
frame 23's region graph must contain **5 clusterable runs** (i.e. 4 eager-placed nodes). **Not
verified** — it needs one engine run with the placement pass registered, which turns "1/2/5 recipes
per region" from a trace count into a structural fact. *(The recipe dump in this manual's source
project, `Archive/by-date/2026-10-11/recipe_boundary/`, is the harness that does this for any
`torch.compile` region.)*

### What is **UNKNOWN**

* **The exact content of the shared-layer op-support table.** We measured *that* a node's placement
  is decided by `shared_layer_validation`, and we measured the outcome per construct; we did not
  enumerate the table. **UNKNOWN.** What would settle it: read the table out of the backend, or run
  the placement pass over a candidate op set.
* **Whether the eager-run-vs-split accounting holds for more than two eager nodes in a row**, and
  for an eager node adjacent to a region boundary. **UNKNOWN** (tested only for the cases in the
  table).

---

## 2.5 Region strategy: breaks, coalescing, and the specialization count

**Where this comes from.** [Chapter 0](00-tldr.md) states two of its facts about *region strategy*: a
graph break costs twice, and fewer/larger regions is the direction — but verify the specialization
count actually fell. This section develops both, adds the metric you verify with, and says plainly
where the evidence runs out.

**Hold one thing from §2.2 before reading on.** The currency is the **submission**, not the region. A
change that reduces the region count without reducing the submission count has bought host work and
nothing else (§2.1).

### 2.5.1 A graph break costs in two currencies — and only one of them is reliable

A graph break — an explicit `torch._dynamo.graph_break()`, a host read of a device value, or anything
else that forces one — splits the compiled program into **more regions**. Two consequences follow.

**(a) It can destroy overlap you already had.** This is the large, reliable cost, and it follows
directly from §2.2. Co-scheduling happens *inside one submission*. Ops that were independent and
co-resident in one region are now in two regions, hence two submissions, and the scheduler does not
co-schedule across submissions at all. The control arm in §2.2 is exactly this case: the same two ops,
one region each instead of one region together, `Launch`/step **2.00**, **`CROSS = 0.0000`**.
**MEASURED.**

**(b) It adds host-side launch and synchronization traffic.** Each region is submitted separately, each
submission is a blocking host-side handshake, and each region also brings its own guard evaluation.
Structurally this is certain: a break takes a body from 1 region / 1 recipe / 1 launch to 2 regions /
2 recipes / 2 launches (**MEASURED** — the break arm in §2.4's table).

**How much that second charge is worth is a separate question, and the answer is "less than you would
guess".** In one measurement the *entire* launch handshake was **~6 ms of a 46.6 ms host self-time
budget**, and a cross-run fit put the marginal cost of one launch at **~13 µs** — a few ms for a few
hundred launches, against a host tail of **~20 ms**. **MEASURED.** So a break's second charge is real
and host-side, but it is a fraction of the host tail, not a multiple of it — and §2.5.5 shows it is
**not the lever**.

> **Read (a) first, (b) second.** When you are deciding about a break, the question that changes the
> answer is *"what overlap does this cost me?"* — not *"how many launches does this add?"*. §2.5.5
> settles the second question: the launch count is not what the wall tracks.

### 2.5.2 Coalescing is the direction — and the trap that makes it silently fail

The direction is **fewer, larger regions**. Two reasons, and they are worth keeping separate:

* **Host-side:** one region means one guard evaluation and one launch/sync set instead of G of each
  (§2.5.1b).
* **Structural:** a region is the *unit inside which* one submission is possible. You cannot get
  independent cross-engine ops into one submission if they are in different regions to begin with.

**But "fewer regions" is necessary, not sufficient.** §2.4 showed that a single region can compile to
several recipes — and therefore several submissions — whenever a node inside it is not clusterable. So
coalescing regions removes the *region-level* overhead; it does not by itself produce the submission
count that §2.2's law is about. **Do not report "we coalesced" as "we fixed the overlap".**

> ### The trap
> **Integers specialize exactly like shapes** ([Chapter 0](00-tldr.md), fact 2). A Python `int` used in
> traced code becomes part of the guard set and produces **a separate compiled graph per distinct
> value**. So a coalesced region that spans G layers **still compiles to G graphs** if it guards on
> each layer's int constant — you have merged the regions, paid the compile, and kept every
> specialization.

**The verification step — stated as a step, not as a caveat:**

> **After any coalescing change, measure the specialization count and confirm it actually fell.** If it
> did not fall, the change bought host-side overhead only, and the int causing the per-layer split is
> still in the graph. Fix the int, or revert.

**The worked instance, and it is a negative result.** A layer-coalescing change took **43 per-layer
modules down to 11 stacks** — a ~4× reduction in modules — and **every operational count was
identical**: launches 401.0, syncs 649.8, distinct recipes 123, region labels 121, region executions
306.8/step, and a wall of 92.6 ms against a 93.2 ms baseline. The prediction that a coalesced region
spanning G layers still needs G graphs **was borne out**. **MEASURED** — `[DSv4F]`, one workload on one
box.

A second measurement makes the same point from the other side: **removing the graph break did not
remove the specialization either.** After the break was gone, the single remaining decode region still
needed **87 compile ids, 43 of them per-layer** — the specialization survives the break's removal.
**MEASURED** — `[DSv4F]`.

**A third trap sits in the same place, and it is about what a *write* costs rather than what a region
costs.** A view-level in-place write is free; the *same* write hidden inside an opaque op that declares
`Tensor(a!)` materialises whole copies of the mutated buffer. If you are moving work into a region,
check which spelling of the write you are moving with it — [Chapter 3](03-porting.md) §3.6.6.

### 2.5.3 The specialization count is a first-class metric

**Track it on every change to regions, kernels or graphs**, alongside launches, syncs and wall time.
Three reasons it earns a column of its own:

1. It is the quantity a coalescing change is *supposed* to move (§2.5.2) — without it you cannot tell
   whether the change did anything.
2. It is the quantity that costs compile time and device memory; each compiled graph holds its own
   buffers.
3. It is the cheapest early warning that something structural changed when you only meant to change a
   kernel.

**How to read it.** The count multiplying by **layer × constant × shape** is *expected*
([Chapter 0](00-tldr.md), fact 3): an N-layer model is an O(N × buckets) specialization case by
construction, not by bug. **So the metric's job is to catch the unexpected, not to be small.**

| what you see | what it means |
|---|---|
| count ≈ (layers × per-layer constants) + a shared set | **expected.** Do not "fix" it. |
| count **grows with steps** rather than with layers | a region is **recompiling per call** — a guard or a shape that changes every iteration |
| count rises and keeps rising in the steady state | **cache-limit thrash** — the compiled set no longer fits, old entries are evicted and recompiled |
| count **unchanged** after a change meant to reduce it | the change did not address the guard — see §2.5.2 |

**How to tell "expected large" from "thrash": look at the steady-state recompile total, not the
count.** In the healthy runs the post-warmup recompile total was **exactly zero** — the compiled set was
reached during warmup and never churned afterwards — while the runs that deliberately forced a
recompile had a non-empty compile list. **MEASURED.** So the baseline to hold is *"zero recompiles
after warmup, at whatever count that implies"*; a rising count in the steady state is the signal.

**Three counting traps, all measured:**

1. **A `recipeHandle` count is not a recipe count.** Handles ranged **1 → 13 across runs at an
   identical per-step recipe count**, because the recipe cache re-missed. The per-step recipe count was
   stable 20/20. Count per-step **launches**; treat the handle count as diagnostic only (§2.4).
2. **A per-region "variants" figure can be inflated by the rank count.** A recorded per-region figure
   of **172** was **43 × 4 ranks** — the same specializations counted once per worker. **Never quote a
   rank-multiplied number as a specialization count.** **MEASURED.**
3. **Region executions per step is the cheap proxy.** It equals the per-step count of the dynamo
   cache-lookup events **exactly**, in every run measured — so you can read "how many times did my
   compiled regions execute this step" from a trace without a graph dump. **MEASURED.**

### 2.5.4 One worked example, and the shape of the number

`[DSv4F]` — one workload, one box. Not a general law.

The port's recorded scale was **~194 distinct recipes per decode step, of which 160 were used by
exactly one layer** — these are **recipes**, not the 160 *windows* of §2.9 rule 8; the digits coincide
and the objects do not — i.e. almost all of the specialization was **per-layer**, so the count scaled with
the number of layers rather than with the number of region kinds. That is [Chapter 0](00-tldr.md)'s
fact 3 made concrete: an N-layer model pays roughly N per-layer graphs before anything is done wrong.

> **A flag on those figures, because it matters more than the figures.** The same archive records that
> 194/160 comes from **one code state** and is **stale** for the later ones, where the measured
> quantities are different again — **43** per-layer compile ids inside the single decode region, **117**
> specialization keys per step, and **70–123** distinct recipe handles depending on the code family.
> **Use the *shape* of the number — "almost all of it is per-layer, so it scales with the layer count"
> — not the number itself.** **MEASURED for the shape; the absolute figures are code-state-specific and
> one of them is explicitly flagged stale in the record.**

The arithmetic, generically:

```
distinct graphs  ≈  (layers × per-layer constants)  +  (shared graphs)
```

The per-layer term dominates. That is why a change touching one shared region moves the count very
little, and why removing a per-layer int can move it a lot.

### 2.5.5 The launch-vs-occupancy question, resolved — and why the host charge is not the lever

**This section once said the question was contested and that neither reading was promoted. That was the
right thing to write at the time and it is the wrong thing to leave standing, because the question has
since been settled.** The sequence is worth keeping visible, because it is this manual's own rule in
action: an unresolved question gets **written down**, and then gets **settled** — it does not get
smoothed over in either direction.

Two competing models were fitted to the **same 13 archived runs** — every run in the corpus that has
both a surviving trace and a measured wall (**MEASURED**):

| model | R² | RMS residual |
|---|---|---|
| `wall = k × node-executions` (launch/node model) | **0.346** | **107.85 ms** |
| `wall = a + b × merged` (occupancy model) | **0.999950** | **0.95 ms** |

**The occupancy model wins by two orders of magnitude in residual** — 0.95 ms against 107.85 ms, on a
run set whose walls span 73–403 ms. The fitted occupancy form is

```
wall  =  21.7  +  0.997 × merged          (merged = |union(device-busy intervals)|, ms/step)
```

i.e. slope ≈ 1 and a ~22 ms constant. That constant **is** the ~20 ms host tail of §2.5.1b — the
intercept and the host-side charge are the same quantity, which is why the host charge can be real and
still not be the lever. It is the same law §2.9 rule 1 states. **The step is
device-occupancy-bound, and the lever is the union, not the launch count.** Source:
`Archive/by-date/2026-10-11/node_law/`; the project's wall-bound analysis reaches the same conclusion
independently, and [Chapter 0](00-tldr.md) carries the same result under *"The one stale claim to watch
for"*.

> **⚠ CAVEAT added 2026-10-12 — a fitted `a + b·X` form is only as identifiable as the corpus it was
> fitted on.** This law is **MEASURED on the trace it was fitted to**, and it stays. But the project's
> own 2-D sweep shows the limit of the *form*: fitting `W = a + b·M` to a **1-D** sweep gives slope
> **3.22** when the swept knob is matmul *size* and **1.54** when it is matmul *count* — the same
> physical quantity, a **2× discrepancy** — and the resulting 2-point fit misses an intermediate size
> by **15 %** (§2.7). **Read a fitted `a + b·X` as a description of its run set, not as a law.** Before
> you extrapolate one, check that the corpus **varied more than one dimension**, or that the dimension
> you are extrapolating along is the one that was swept. This is §2.9 rule 4 — a collinear corpus
> cannot be separated by more fits — applied to the *form* of the fit rather than to the choice
> between two of them.

The launch model does not merely fit worse — it has **no usable coefficient.** Its per-run `k` varies
by **171 % of its own mean** across the 13 runs, so there is no stable microseconds-per-node to
design against. A quantity whose coefficient moves by more than a factor of 3 between runs at
comparable node counts is not a law; it is a coincidence of the run set.

**Robustness, because the obvious objection is the two outlier runs.** Excluding the two runs with the
pathological write path (n = 8), the launch model improves to R² 0.536 / RMS 4.73 ms while the
occupancy model gives R² 0.983 / RMS 0.90 ms — **the occupancy model still wins by ~5× in residual**,
and the per-run `k` is *still* not constant (≈2.5–3.1 µs/node on the healthy runs against ≈9–10 on the
pathology). The conclusion does not rest on the outliers. **MEASURED.**

#### What this does to §2.5.1b

> **The second charge is real, bounded, and NOT the lever.** The measured magnitude — ~6 ms of a
> 46.6 ms host self-time budget, ~13 µs marginal per launch, against a ~20 ms host tail — is
> *consistent* with this resolution rather than in tension with it. "A graph break costs twice" is
> **true**: the second charge exists and it is host-side. But it is a **fraction of the host tail**,
> and the host tail is a **fraction of the step**. The lever is the **union**.

That is the correct reading, not a hedge. It is also why §2.5.1a — the overlap a break destroys — is
the charge to weigh first: it is a `MAX`-to-`SUM` change in the quantity the wall actually tracks.

#### The clinching intuition

**The port once removed 34 % of a step's summed kernel time and the wall was flat.** Under the launch
model that is a puzzle: less work should be faster. Under the occupancy model it is exactly what you
would expect — **removing summed work does not reduce the union**, and the wall tracks the union.
The removed kernels were overlapped, so the union barely moved. That is the same fact as §2.9 rule 1's
**925 ms of summed device work inside a 70.5 ms wall**; the two observations are one observation.

#### A prospective test, for a reader on a different code state

The fit above is **cross-sectional** — it compares runs to each other. If you are on a different code
state, or you simply do not trust a fit, the test below measures the same thing **within** one code
state. It is now a **confirmation test rather than a tiebreaker**: the conclusion above says which way
it will come out, and the experiment is how you check that on your own machine.

Take one kernel family and remove its **device time only**: make the kernel return immediately, or run
its loop twice. Keep its **launch, op count, region structure and recipe count byte-identical**, so the
only thing that changes is how long the kernel occupies the device.

| hypothesis | predicted effect on the wall |
|---|---|
| device-occupancy-bound (**the resolved answer**) | the wall moves by that family's **measured union contribution** |
| launch/serialization-bound (**the stale framing**) | the wall is **flat** |

The two predictions are **~20 ms apart on a ~90 ms step** — a 20 %+ effect, far above the run-to-run
spread — so one run decides it. If you inherit a launch-bound claim, this is how you refute it on your
own evidence.

#### What is still not measured

**How much of the ~20 ms host tail is sync *waiting* versus host *computing*.** That sub-question is
still open (§2.12, item 8) — the sync events carry a runtime stream id rather than an OS thread id, so
they cannot be intersected with the runner-thread frames. It is a **different question** from which
side of the ledger the step's wall is on, and the answer above does not depend on it.

---

## 2.6 The `MAX` rule, and the precondition that makes it true

The composition rule people want to write down is a `MAX`:

```
per recipe r:  T_r = MAX over independent cross-engine groups in r ( SUM within each group )
T_step      = SUM over recipes r of T_r  +  n_ops × per_op_cost
```

`MAX` applies **only** between groups that are independent **and** cross-engine **and** share one
recipe. Everything else is a `SUM`.

> **The `MAX` holds only when the groups do NOT share a bottleneck — engine OR memory.**
> Where they do share one, `T` moves toward the sum of the **shared-resource** demands.

**MEASURED** on three synthetic recipes with **identical dispatch structure**:

| pair | shared bottleneck | **absorbed** |
|---|---|---|
| MME + TPC | HBM only — which the MME barely touches (~13 % of its time) | **82 %** |
| MME + DMA | **the DMA engine itself** — the MME's staging `memcpy_dma` nodes ride the same queues as your explicit copies | **48 %** |
| TPC + DMA | **two HBM-saturating workloads** | **21 %** |
| TPC + MME + DMA | mixed | **42 %** |

**82 / 48 / 21 % from three recipes with identical dispatch structure.** Dispatch order cannot
produce that spread; a shared bottleneck can. The accounting also closes against an independent
model: a pure HBM-bandwidth-sharing model (the most optimistic, with **no** engine serialisation at
all) predicts 1.070 ms for `tpc+dma` against 1.082 observed — a **1.1 %** match. **MEASURED.**

**Practical reading: MME × TPC overlaps beautifully; TPC × DMA barely at all.** And the two routes
agree — the composition interaction measured independently (adding TPC at fixed DMA work inflates
the DMA window 2.64×, §2.7) lands on the same conclusion as the table's own 21 % row.

### Scope qualifiers — all of them explicit

* **The percentages do NOT transfer to your model.** They come from synthetic C-API recipes with
  identical dispatch structure. Use them as *ordering*, not as arithmetic.
* **A pure bandwidth-sharing model does not hold across sizes.** That 1.1 % match was at **one**
  size; the sweep gives a fixed component plus a per-byte component (§2.7). **MEASURED.**
* **Not characterised:** how the scheduler shares time when the cross-engine groups are *unequal*;
  and **more than two engines in one recipe**. **UNKNOWN.** *(The second clause is **SUPERSEDED
  2026-10-12**: three engines in one recipe **are** now measured — see the closure of §2.12 item 4.
  The *unequal-groups* clause stands.)*
* **`MAX` is not bounded by anything other than the longer group** — the answer to "is it bounded by
  something else" is **no**, in general. **MEASURED** (for the pairs tested).

---

## 2.7 The co-residency effect — the most useful worked example in this chapter

This is the section to read if you read only one.

### The observation that started it

Measuring an engine's throughput "in the model" gave inconsistent numbers — apparently
**shape-dependent**. A solo-vs-solo comparison killed that explanation (**MEASURED**):

| shape, slice factor (both arms) | solo | + a DRAM-bound TPC co-tenant | ratio |
|---|---:|---:|---:|
| 4096³, f = 4 | **430.0 TFLOP/s** | 407.4 | 1.056× |
| 3072³, f = 3 | **427.6 TFLOP/s** | 285.7 | **1.497×** |

**Solo the rate is flat — 430.0 vs 427.6 TFLOP/s, 0.6 % apart.** The spread appears **only** when
another engine shares the recipe. So the apparent "shape dependence" was a **recipe-context
(co-tenant) effect, not a shape effect**. Note that normalising by the slice factor would have been
**wrong**: the factor is identical within each pair, so it cannot separate them.

### The controlled experiment: vary the co-tenant, hold everything else

A DMA-bound group measured at fixed DMA work, with the co-resident TPC load as the only variable.
The DMA nodes are byte-identical across arms (verified from the graph dumps: the same per-node
`dma_expected_duration` values, Σ = 78.101 µs in both) — **MEASURED**:

| arm | TPC co-resident? | Σ modelled DMA work | measured DMA union | **union / model** | identical-pair spread |
|---|---:|---:|---:|---:|---:|
| A | no | 78.101 | 91.426 | **1.171×** | 1.02–1.07× |
| B | **yes** | 78.101 | **303.996** | **3.892×** | 1.07–1.73× |

**Identical DMA work, 3.33× different measured DMA busy, caused only by adding TPC ops.** Two things
follow:

1. **Per-transfer variation is not intrinsic to the DMA engine.** The pre-committed falsifier —
   *"two identically-priced, identically-structured transfers legitimately differ ~3×"* — was
   **not met**: without a TPC co-tenant the spread is 1.02–1.07×. A hypothesis that DMA records are
   inherently unreliable is **refuted**.
2. **"A TPC op cannot occupy a DMA queue, so a TPC op cannot affect the DMA record" is false.**
   The DMA node count is *equal* in both arms, so this is not queue load either.

### The size sweep, and the falsifier that fired against its own author

Fixed TPC load, four matmul sizes (**MEASURED**):

| arm | TPC | Σ model `M` | DMA union `W` | `W/M` | identical-pair spread |
|---|---:|---:|---:|---:|---:|
| `2048²` no TPC | no | 41.044 | 59.361 | **1.446** | 1.36–1.47× |
| `4096²` no TPC | no | 134.673 | 147.976 | **1.099** | 1.09–1.17× |
| `2048²` + TPC | yes | 41.044 | 184.830 | **4.503** | 1.95–2.91× |
| `4096²` + TPC | yes | 134.673 | 387.419 | **2.877** | 4.23–4.53× |

Two fits fall out:

```
no TPC :   W ≈ 20.5 + 0.947·M        (slope ≈ 1 — the model's own rate; a ~20 µs fixed offset)
with TPC:  W ≈ 96.0 + 2.164·M        (~96 µs fixed offset; a 2.16× per-byte penalty)
```

> ### ⚠ CORRECTION 2026-10-12 — the `W = a + b·M` form above is **not identifiable from a 1-D sweep**
>
> **Keep the fits above — they are what this sweep measured, on the arms it measured. But do not
> read them as a law.** **MEASURED**, on a 2-D sweep (`t` independent TPC muls × **one** matmul of
> size `m`, one region, 12 arms): the **same physical quantity gets two different slopes depending on
> which knob you turn**.
>
> ```
> sw grid, one matmul, size varies:
>   TPC=0 mul :  W =  4.13 us + 1.023 * M      <- no co-tenant: the model's own rate
>   TPC=1 mul :  W =  8.07 us + 3.219 * M
>   TPC=2 mul :  W =  0.44 us + 3.958 * M
>   TPC=4 mul :  W =  0.25 us + 3.932 * M
>
> ratio grid, TPC=1 mul, BOTH count and size vary (8 arms, M 10.3..156.2 us):
>   TPC=1 mul :  W = 25.32 us + 1.537 * M
> ```
>
> **At TPC = 1 mul the slope is 3.22 if you vary matmul *size* and 1.54 if you vary matmul *count* —
> a 2× discrepancy.** So `W` is **not a function of `M` alone**, and a two-parameter linear form is
> **not identifiable** from a 1-D sweep. Concretely: the archived 2-point fit `96.0 + 2.164·M`
> predicts **265.0 µs** at a point the 2-D sweep measures at **305.0 µs — 15 % off**, at a size the
> fit never saw. **The harness reproduces the archived *endpoints* to 0.4 %** (`t2_08_tpc_mme_indep`
> p = 3.892 archived vs 3.91 here; `x_dma4096_tpc` p = 2.877 archived), so **this is the fit
> failing, not the harness.**
>
> **Two structural facts about the window itself** (also MEASURED, from per-node grouping) — they are
> part of why the "fixed component + per-byte term" reading is suspect:
>
> * **runs/step = 1.00** for essentially every node group: one node execution is **one contiguous
>   window**, not a scatter. Marker fan-out is 25×–130× (many beats inside one window), consistent
>   with §2.9 rule 2.
> * **The window is invariant to DMA queue load but sensitive to a TPC co-tenant.** The same node
>   (`…/memcpy/0`, 3072² shape) measures **8.89 µs** with no co-tenant (identical in both runs) and **22.4 µs** with one TPC
>   mul (**run 2; run 1 measures 23.12 µs**) — 2.5× (run 2) / 2.6× (run 1). Yet at fixed DMA it is **flat** across TPC load: W = 77.2 / 76.3 / 75.4 µs for 1 / 2 /
>   4 TPC muls. So the inflation is a **step change when the first co-tenant appears, plus a weaker
>   per-byte term** — a threshold, not a proportion. **Why** it is a threshold rather than
>   proportional was not isolated (HBM contention vs residency vs queue interaction): **UNKNOWN.**
>
> *(Source: `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §4.2.
> The ratio curve that replaces the linear law as the usable form is in the
> closure of §2.12 item 5.)*

> **The pre-committed falsifier fired.** The author had committed in advance to *"if the no-TPC arms
> give `W/M` in [1.05, 1.25], then my 'union exceeds the modelled work' finding is specific to
> TPC-bearing recipes."* At 2048² the no-TPC arm gives **1.446** — outside the band. So the claim
> *"without a co-tenant the channel IS the transfer time, proportionally, at every size"* is
> **size-dependent and not general**. That is a narrowing of the finding, and it is recorded as one.

**Design lesson from the same experiment.** A second arm varied the **TPC op count** at **fixed DMA
size**: ΔW = **+77.5 / +140.6 / +245.8 µs** for TPC unions of 0.278 / 0.554 / 1.100 ms, i.e.
**279 → 254 → 223 µs of DMA window per ms of TPC** — near-linear and mildly sub-linear. So the
co-tenant effect is **load-proportional**, which makes it a **memory** lever rather than a fixed
overhead. **MEASURED.** But the same design **held the DMA size fixed**, so at 1 and 2 TPC ops there
is a single `M` and the total effect **cannot be split into intercept and slope** — the intercept is
identifiable only at 0 and 4 TPC ops. The question *"does the rate penalty scale with the load?"*
**was never measured**. **UNKNOWN**, and the arm that would settle it is a 2-D sweep (vary TPC count
**and** DMA size), not a 1-D one.

> **SUPERSEDED 2026-10-12 — the 2-D sweep was run.** **MEASURED**: the co-residency inflation
> `p = W/M` as a function of the TPC:DMA ratio (**19 arms — 17 ratio points, the three no-co-tenant `sw_t0_*` arms collapsing into one row** — `t` independent TPC muls × matmuls):
>
> | regime | TPC/W ratio | `p = W/M` |
> |---|---|---|
> | **no co-tenant** | 0.00 | **1.15 – 1.42** |
> | co-tenant, ratio **1.0–1.8** | 1.01 – 1.77 | **1.67 – 2.28** |
> | **transition** | **1.96 – 2.29** | **3.34 – 3.50** |
> | saturating | **≳ 3.2** | **3.86 – 4.01** |
>
> *(Corrected 2026-10-12 — the saturating row previously read `≥ 1.96 → 3.86 – 4.01`, which left the
> three arms at ratio 1.96 / 2.08 / 2.29 — `p` = 3.50 / 3.34 / 3.46 — in no row of the table. Over
> ratio **≥ 1.96** the artifact holds **12** arms spanning `p` = **3.34 … 4.01**, and the
> **3.86 – 4.01** band holds only for ratio **≳ 3.2** (the **9** arms at ratio 3.22, 3.28, 3.85, 5.34,
> 5.68, 6.67, 9.87, 10.67, 19.09). Counted from the arm records, `raw/eo_results.json` +
> `raw_ratio/eo_results.json`, re-reduced by `eo_analyze.py` §C3.)*
>
> So the penalty is **not** a fixed overhead and **not** linear in the load: it is a **step at the
> first co-tenant**, then a rise through a **transition band at ratio 1.9–2.3** (`p` 3.34–3.50) to a
> **plateau at 3.86–4.01 for ratio ≳ 3.2**. The two arms that bracket the model's
> own operating ratio are `rat_t1_n4_m3072` (r 1.51, p 2.28) and `rat_t1_n2_m4096` (r 1.77, p 2.23).
> **The full curve, and what it is for — the ceiling on `p` for the engine's disputed intersection —
> are in §2.12 item 5 and in chapter 3 §3.6.3.** *(Source:
> `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §4.3.)*

### The rule

> **Always control the co-tenant.** A rate, a duration or an overlap fraction measured with an
> uncontrolled co-tenant is a measurement of *that recipe context*, not of the hardware.

This applies to your own code too: if you are trying to decide whether an op is "slow", measure it
**solo** in a region that contains nothing else, and then measure it again in the region you
actually ship. The difference is the co-tenant cost, and on Gaudi 2 it is routinely 2–4×.

---

## 2.8 The validity gate: precondition, or `INCONCLUSIVE` — never "negative"

An arm whose device queue is **empty** at the moment the host stops issuing cannot test overlap. It
is **INCONCLUSIVE**, never a negative result. This is not a stylistic preference; it is how the
project reached a wrong conclusion once, and the correction cost a full re-run.

**The precondition to assert, per arm:**

```
precondition_ok  =  (ops_available_at_H_end >= 2)  AND  (device_busy_frac >= 0.80)
```

where `ops_available_at_H_end = (D_end − H_end) / per_op` is the already-issued, unexecuted device
work in declared-op units, and `device_busy_frac = union(all) / (device span per step)`.

### Why: the achieved rate tracks `device_busy`, and the collapse is starvation

Five solo arms, same op, varying shape (**MEASURED**):

| `device_busy` | achieved rate (TFLOP/s) |
|---:|---:|
| 97.5 % | **430.0** |
| 91.6 % | 427.6 |
| 35.7 % | 396.0 |
| 7.1 % | 332.3 |
| **2.0 %** | **159.0** |

The rate collapses as the queue drains. **The collapse is queue starvation, not a hardware
property.** Two consequences:

* **Never quote a rate from an arm that failed its precondition.** In that five-arm set the
  pre-committed falsifier **fired for four of the five**; those four rates are INCONCLUSIVE, not
  negatives. A naive reading of the lowest point would have scored a false *"the hardware is slow at
  small shapes"*.
* **A "no overlap" result with an empty queue is not evidence of no overlap.** Reproducing the
  project's own earlier arm gave exactly that: a 2-op compiled arm reported a false INCONCLUSIVE
  until two measurement bugs were fixed (§2.9, rules 6–7).

### The gate is a claim about the *arm*, not about the code

Write the assertion into the arm. Report `INCONCLUSIVE` and the reason. A number that passes the
gate and a number that fails it must never appear in the same column.

---

## 2.9 Measurement discipline — ten rules, each with the instance that produced it

These are stated as rules because each one cost us a wrong number.

### Rule 1 — Build durations from **unions**, never from sums

*(Chapter 1 §1.7 covers the marker and duration statistics in full; the rules are repeated here
because the overlap measurement is stated in terms of them.)*

A duration is `|merge(intervals)|` over the relevant device events, replication-corrected. Summing
event durations measures *work issued*, not *time spent*, and on this hardware the two differ by an
order of magnitude: **925.1 ms/step of summed device work executes inside a 70.5 ms wall — 13.1×
parallelism** (**MEASURED**). Only the **un-overlapped** part of a reduction counts. A useful
corollary for optimisation: *reducing summed work is not the lever; reducing occupancy is.*

**A union is an occupancy, not a removal price — and this is the second half of rule 1.** Once you
have a union, the natural next step is to read it as *"what this family costs me"*. It is not. The
most a change that **removes** a family can free is that family's **leave-one-out** (LOO) — how much
the merged union actually falls when its intervals vanish — and **the LOO is smaller than the
family's own union, because families overlap.** Measured on the decode trace: the **710** family
unions sum to **88.65 ms** against a merged union of **66.14 ms**, so **22.5 ms (25 %)** of the summed
union is cross-family overlap; and over the **seven** event-subset cuts whose LOO is fractional, **the
LOO is only 23.4 %–65.5 % of the cut's own union** — `fp4_dequant` 22.6330 → 12.0862 (53.4 %),
`DmaMemcpy` 16.9399 → 10.7841 (63.7 %), the `_internal` staging subset 16.9219 → 10.7751 (63.7 %),
`GEMM` 11.4767 → 7.5155 (65.5 %), `DmaTranspose` 7.0449 → 1.6471 (23.4 %), `DmaBroadcast` 5.0750 →
1.5357 (30.3 %), `gather_nd_*_u8` 4.3820 → 2.0692 (47.2 %). **So quote the LOO when you are pricing
and the union when you are describing occupancy — and never substitute one for the other.**
*(Chapter 3 §3.6.4 carries the same table with both columns.)* The limit case is instructive: a cut
**inside** a recipe can have an LOO of **exactly 0.0000** on **0.2505 ms** of work, because the rest of
the recipe still occupies that time. **MEASURED** *(artifact: `Archive/by-date/2026-10-12/span-ranking/`
— the event-subset table, `sr8_spans.py`, `sr11_final.py`; the family unions and `merged` cross-check
against the project's own per-family LOO ranking.)*

**"A union is an occupancy, not a saving" — two instances, both on the same model, both with the
prediction right about the *removed* work and wrong about the *saving*.** Deleting work from a
multi-engine union frees the **work**, not the **time**; the time freed is only the part that was on
the **critical path**. **(i)** The MoE dequant's `repeat_interleave` expand: its four nodes carry
**74.2973 µs/call** of `dma_expected_duration` and a same-kernel ablation measures **68.1 / 83.8 µs
(mean ≈ 76)** — so the *removal* is confirmed to ~3 % — but a replacement kernel that needs no expand
nets only **27.4 µs/call**, because the other ~49 µs was **already overlapped with the TPC kernel**.
**(ii)** An fp4 kernel change that removed **18.22 ms** of device busy yielded **11.32 ms** of
`merged` (**62.1 %**): the shorter kernel **exposed** 6.91 ms of its own recipe's GEMM/DMA work that
it had been covering. *Artifacts: `notes/fp4-cm-major-route.md` §4–§5 and
`Archive/by-date/2026-10-14/fp4-layout-probe-cm/` (i); `Archive/by-date/2026-10-13/dsv4-profile-run/`
(ii).* **MEASURED**

**A span table should also report, per span, how many recipe LAUNCHES it contains** — a span's
headline cost can be dominated by **one staging node that its boundary happens to enclose**
(Chapter 3 §3.6.5).

Corollary for engine attribution: the same engine can be recorded **twice**, and the two records
disagree. On one trace the DMA engine's union was **21.713 ms** via one channel and **7.045 ms** via
another, on the same tids with the same marker — so at least one of those interval sets is not an
execution window. **MEASURED.** If two records of the same resource disagree by 3×, pick one
deliberately and say which, and validate the choice against an independent property (e.g. does it go
serial when the resource saturates?).

### Rule 2 — A marker's duration is **not** an execution window

A node execution emits many overlapping marker sub-events; their **union** is the window. Measured
self-overlap: MME markers **2.7×** their own global union; DMA markers **14.5×**. But the MME's
inflation is **entirely cross-tid** — *within* each MME tid the markers are serial (**1.03–1.04×**)
— so the MME's **global union is a defensible duration**, and the marker route is a usable rate
measurement *in a given recipe context*. **MEASURED.** The DMA marker caveat is unchanged, so every
DMA `ms` figure that comes from markers is a *marker* figure and should be labelled as one.

### Rule 3 — Never infer a mechanism from an identifier

**A name, filename, comment or single record cannot be *wrong*, so it cannot be *evidence*.** Four
instances in one day, all from the project lead, two of them on his own paraphrase of an argument.
Concrete instances worth remembering:

* A **clock constant** was asserted for weeks on the basis of a string in a summary; the string
  appears in neither relevant binary.
* An **engine label** on a benchmark row was wrong: a "DMA engine" row was actually a TPC kernel —
  eager `y.copy_(x)` never touches the DMA engine (768 TPC kernel events, zero DMA events); under
  `torch.compile` the same op does go to the engine. The row had propagated into two notes.
* A **throughput label** ("215 TFLOPS") conflicted with the value the same artifact's slope implied.

**Parse structured strings by field, never by substring.** A retracted figure in our own work came
from `'Height=1' in s` — which is `True` for `'Height=128'`. The same class of bug appeared again as
a column mix-up: a derived "row dimension" was actually read from the *feature* column
(`out[0]` instead of `out[1]`), and the resulting claim survived two review rounds because nobody
cross-tabulated. **Cross-tabulate your filters: check that "M is present" and "the shape is 2-D" are
the same set.** If they are not, your filter is not testing what it claims.

### Rule 4 — A result is overturned by adding a **control**, not by adding a measurement

A **collinear corpus cannot be rescued by more measurements.** If every arm moves several variables
at once, the answer is a new **arm**, not more rows. Our clearest instance: the 2-op overlap question
was answered the day the harness stopped changing shape, ops, size *and* orchestration together and
instead held all but one fixed.

Two corollaries:

* **Check whether your corpus can separate the things you are comparing.** In one fit, launch count
  and sync count correlate at r ≈ 0.99 across the runs available — the data **cannot** separate them,
  and any claim to have done so is an artefact of the fit. **MEASURED.** (This is a *fit* statement —
  it says the two counts are collinear, not that either predicts the wall. §2.5.5 answers the second
  question, and the answer is that **neither** count is what the wall tracks.)
* **Prefer a pair that differs in exactly one dimension.** The cleanest result in this chapter is a
  2×2 whose four cells differ in exactly one thing (see §2.7's table A/B, and §2.2's size-matched
  pair).

### Rule 5 — A pre-committed falsifier is only usable if the arm **varies** the quantity it is about

An arm that holds the variable fixed renders its own falsifier **unevaluable by construction**. This
is a **design flaw, not a null result** — and it must be reported as a design flaw. Our instance:
an arm pre-committed to *"if the rate penalty scales with load, X"* held the load fixed at every point,
so the question was never measured (§2.7). **Say so explicitly**; do not quietly report the arm as
having "passed". *(The question itself was subsequently measured, by the 2-D sweep this rule implies —
see §2.7's 2026-10-12 update and §2.12 item 5. The rule stands: it is about the **arm**, not about
the answer.)*

### Rule 6 — Collect the profiler activity you need, and no more

Collecting CPU activity alongside device activity **inflated the trace's host span ~3×** (8.043 ms
vs **2.505 ms** with device activity only, against an unprofiled wall of 0.29 ms/step). That
collapsed the queue-depth metric from **23.3 → 1.67** and turned a device-bound arm into a **false
INCONCLUSIVE**. **MEASURED.** If your analysis reads only device events, collect only device events.

### Rule 7 — A host that blocks in the allocator looks host-bound

An arm that allocated ~2.3 GB/step of fresh output buffers was paced by the host at the device rate
— device busy **46 %**, queue drained at `H_end`. Fixed by reusing a preallocated output buffer. The
enqueue cost measured on its own is only ~8.5 µs/op. **MEASURED.** Before concluding "the step is
host-bound", check that your arm is not simply allocating.


### Rule 8 — A factor that is exactly the step count is a **normalisation**, not a contradiction

**Two counts of the same thing that agree on the moments but differ by exactly the number of steps are
one measurement, not two.** Agreeing to four digits on the median *and* the total is the tell — a
genuinely second mechanism would not reproduce both.

**The instance, and it cost the project a whole session of carrying a false contradiction.** Two of its
own notes reported the same window population:

```
note A:  total = 1.980 ms/step   n_windows/step = 40.0   median = 49.52 µs
note B:  windows = 160           total = 1.980 ms/step   median = 49.52 µs
```

Same median, same mean, same max, same total — and a count differing by a factor of **4**, which is
exactly the number of steps in the trace. **They are one object at two normalisations:** one source
prints the **raw count over the whole 4-step trace** (160) while normalising the *total* by steps; the
other prints the **per-step** figure (40). **160 = 4 × 40.** Both artifacts reproduce from the trace to
four digits. **MEASURED.**

**The object, named precisely, with its units:** the **`fp4_dequant` × `GEMM` intersection windows** —
the intervals in which a TPC `fp4_dequant` execution and an MME `GEMM` execution are simultaneously in
flight. There are **160 of them in the 4-step trace, i.e. 40 per step**, totalling **1.980 ms/step**,
median **49.52 µs**.

**The histogram, so you can see where both numbers come from:**

| window duration | whole-trace n | per step |
|---|---:|---:|
| < 10 µs | **0** | 0 |
| 20–50 µs | **124** | 31.0 |
| 50–51.85 µs | **36** | 9.0 |
| **total** | **160** | **40.0** |

Every window is 20–51.85 µs and **none is below 10 µs**; median 49.52 µs, mean 49.50 µs, max 51.85 µs.
The two buckets are the instrument's own split at 50 µs — the population is a **narrow spread straddling
50 µs**, not two separated modes. **MEASURED.**

**What the discrepancy was *not* — check all three before accepting "two mechanisms":**

* **not a threshold** splitting or merging windows — no threshold appears in either instrument, and
  there are **zero** sub-10 µs windows for a threshold to act on;
* **not launches versus windows** — the number of launches carrying both families is **297**
  (74.2/step), a different quantity entirely;
* **not a different code state** — both instruments read the same trace.

> **"One window per layer" — approximately, not exactly.** The model has **43 layers** and the
> measurement gives **40 windows per step**, so the phrase is a **40/43 ≈ 93 % near-match, not a
> derivation**. Write it as *"≈ one per layer (40/43)"*, and say that **what the missing 3 are is
> unexplained**. An approximate match must never be written as an identity — that is the failure mode
> rule 3 documents one level up: a number that *looks like* a mechanism is not one until the mechanism
> itself is shown.

**The general habit this gives you.** Before treating two counts of "the same thing" as a
contradiction, divide them. If the ratio is the step count, the run count, the rank count or the layer
count, you are almost certainly looking at a **normalisation** — and the moments (median, mean, total,
max) agreeing is the evidence that it is one object. Compare the **moments**, not just the counts.

### Rule 9 — Two views of the same quantity are not two instruments

**"Two instruments agree" is a claim about independence, and it is false whenever both readings are
computed from the same measurement.** The test is not whether the numbers match — matching is exactly
what a shared definition produces. The test is: *would a change in the quantity being measured move one
reading and not the other?*

**The instance.** An investigation into what a compiled in-place write costs reported that a graph-dump
node census and an allocator peak "agree exactly", both giving 23.25 GB. They did not agree: the two
numbers are **23,251,200,000** and **23,251,465,216**. Both are computed from the same allocation, so
neither could have contradicted the other, and the dump side additionally **double-counted** — summing
four per-operation dumps of *one* buffer as four clones. The dump carries **no byte field at all**, so
it could not have been a byte instrument in the first place. The genuinely independent instrument in
that dataset was **wall time**, which is what actually licensed the conclusion that the copies execute.
**MEASURED.**

**The habit.** Name what each instrument is a *view of*, and say which one is independent. If two
readings are two views of one quantity, report them as one number plus one cross-check — not as two
instruments agreeing. And when a dump is involved, check that the field you are summing is a size at
all.

### Rule 10 — An instrument that overwrites its own evidence is not an instrument

**If a re-run can destroy the previous run's record, then any one-off observation is unrecoverable and
any claimed replication is unverifiable.** Artifact paths must carry the run's identity — arm, mode,
device, timestamp, attempt — so a repeat shows up as a repeat instead of being averaged away or
silently replacing the thing it was meant to check.

**The instance, and it cost two claims in one investigation.** A probe harness wrote every run of an arm
to the same output path and the same dump directory, so re-running an arm deleted the earlier record.
The consequence was concrete: a single sighting of a compiled write **silently not landing** could be
neither confirmed nor refuted, because no file anywhere recorded it — and a claimed replication of a
second arm survived only in a prose journal, not as an artifact. The honest status of the first was
**"the observation was never recorded"**, which is a *different statement* from "it did not reproduce":
conflating the two would have upgraded an unrecorded sighting into either a defect report or a clean
negative, and both are stronger than the evidence. **MEASURED.**

**The habit.** Before running an arm a second time, check that the second run cannot overwrite the
first. Then a non-reproduction is evidence rather than an absence of it.

---

## 2.10 The instrument: `DUMP_POST_GRAPHS`

For structural questions — *which engine is this node on, which recipe is it in, what is the modelled
cost* — the right instrument is the compiler's own post-graph dump, not the profiler trace. A trace
tells you what executed and when; the dump tells you **what the compiler built**, per node, before
anything ran. The two answer different questions, and for the questions in §2.2–§2.5 the dump is the
one that is definitive.

### What it gives

Set `DUMP_POST_GRAPHS=<dir>` and the GC writes **one JSON per compiled graph** into
`<dir>/.graph_dumps/`. Each file is
`{graphs:[{name, id, recipe_debug_id, device, nodes, tensors}]}`, and **every node carries a literal
`engine` field** — `"TPC"` / `"MME"` / `"DMA"` / `None`. **MEASURED.**

| field | present on | what it is |
|---|---|---|
| `engine` | all nodes | the literal engine: `TPC`/`MME`/`DMA`/`None` (`None` = logical/host wrapper) |
| `guid`, `type` | all nodes | op identity (e.g. `gemm`/`GEMM`, `memcpy_dma`/`DmaMemcpy`) |
| `exec_order_idx` | all nodes | position in the execution order |
| `bundle_index` | many nodes | which bundle/slice group the node belongs to |
| `origin_nodes` | all nodes | the IR node ids this physical task came from. **Present and non-empty in all 118 staging nodes censused, but not always resolvable** — treat it as a hint, never as a naming instrument (trap 4) |
| `is_logical` | all nodes | `true` for wrappers only — **all `engine=None`** |
| `input_tensors` / `output_tensors` | all nodes | tensor ids into the graph's own `tensors` list, which carries `max_shape`/`min_shape`/`dtype` |
| `mme_node_strategy` | MME | the geometry string (`Name=`, `Width=`, `Height=`, `Pattern=`, `BatchConcurrency=`, …) |
| `mme_expected_compute_cycles`, `unsliced_expected_compute_cycles`, `unsliced_expected_compute_utilization`, `rollups` | MME | the MME cost model |
| `dma_expected_duration` | DMA | the DMA cost model, µs (**a lower bound — see below**) |
| `tpc_working_engines` | all TPC | **the number of TPCs that ROI runs on** — an array with **one entry per ROI** (`[24]`, `[24,24,24]`), not a list of TPC ids. It is present on every TPC node in the census behind [Chapter 1 §1.7](01-basics.md) (**766 of 766** nodes) and equals the trace's `tid` count in **14 of 14** dumped shapes. *(The TPC *cycle* fields `tpc_tpc_cycles` / `tpc_asic_cycles` / `tpc_asic_time` are **not obtainable** — §1.7.)* |
| `num_of_ROIs` | all nodes | how many **passes** the node's index space is cut into — one `tpc_working_engines` entry per ROI. It is the **wave counter**: stock `relu_fwd_bf16` = 3, an archived `relu_fwd_f32` on 128 MB = 8, our custom nodes = 1 even at 128 MB. **The ROIs execute serially.** |
| `blocking_nodes` | all nodes | the compiler's own statement of which nodes this one waits on — **`[]` means independent**, and it is the field to read before claiming two nodes can co-schedule. It is also the observable for a **user-imposed** order: `synNodeSetUserProgrammability` makes the GC emit them (§2.1). |

**Three things the dump gives you that a trace cannot:**

1. **The per-node engine identity, from the compiler, not from a name heuristic.** Kernel-name
   bucketing is op taxonomy; this is the compiler's own field.
2. **The recipe membership.** Nodes that share a recipe file share a submission. This is how you
   check §2.2 condition 1 *structurally* rather than from a launch count.
3. **The modelled cost**, which you can compare against a measured window to get an error *sign*
   (see the caveat below).

### Four traps

1. **`mkdir -p` the dump directory first, every time.** A dump target that does not exist is treated
   as a **FILE**; concurrent dump writers then race and **fail the compile**:
   `Graph compile failed … synStatus 26`, `size mismatch after 3 retries … expected 29855 got 0`,
   `rename failed after 3 retries … errno: 2`. **MEASURED** (it cost a run). With four worker
   processes compiling concurrently this is not a risk, it is a certainty.
2. **`**`-globbing silently skips the hidden `.graph_dumps/` directory.** One recon saw **24 of 1,141**
   files on its first pass. **MEASURED.** Use `os.walk` / `find`, or glob `*/.graph_dumps/*.post.json`
   explicitly.
3. **`dma_expected_duration` is a modelled lower bound, not a measurement.** It is a per-node
   execution estimate at an assumed peak (~2.08 TB/s) plus a small constant, with **no queue wait and
   no contention** — i.e. the *optimistic* side. Measured against real transfers it is **~16 %
   optimistic at 64 MB and ~3 % at ≥256 MB** (**MEASURED**). Two consequences: (a) never treat it as
   a time; (b) when you divide a measured window by it, you are computing *"measured ÷ optimistic
   estimate"*, which is why a ratio near 1.15 validates and a ratio of 5.5 does not. **Its sign is
   not global** — on a **243-ROI** node of the model's own decode graph the same field is
   **1.19× HIGH** against the trace's summed descriptor durations (§3.6.5). The two readings are not
   the same quantity: a node's *window* versus a *sum of descriptor durations over a multi-ROI node*.
   Say which you mean.
4. **`origin_nodes` is not a naming instrument, and its ids may not resolve.** On the model's own
   decode graph the staging node carries `origin_nodes: [97147]`, and **that id is not in the post
   graph's own node-id set** (the graph's ids run 97422…97661) — so the field cannot name the tensor
   it came from, even though it is present and non-empty (which is why "absent/empty" tests do not
   fire on it either). **The instrument that does name the tensor is `input_tensors` /
   `output_tensors` resolved through the graph's own `tensors` table**, which carries `dtype`,
   `max_shape`, `strides`, `alias_of`, `offset`, `allocation` and `persistent`. **MEASURED**
   *(Chapter 3 §3.6.5; `Archive/by-date/2026-10-13/staging-origin-nodes-dump/`.)*

### A note on the MME cost model

*(Chapter 1 §1.6 has both cost models in full, including the TPC side. What matters for this
chapter is only the error signs.)*

The MME cost model is **accurate to 0.5 %**: the engine library carries its own clock constant
(**1650 MHz**, hardcoded, matching the SoC clock `hl-smi` reports) and
`262144 FLOP/cycle × 1.65 GHz = 432.5 TFLOP/s` against a measured **430.0–432.7**. **MEASURED** for
the clock and the agreement; the `262144` constant itself is **INFERRED** (it is not present as an
immediate in the disassembly; it is derived from geometry). *(Artifact for both:
`Archive/by-date/2026-10-11/synapse_cpp/GC_COST_MODEL_CLOCK.md` — §8 for the clock and its
`hl-smi` agreement, §9 for the absent `262144`/`131072` immediates.)*

**The two models have different error signs.** The MME model is accurate; the DMA model is optimistic.
**Neither may be used as a neutral reference without saying which one you are quoting.**

### What the dump does **not** give you

* **No timings.** It is a static artifact. Pair it with a trace when you need time.
* **No submission count.** Use `Launch`/`enqueueWithExternalEventsExt` for that.
* **No per-rank uniqueness by default.** With TP > 1 the workers share one dump directory unless you
  rewrite `DUMP_POST_GRAPHS` per rank *before any compile happens*; a shared directory reproduces the
  `synStatus 26` race above. **MEASURED.**
* **Not every graph — and a warm recipe cache gives an EMPTY dump.** Recipes deserialised from the
  recipe cache are never re-dumped, so a dump directory can under-represent what actually executed.
  **The cold-cache requirement and the run are MEASURED (2026-10-13):** the dump run pointed
  `LAD_RECIPE_CACHE_ROOT` at a fresh empty directory, cold-compiled every decode recipe, and wrote
  **21,763** post-graph files. **The warm-cache consequence is the *stated mechanism* for that
  precaution, not an observation (`source read`):** `DUMP_POST_GRAPHS` is written by the graph
  compiler, and a Synapse **recipe-cache hit loads a compiled blob and builds no graph** — so a warm
  cache yields an **empty dump directory and a false negative**, not an error. **Point
  `LAD_RECIPE_CACHE_ROOT` at a fresh empty directory whenever a dump is the instrument.** *(No
  warm-cache dump attempt exists in the archive, so the consequence itself is not measured; the size
  of the gap on a *partially* warm cache is still **UNKNOWN**.)*

---

## 2.11 A checklist you can apply

Working through this list on a region that "does not overlap" finds the cause in most cases.

**The four conditions of §2.2, in order:**

1. **Is it one submission?** Count `Launch`/step. If > 1, stop — nothing else matters until it is 1.
   Then confirm structurally: do your ops share one recipe (graph dump) and one `recipeHandle`?
2. **Are the groups on different engines?** Read the `engine` field. Do not guess from names.
3. **Is there a dataflow edge?** Read the IR. Check for buffer reuse/aliasing, not just obvious
   operand sharing.
4. **Are the durations comparable?** Measure each group solo, in the same region context. If the
   ratio is worse than ~2:1, you cannot detect overlap — fix the experiment first.

**Then, whatever you changed:**

5. **Did the specialization count fall?** If you coalesced regions, kernels or graphs, measure it
   (§2.5.2). A coalescing change that left the count unchanged bought host overhead only.

**And the arm's own hygiene:**

6. **Did the arm pass its precondition?** `ops@H_end >= 2` and `device_busy >= 80 %`. If not, report
   INCONCLUSIVE and change the arm.
7. **Did you control the co-tenant?** Measure solo *and* in-context; the difference is the
   co-residency cost, and it is routinely 2–4×.

**And the two rules about how you are reasoning:**

8. **Is anything you concluded resting on an identifier?** A name, a filename, a comment, a single
   record. If so, find a measurement or mark it **UNKNOWN**.
9. **Are two counts you are treating as a contradiction actually the same object at two
   normalisations?** Divide them. If the ratio is the step count, the rank count or the layer count,
   it is a unit, not a mechanism (§2.9, rule 8).
10. **If you are adding a measurement to resolve a disagreement, stop and add a control instead.**

---

## 2.12 What this chapter does **not** establish

Stated rather than omitted, per the README's convention.

| # | unknown | what would settle it |
|---|---|---|
| 1 | The exact shared-layer op-support table that decides `hpu_cluster` vs `eager`. We measured the decision *mechanism* and the outcome per construct, not the table. *(**Refined 2026-10-12:** the table only decides for a node **outside** the `custom_op` namespace, which short-circuits ahead of it — see the correction in §2.4. The table's **contents** remain unenumerated, which is what this item is about.)* | read the table from the backend, or run the placement pass over a candidate op set |
| 2 | Whether the engine's per-region recipe multiplicity follows this chapter's rule. **INFERRED**, not verified. | one engine run with the placement pass registered |
| 3 | How the scheduler shares time between **unequal** cross-engine groups. | an arm with a deliberately unequal pair (e.g. 4:1 durations) |
| ~~4~~ | ~~Behaviour with **more than two engines in one recipe**.~~ **CLOSED — RESOLVED 2026-10-12 (the independent case).** **MEASURED, three engines in one recipe, independent:** `union(all) = 0.7400` against `Σ[kernel] = 1.1232` and `max(single) = 0.6380` — the third engine costs only **+0.102 ms** of union, so **all three co-schedule** and the two added engines are almost entirely hidden. Pairwise fractions, **named to their arm set** (corrected 2026-10-12 — this row previously quoted `TPC∥DMA 0.63–0.99` and `MME∥DMA 0.79–0.89` "from the same arm family", which is not the 3-engine family's own range): **this arm's own pairs are TPC∥DMAk 0.632, TPC∥DMAh 0.978, MME∥DMAk 0.549, MME∥DMAh 0.873** — so **MME∥DMA over this arm is 0.55–0.87**, not 0.79–0.89. The wider set (**the twelve arms of the source report's §3.2 `ov` table** — `Archive/by-date/2026-10-12/engine-overlap/REPORT.md`) spans **TPC∥DMAh 0.464–0.994** and **MME∥DMAh 0.787–0.963**. *(Controls in the same wave: one dataflow edge collapses the three TPC-involving pairwise intersections to `0.000` and drops CROSS from 0.3831 to 0.0939 — that residual being MME∥DMAk, not MME∥DMAh; emission order changes nothing, 0.3831 vs 0.3826.)* **Still not run: the *bottleneck-varied* 3-engine recipe this row asked for** — every arm above is an *independent* body, so "how the scheduler shares time when three engines contend for one resource" folds into item 3 and stays open. Source: `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §2–§3. | — **done** for the independent case; the bottleneck-varied case is item 3, generalised. |
| ~~5~~ | ~~Whether the co-residency rate penalty **scales with the co-tenant load**. The deciding arm (a 2-D sweep over TPC count × DMA size) was designed but never run; the 1-D version held the variable fixed, making its own falsifier unevaluable.~~ **CLOSED — RESOLVED 2026-10-12. The 2-D sweep was run.** **MEASURED**: the window inflation `p = W/M` is **1.15–1.42 with no co-tenant**, **1.67–2.28 at TPC:DMA ratio 1.0–1.8**, **3.34–3.50 in the transition band at ratio 1.9–2.3**, and **saturating at 3.86–4.01 only for ratio ≳ 3.2**. *(Corrected 2026-10-12 — this row previously said "saturating at ~3.9–4.0 (ratio ≥ 1.96)"; over ratio ≥ 1.96 the artifact holds 12 arms spanning `p` = 3.34 … 4.01, so that band left the 1.96/2.08/2.29 arms unplaced. §2.7's regime table carries the corrected rows.)* So the answer to this row's question is **yes, it scales — with the TPC:DMA ratio — and it saturates**; it is *not* a fixed overhead. Note the second, equally measured fact that constrains the reading: **at *fixed* DMA the window is flat across TPC load** (W = 77.2 / 76.3 / 75.4 µs for 1 / 2 / 4 TPC muls), so the inflation is a **step at the first co-tenant plus a weaker per-byte term**, not a proportion. Curve and fits: §2.7's 2026-10-12 update. What the curve is *for* — the ceiling on `p` for the engine's disputed intersection — is item 11 and chapter 3 §3.6.3. Source: `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §4.2–§4.3. | — **done.** |
| 6 | Whether two HPU streams can *ever* overlap. We measured one configuration (36 ops/step, 119.1 ops outstanding at `H_end`, both queues provably loaded) and got exactly `0.0000`. **MEASURED 2026-10-12 — re-run with a *device-bound* arm; the precondition is superseded as this row's cited evidence, but it was *not* the null test.** The new arm is two HPU streams with deep queues: `ov = 0.0000` **exactly, both runs**, with **18.5 ops outstanding at `H_end` / 83.5 % device busy** (run 1) and **55.9 / 97.0 %** (run 2), and `union(all) = TPC + MME` to four decimals. **Two loaded queues still buy exactly zero overlap.** **The reconciliation:** there are **two distinct archived two-stream arms**, and the "0/40 ops in flight" reading is the **E5 audit's `07_streams_indep`** (`Archive/by-date/2026-10-12/e5_method_audit/AUDIT.md` §6 — host-bound, 10.9 % device busy, 0.3 ops outstanding at `H_end`), a **null test**. But the arm *this row* quotes is the **`t2_graph_overlap` arm `t1_02_streams_tpc_mme`** (`Archive/by-date/2026-10-10/t2_graph_overlap/`, README §5), which is **device-bound** — 36 ops/step, **119.1 ops outstanding at `H_end`, 97.8 % device busy, both queues provably loaded** (`precondition_ok = True`) — so the "119.1 / both queues loaded" precondition was **valid for its own arm**, and the 0/40 null belongs to a *different* arm. The *conclusion* is unchanged and now rests on **two** device-bound arms (`t1_02` and the new `str_bound`). | — **closed for the tested structure.** A stream arm with a *different* submission structure is still open. |
| 7 | Whether a `hpu_op` DMA interval is **execution** time or **submission** time. | a run with kernel-level (not `hpu_op`) DMA metrics |
| 8 | **The one surviving piece of the closed item 9:** what fraction of the ~20 ms host tail is *sync waiting* vs *host computing*. The sync events carry a runtime stream id, not an OS thread id, so they cannot be intersected with the runner-thread frames. | sync events carrying the calling tid, or a `record_function` around each sync at the point the runtime issues it |
| ~~9~~ | ~~**Whether a decode-shaped step is bound by launch/serialization or by device occupancy.**~~ **CLOSED 2026-10-11 — the occupancy model wins.** Both models fitted to the same 13 archived runs: `wall = k × node-executions` gives R² **0.346** / RMS **107.85 ms**, against `wall = a + b × merged` at R² **0.999950** / RMS **0.95 ms** — two orders of magnitude in residual, and the launch model's per-run coefficient varies by 171 % of its own mean. **The step is device-occupancy-bound and the lever is the union.** Source: `Archive/by-date/2026-10-11/node_law/`. See §2.5.5. | — **closed.** The surviving sub-question is item 8. |
| 10 | The size of the gap between "recipes in a dump" and "recipes that executed" (cache-deserialised recipes may not be re-dumped). | a run with the recipe cache cold vs warm, comparing dump contents |
| 11 | A worked, *engine-scale* overlap verdict. The harness results in this chapter are synthetic. The project's one engine-scale TPC∥DMA figure moved three times under measurement and is recorded as **UNRESOLVED**, not as a validated concurrency measurement. A *separate* engine-scale intersection — `fp4_dequant` (TPC) × `GEMM` (MME), **1.980 ms/step over 40 windows/step** (§2.9 rule 8) — is measured and reproduces exactly, but it is a **raw interval intersection**, not a validated concurrency claim: nothing here establishes that the two engines were doing useful work at the same time, only that their execution windows intersected. **NARROWED 2026-10-12 — still NOT closed, in either direction.** The engine-scale TPC∥DMA figure is no longer merely "unresolved": it sits **just over the ceiling `O = M`** — ≈100 % of the available DMA work hidden, against ≈45 % of the measured `hpu_op` DMA union — so it is **neither established as real concurrency nor dismissed as an artefact**, and it fails the `O ≤ M` test at the measured `p`. Quantified: the ceiling for it to be real is **`p ≤ 2.200`**, the `p` **bracketed at** the engine's own TPC:DMA ratio is **2.23–2.28**, and `O/M = 1.02` (band 1.01–1.04; the reconciliation is in §3.6.3 — the source's wider 1.01–1.59 came from a three-arm set that does not bracket the engine's ratio). Chapter 3 §3.6.3 carries the numbers and the one run that would settle it — an engine-side `DUMP_POST_GRAPHS` on one W0 decode step — **NOT RUN**. | the engine-side dump paired with the trace — **unchanged; that is still the deciding instrument** |

---

## 2.13 Examples

Runnable code for this chapter is in [`examples/02-regions/`](examples/02-regions/). Each file is
self-contained and annotated with **what to look for**:

| file | what it demonstrates |
|---|---|
| [`01_two_engine_recipe.py`](examples/02-regions/01_two_engine_recipe.py) | the minimal two-engine recipe: an independent TPC op and MME op in **one** `torch.compile` region. Read this first — it is the smallest thing that overlaps. |
| [`02_region_that_overlaps_and_one_that_does_not.py`](examples/02-regions/02_region_that_overlaps_and_one_that_does_not.py) | the same two ops, twice: once independent (overlaps) and once with a single dataflow edge (does not). The one-line difference is §2.2 condition 3. |
| [`03_inspect_graph_dumps.py`](examples/02-regions/03_inspect_graph_dumps.py) | how to read a `DUMP_POST_GRAPHS` directory: per-node `engine`, recipe membership, modelled cost. Uses `os.walk` (the `**` trap) and parses `mme_node_strategy` by field (the substring trap). |

See also [`examples/02-regions/README.md`](examples/02-regions/README.md) for the environment these
were run in and the traps that bite first.
