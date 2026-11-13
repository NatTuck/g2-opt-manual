# Chapter 0 — TL;DR

**Read this first, before writing or judging any HPU code.** A rulebook, not a summary — each rule
points at the chapter that develops it. **If this file disagrees with a chapter, the chapter wins; fix
this file.** Only well-established numbers appear here (verified hardware constants; bandwidth figures
consistent with spec less plausible overhead); one-experiment and one-workload figures stay in the
chapters. Workload-specific numbers are **[DSv4F]** (DeepSeek V4 Flash, TP4, one box).

**Tags**: **MEASURED** / **source read** / **ASSUMED** / **INFERRED** / **UNKNOWN** — see
[README](README.md). UNKNOWN means *we looked and did not find*; never fill it from general knowledge.

---

## Terms — the ones you cannot fudge

| term | what it is |
|---|---|
| **region** | one dynamo-traced FX graph from a `torch.compile` region |
| **node** | one op in that graph, carrying a **placement** (`hpu_cluster` or `eager`) |
| **eager placement** | an op the partitioner will not cluster; it runs outside every recipe. **Not** the same thing as eager *mode*. |
| **lazy mode** | `PT_HPU_LAZY_MODE=1` — the legacy execution model: ops accumulate into an implicit graph closed at a mark-step. **Legacy; not the default; not a viable target.** |
| **eager mode** | `PT_HPU_LAZY_MODE=0` — ops execute as issued; **each op still compiles to its own one-node recipe** (there is no region recipe). A reference point, never an option: N one-node recipes is N submissions and N program downloads. → *ch 6* |
| **compiled mode** | `torch.compile(fn, backend="hpu_backend")` — **the only supported target.** Modern vLLM Gaudi is `torch.compile`-only. |
| **recipe** | one compiled binary — **one per maximal contiguous run of clusterable nodes**. The region is partitioned *before* it becomes a recipe, so a `torch.compile` boundary is **not** a recipe boundary. → *ch 1 §1.3, ch 2 §2.2* |
| **submission** | one Synapse `Launch` — the only unit the scheduler co-schedules *within* |
| **engine** | TPC (24 VLIW cores) · MME (4 matmul) · DMA (8 tids = 4 cores × 2 lanes, 5 carrying HBM↔SRAM traffic) — read the dump's `engine` field; **never infer from an op name** |
| **bundle** | a VLIW issue group of ≤4 slots inside one TPC — **the TPC cost unit**; unrelated to a dump's `bundle_index` |
| **`merged`** | union of all device-busy intervals in a step — the quantity the wall tracks |
| **union / LOO** | a merged-interval duration (never a sum of events) / how much `merged` falls when a family's intervals vanish |

An `eager`-placed node runs **outside** every recipe, splits the recipe run, and emits no
`LaunchRecipe` — a recipe/handle count is not a submission count.

**The default is non-lazy.** The shipped bridge disables autoload only when `PT_HPU_LAZY_MODE=1`, and
the vendor comment claiming "the default mode is lazy" is **stale** — an instance of the "stale claim
to watch for" pattern below. The citations and the layer analysis are in *ch 8*.

---

## The five facts that change how you write code

1. **Everything you can ship is statically compiled — there are no dynamic shapes.** A `dynamic=True`
   request is forced to static compilation; every new shape is a new program. Design a fixed shape set.
   → *ch 1 §1.4* — and read the layer: a **bridge** policy (**ASSUMED (vendor)** as a mechanism,
   MEASURED as a recipe count), not a device limit. At the **Synapse** level a recipe can be declared
   size-parametric (`DATA_TENSOR_DYNAMIC`, exact M/N/K down to 1, `synStatus 35` outside the declared
   bound); nothing below Synapse is known — **hardware UNKNOWN**.
   [Residency note](notes/recipe-residency-and-memory-budget.md) §9.
2. **Integers: a Python `int` is *two* recipes per process, and an escape hatch exists.** A Python
   `int` in traced code joins the guard set — the first value is baked as a constant, the second
   recompiles to a graph that takes the int as a symbolic input and serves every later value, so the
   count is flat at **two** (it is one per value if you force `dynamic=False`, which our own
   benchmarking advice does). The unbounded case is a **cold process**, which re-bakes whatever value it
   meets first. A **0-d tensor** removes the program count entirely — one recipe across values *and*
   across processes (measured: 0 new recipes at an unseen value). Structural ints (`x[:n]`) cannot be
   escaped and grow with distinct values. → *ch 1 §1.4, ch 8 §8.8*
3. **The graph count multiplies: O(layers × constants × buckets) by construction, not by bug.** Check a
   large count against this arithmetic before treating it as a defect; a coalesced region spanning G
   layers still needs G graphs if it guards on per-layer ints. → *ch 1 §1.4, ch 2 §2.5*
4. **A graph break costs twice** — it destroys in-submission overlap *and* adds host launch/sync traffic.
   The overlap loss is the big charge; the host charge is a fraction of the host tail, not the lever.
   → *ch 2 §2.5*
5. **Prefer fewer, larger regions — necessary, not sufficient.** Verify after any coalescing change:
   **submissions** (a region still compiles to several recipes if any node is not clusterable) and the
   **specialization count** (it does not fall while the int guards remain). Coalescing alone has measured
   *moved no operational count*. → *ch 2 §2.5*

---

## The one measurement rule that changes what you optimise

**Price in device occupancy — the union — not in summed work.** The wall tracks `merged` (R² ≈ 0.9999
on the run corpus); summed device work exceeds `merged` by an order of magnitude on a decode step. Two
levels: **union within a recipe, sum across recipes** — recipes are serial, so per-recipe unions
partition the step exactly. A union is an occupancy, not a removal price: **removing work frees its
leave-one-out, routinely far smaller than the removed family's own union.** → *ch 2 §2.9 rule 1, ch 3 §3.6*

## The one stale claim to watch for

**"A decode step is launch/serialization-bound" is stale on our measurements; the step is
device-occupancy-bound and the lever is the union.** The occupancy model beats the launch model by ~2
orders of magnitude in residual; the launch model's per-run coefficient varies by 171 % of its own mean.
Removing a third of kernel time once left the wall flat. If you inherit a launch-bound claim, test it:
hold launches, op counts and recipes fixed and change only device time. → *ch 2 §2.5.5*

---

## Non-negotiables — the obvious slowdowns

- **All device work must be graph-compiled**, warmed at the exact shapes it will see, with **zero
  compiles after warmup** (count `compileGraph` events — the Synapse level, not dynamo's). A rising
  specialization count in steady state is cache-limit thrash; a count growing with steps is a per-call
  recompile. Track standing metrics on every change: specialization count, submissions, syncs, wall,
  `merged`. → *ch 2 §2.5, ch 3 §3.5/§3.10*
- **Tensor data stays on the device — no D2H/H2D mid-computation, not even a little.** A `.item()` is a
  graph break; `x.cpu().to('hpu')` round-trips bytes *and* splits the recipe. An arm allocating fresh
  outputs every step is paced by the host allocator and looks host-bound — reuse buffers.
  → *ch 1 §1.4, ch 2 §2.9 rule 7*
- **Graph breaks and extra syncs are the structural cost to avoid** — but price them in `merged`, and
  weigh a break first against the overlap it destroys.

---

## The overlap law → *ch 2 §2.2*

Cross-engine overlap exists **iff all four hold** (missing any of the first three gives exactly
`CROSS = 0.0000`): **(1)** the whole group in **one submission** (`Launch`/step == 1, same
`recipeHandle`) — **(2)** on **different engines** — **(3)** with **no dataflow edge** (cross-engine →
monitor wait; same-engine elementwise → fused, usually a win) — **(4)** at **comparable durations**
(detectability: groups differing >~2× show nothing either way).

- **Recipe boundaries are placement, not op identity.** Any node the shared-layer validator rejects is
  placed `eager` and splits the run. Ops in the **`custom_op` namespace short-circuit** the validator
  and cluster regardless; to co-schedule a hand-written TPC kernel, register it in the *existing*
  `custom_op` library, keep its body edge-free to the other branches, duration-match, and **verify
  placement** (`PRE_PARTITIONER` pass + trace) — never infer from the namespace. → *§2.3–2.4*
- **Two HPU streams buy exactly zero overlap** — measured device-bound. Do not design around streams.
  → *§2.12, ch 3 §3.8*
- **MME ∥ TPC overlaps well; TPC ∥ DMA barely at all** — a shared-bottleneck effect, not dispatch order.
  → *§2.6*
- **Control the co-tenant.** A rate measured with an uncontrolled co-resident engine is a property of
  that recipe context, not the hardware; the inflation is large and routine. Measure solo, then in
  context. → *§2.7*

---

## Hardware pocket card → *ch 1*

- **Engines:** 24 TPC (elementwise/reduce/gather/casts/softmax/custom kernels) · 4 MME (matmul only) ·
  8 DMA tids, 5 carrying HBM↔SRAM traffic — **one transfer at a time per lane, lanes chosen per-transfer
  round-robin; you cannot steer a copy**.
- **Memory:** 96 GiB HBM per card · **48 MiB SRAM** (one shared pool; compiler-managed) · **no L1/L2
  data cache between HBM and SRAM** — reuse must be engineered, it does not happen automatically · TPC
  local memory 1 KB scalar + 80 KB vector per core, plus small private per-TPC caches (64 KiB
  instruction, 32 data lines).
- **Bandwidth is size- and engine-dependent; a figure without its size is not a figure** (all
  **MEASURED**, ≥16 MB): TPC stream ≈ **2,147 GB/s** at ≥134 MB; DMA engine ≈ **1,776 GB/s** at 16–64 MB,
  ≈ **2,025 GB/s** at ≥256 MB (nominal spec 2,450 GB/s). Below ~16 MB, small-tile GEMV work loses
  bandwidth sharply — a TPC cliff, **not** a DMA curve (no DMA measurement exists there). Keep decode
  weight tiles ≥ ~16 MB.
- **Eager `torch.copy_` is a TPC kernel, not the DMA engine** — compiled, the same op does use DMA.
  Check `args["HW event name"]`, not the op you called.
- **Cost models have different error signs**: MME accurate (~0.5 %); DMA an optimistic lower bound,
  worse at small sizes; the TPC kernel cost model runs at **1800 MHz** on Gaudi 2 with its bandwidth
  terms hard-wired at **1700 MHz** (5.6 % apart, so the sign of its error depends on which term
  dominates). `dma_expected_duration` in a dump is a model, not a measurement. Neither is a neutral
  reference.
- **Markers are not execution windows.** Only the TPC marker is a true start→halt span; MME and DMA
  markers self-overlap heavily — build every duration as a per-node merged union, labelled with Synapse's
  own `args["HW event name"]`, never a name heuristic. → *ch 1 §1.8, ch 2 §2.9 rules 2–3*

---

## Writing a TPC kernel → *ch 4 (the model), ch 9 (the workflow)*

- **The cost currency is the VLIW bundle, not the instruction** — the device retires ~one bundle per
  cycle (~0.556 ns at 1.8 GHz); one vector slot per bundle, so the floor is the vector-op count. Count
  bundles (`--tpc-asm-format=2`), never lines or mnemonics.
- **The shipping scheduler packs ~1 instruction per bundle.** `-mllvm --pre-RA-sched=vliw-td` on the
  `tpc-clang` line in the kernel's `build.sh` cuts bundle counts substantially at equal or higher
  instruction counts, bit-exact at kernel level — **model-level safety is NOT established**, and the flag
  reaches only the kernels we compile.
- **One 256-byte vector register; 256/128/64 lanes for 8/16/32-bit elements.** Cost is per register, so
  per-element throughput is `i8/fp8 : bf16 : f32 = 4 : 2 : 1`. fp8 is native for logic/select/max, but an
  fp8 **add is ~3 instructions** — price fp8 arithmetic.
- **No wave, no warp, no occupancy: a TPC is single-context.** Only the asynchronous memory drain hides
  latency, so intra-TPC time is `MAX(ALU, memory)`, not a sum. Expose independent work **inside the loop
  body** (independent accumulators); a bigger `#pragma unroll` is *not* the lever and can be a loss.
- **Read the listing, including operands** — a clean compile can emit the wrong access (`ld_l`/`st_l`
  local instead of `ld_g`/`st_g` global), and an instruction count cannot tell a live op from a
  dead-code-eliminated one. Compare `.text`, never `.o` md5. Per-launch overhead grows sublinearly with
  bundle count, so small-shape microbenchmarks mostly measure the residue, not the kernel.
- **Shipping a kernel: two artifacts, and the error message is the least informative part.** The device
  ELF is embedded as bytes by `objcopy -I binary`, so the glue's `extern` symbol is derived from the
  *input filename* — get it wrong and the link still **succeeds**, failing only at run time.
  `GC_KERNEL_PATH` is a colon list and **both** entries are load-bearing: the stock
  `libtpc_kernels.so` is needed for **any** `torch.compile` region, including one with no custom op in
  it. An **enabled** disk recipe cache cannot coexist with that list — the bridge's own env-flag
  validation rejects it as "not a regular file" — and that cache's key is a **constant**, so a rebuilt
  kernel is served the *stale* recipe. Gate bit-exactly on a value that is not 0 or 1, and prove the
  gate fails. → *ch 9*

---

## Quantization landmines → *ch 3 §3.4*

- **Device fp8 is `e4m3fn` codes, bias 7 (the default; the bias is selectable per op and is part of the
  operand contract), max finite 240.** Bytes `0x78–0x7E` / `0xF8–0xFE` decode to ±inf/NaN — poison,
  created at decode, **unrecoverable**. **CPU torch disagrees** (reads them as finite up to 448): fp8
  reasoning from CPU behaviour computes the wrong poison set. Saturate at 240, not 448.
- **The standard fix — fp8 weights ×0.5, fp32 scales ×2.0 — is safe only on host/checkpoint tensors**,
  asserted structurally at load. On an HPU tensor the same expression is a no-op that preserves the
  poison (`inf × 0.5 = inf`).
- **e8m0 scale bytes: 255 is NaN** — raise at load, never substitute a value. Checkpoints store scales as
  raw `uint8` exponent bytes more often than you expect; the loader must be dtype-aware.
- **fp4 (MXFP4): fold the per-32 e8m0 scale into the fp8 exponent inside the dequant kernel** —
  `fp8_gemm_v2` has no block-scale input and is **2-D only**, so a batched expert GEMM is G calls. The
  fold is exact only within a bounded exponent range: guard at load with an assertion (zero runtime
  cost) and know the kernel's exactness boundary. The vendor `mxfp4_fused_weights` op is a drop-in only
  for a bf16-input, SILU-no-clamp, block-32, provably-in-range-scale model — **one out-of-range scale
  NaNs the whole layer**, `block_size` is inert (fixed 32). Histogram your scales before choosing either
  path.
- **A green test is not evidence about the thing you think it tests**: a gate that cannot fail on the
  unfixed code, or whose oracle is produced by the code under test, is decoration. Require RED on the
  unfixed code first.

---

## Porting doctrine → *ch 3*

- **Prefill and decode bind on different resources, and the doctrine inverts.** Decode at bs=1 is
  **weight-read bound**: compute the byte floor from the config and checkpoint headers — *not* from a
  trace — and check the largest term before optimising (in our port attention projections, not the MoE,
  are largest). → *§3.1*
- **Rank spans by removable bytes, not milliseconds.** `traffic amplification = port bytes ÷ model
  bytes`; a span already at the bandwidth envelope has only traffic headroom, and a milliseconds-ranked
  list hides exactly those spans. → *§3.1.4*
- **Replace spans, not kernels** — pull the slow contiguous span out and write different code computing
  the same result; validate per step by ulp histogram (there is **no ulp-based criterion across an
  accumulation** — unfalsifiable; use paired-differential fidelity). A signature mismatch is evidence
  about where you drew the boundary: widen the span. → *§3.0.1*
- **The MoE path is six materialising stages**, not four — the easy ones to miss are the scale gather and
  the `repeat_interleave` expand (a broadcast; even a stride-0 view materialises it).
- **Batching a sparse model is a step function of the compiled bucket, ~flat in rows inside a bucket**
  (the routed-expert-union mechanism is refuted by direct test). Crossing a bucket is what costs.
  → *§3.1.5*
- **Do not hide a mutation from the functionaliser.** A view-level ATen in-place write has a cheap
  compiled path; the same write inside an opaque op declaring `Tensor(a!)` on a large buffer materialises
  whole-buffer copies under `torch.compile`. → *§3.6.6*
- **Read a harness's environment before quoting its numbers** — a harness that sets env flags measures a
  configuration, not the product. → *§3.10*

---

## Profiling: identity is a run-time decision → *ch 5*

- **Set `TORCH_TRACE` and `DUMP_POST_GRAPHS` before the run** (plus a cold recipe cache — a warm cache
  yields an empty dump; `mkdir -p` the directory; per-rank directories under TP). A trace without them
  can never be attributed to code and no analysis repairs it: **discard and re-run**.
- **Collect CPU+HPU whenever you want attribution** — compile ids live in a CPU-side annotation; HPU-only
  drops every region span. Device durations are unaffected by CPU collection; what it corrupts ~3× is the
  host span and queue depth — the metrics that make a device-bound arm look host-bound.
- **Attribute launches (`launch_recipe`), never device events, to regions** — device events are
  asynchronous and mostly fall outside every host span. Key recipes by `recipeId`: `recipeName` is empty
  on ~99.9 % of device events. On a stock bridge, region→recipe has no key at all (our fork adds one).
- **Pre-flight every trace**: a stale `HABANA_PLUGINS_LIB_PATH` makes the profiler start fail *silently*,
  yielding a trace with no device events that parses fine. Require `kernel` + `hpu_op` categories (and
  `Torch-Compiled Region` spans for attribution) before reading anything.
- **Declare the clip** for every per-step figure, and **gate the arm before quoting it** (≥2 ops
  outstanding, ≥80 % device busy — otherwise INCONCLUSIVE, never a negative). **A zero count in a log is
  a claim about the instrument first** — check whether the emitter is env-gated before reading "never
  happened". → *§5.7*

## The iteration workflow → *ch 6*

A full model run costs minutes; a span harness costs seconds (~100×). **One attributed profile → pick a
span at a recipe boundary → compute the floor and its binding regime first (bytes / FLOPs / launches),
declared in writing → capture the span's bytes (inputs *and* state — KV cache, block tables, routing;
audit against the recipe's own tensor table) → gate on correctness *running the compiled region* → check
the harness is a proxy (recipe count, node mix, op counts, union vs. the profile) → iterate reading time,
union, op count, launch count and busy fraction together → confirm in situ once per span.**

- A span smaller than the harness's own overhead measures the harness; a whole decode step is a valid
  span for *integration*, not exploration. Choose by share of the step, not by how cleanly a span
  extracts.
- **No tensor caches and no identity keys inside traced code** — a cache hit inside a traced region is a
  baked constant (the input is silently ignored), and `id(tensor)` / `tensor is x` is either a crash or
  *arbitrary* under dynamo. Thread tensors as graph arguments. A bit-exact gate on self-consistent
  inputs cannot detect this: the required case class is **same shape, different data**. The negative
  control must fail on the pre-fix code.
- Eager is not a cheaper version of the compiled program — it is a different one (ch 6);
  a gate that never executes the compiled path cannot see that the compiled path exists. A confirmation
  that comes back flat is a result: the span was not on the critical path, or the change moved cost into
  the host term.

---

## The SRAM → *ch 7*

**You cannot place a tensor in SRAM from your own code — placement is the graph compiler's decision.**
`synDeviceMalloc` at an SRAM address returns `synSuccess` and hands back DRAM; a TPC kernel has 32-bit
pointers, so a hardcoded 64-bit address compiles to a *local-memory* access that appears to work and can
fault the card asynchronously; `AuxTensor.sramAlloc` is a dead field.

- **Eligibility is graph structure, not requests**: a tensor needs a *real compute producer* (a view does
  not qualify; one elementwise pass does), and construction decides — a single `bmm` gets its tile sliced
  into SRAM while per-expert GEMMs plus a `cat` silently get nothing, same shapes.
- **Populate or observe**: host DMA into the aperture works if the staging buffer comes from
  `synHostMalloc` (plain heap returns code 34 — a statement about the host side, not the device); a TPC
  can read/write a data-free **aux tensor** placed in SRAM under the scratchpad knob.
- **Observe, do not request**: gate on the placement line for the *specific operand*, on `maxRange.size`,
  and on `workspace_size` (256 B = nothing materialised; more = a DRAM copy). The GC log is asynchronous
  and truncated at process start — poll a completed log delta or trust nothing.
- **What the pool is worth is MEASURED on the TPC path, and it is small:** **+10.2 %** at 4 MB with
  three reps per arm and **non-overlapping** distributions, and **1.18×** at the highest rate seen
  (32 MB, one SRAM rep against the best of three DRAM runs — best-vs-best). Both arms land within
  ~10 % of the same ~2.3–2.4 TB/s per-core band, which points at the **engines**, not the memory, as
  the binding constraint (**INFERRED** — one SRAM point cannot settle it). The MME path — the only
  engine with the appetite to reach the pool's bandwidth — is unmeasured, and the compiler cannot yet
  feed it from SRAM without a staging copy. Do not anchor future investigation on any point figure.
- **Safety: `synMemCopyAsync` does not validate addresses — copying past the 48 MiB aperture faults the
  card** (measured, DFA, hard reset required). Bound every sweep by `SRAM_BASE + SRAM_SIZE` read at
  runtime; check uncorrected-error counters before and after.

---

## Devices, pinning, processes → *ch 3 §3.7*

- **One card has four disagreeing numberings** (module id, `hl-smi` index, local torch index, display
  row). Reason and pin in **module ids**; a module is acquired **exclusively** — a pin set must be a
  partition, and a second requester of a held module fails loudly. `HLS_MODULE_ID` is the pin (read
  lazily at acquire; settable after import but before the first HPU tensor); `torch.hpu.set_device(i)` as
  the first HPU call writes it. **`HABANA_VISIBLE_DEVICES` set inside a container is inert** — the
  container's device flags already chose the cards.
- **No self-report API is reliable** (`current_device`, `tensor.device` all report 0); read the module-id
  env var. Unset = unpinned, not module 0.
- Concurrency between HPU jobs perturbs timing little, but **stagger launches ~2 s** (simultaneous starts
  can lose a worker to the acquire race) and **never sample `hl-smi` during startup/teardown** (transient
  full-memory readings).
- `synStatus=8` and `synStatus 26` are **not diagnostic** — each arises from several causes. Retry, raise
  the GC log level, and bisect with `synchronize()` markers behind a default-off flag (swept before
  commit). Never `pkill -f` on a shared box.

---

## Dead ends — do not re-walk these → *ch 3 §3.8*

Two HPU streams for overlap · region coalescing alone · launch/sync/recipe counts as the lever ·
"remove the `cat`" · pre-dequantizing experts at load (HBM-blocked) · re-expressing a fused kernel in
native torch ops · "avoid custom ops" as a heuristic · a node-count census as a cost model (metadata
nodes are free on the device; 1–3 µs integer nodes can dominate — rank by engine × count × mean
duration) · `torch.stack([a, a], dim=-1)` in a benchmark (the same tensor twice) · profiling with CPU
activity when you need host/queue metrics · `**`-globbing `.graph_dumps` (hidden directory) · reaching
SRAM from user code · treating a `dynamic=True` request as a fix · a detector whose allow-list excludes
the default it must see.

---

## Measurement discipline — the short list → *ch 2 §2.9*

1. **Durations are unions; never sum event durations.** Recipe unions may be summed (recipes are serial);
   event durations may not. Sum per device, never across TP ranks.
2. **A marker duration is not an execution window** — merge per node per tid first.
3. **Never infer a mechanism from an identifier** — a name, guid, filename, or single record names
   nothing. Parse structured strings by field; cross-tabulate your filters.
4. **A result is overturned by adding a control, not a measurement** — a collinear corpus cannot be
   separated by more fits. Prefer the arm that differs in exactly one dimension. Pre-commit a falsifier
   for the claim the arm is about — and make sure the arm actually varies the quantity it is about.
5. **A precondition failure is INCONCLUSIVE, never a negative** — the achieved rate tracks device busy.
6. **Collect only the profiler activity you need** — CPU collection inflates host span ~3×; a host
   blocking in the allocator looks host-bound.
7. **Two counts differing by exactly the step / rank / layer count are one measurement at two
   normalisations** — compare the moments, not the counts.
8. **Two views of the same quantity are not two instruments** — name what each reading is a view of; an
   instrument that overwrites its own evidence is not an instrument (artifact paths carry run identity).
9. **State a number's derivation, not just the number** — a derivable figure can be re-checked in a
   minute.

---

**Tagging convention** as throughout: **MEASURED** / **source read** / **ASSUMED** / **INFERRED** /
**UNKNOWN** — see [README](README.md). The facts above are general to Gaudi 2 in *shape*; the quoted
figures are from one measurement campaign (torch 2.13 / Habana 1.24.2, one 4-AIP box), the
workload-specific ones **[DSv4F]**. Where a number matters to your decision, read the chapter and the
conditions it was measured under — not this file.
