# Chapter 1 — Gaudi/HPU Basics

This chapter is the hardware model the rest of the manual builds on: what executes your code,
how many of each thing there is, how fast each can move data, and how to tell what actually
happened. Nothing here is specific to one model or one container.

**Read the [tag legend](README.md#how-to-read-this-manual) before using any number here.** Every
non-obvious claim carries **MEASURED / ASSUMED / INFERRED / UNKNOWN**. Where we did not measure
something, it says **UNKNOWN** and says what would settle it — that is deliberate, and it is the
most useful property of this document. Three separate errors behind this manual came from a
number that had been inherited from a summary and never checked.

Example code for this chapter lives in [`examples/01-basics/`](examples/01-basics/).

---

## 1.1 The three engines

A Gaudi 2 has **three** kinds of compute/data-movement engine. They are physically distinct
units, they are scheduled independently, and **which one your op lands on is not something you
choose at the call site** — the Synapse graph compiler assigns it during lowering.

| engine | count | what it does | how you get work onto it |
|---|---|---|---|
| **TPC** | **24** | general-purpose VLIW cores. Elementwise ops, reductions, gathers/scatters, casts, softmax, and **any custom kernel you compile yourself** | native elementwise/reduce ops; custom TPC kernels via `GC_KERNEL_PATH` |
| **MME** | **4** | fixed-function matrix-multiply (systolic) engines | a matrix-multiply node; the GC lowers it there |
| **DMA** | **8 tids** (see §1.2) | data movement between HBM and on-die SRAM, and H2D/D2H | `DmaMemcpy`/`DmaBroadcast`/`DmaMemset`/`DmaTranspose` — mostly inserted by the compiler |

**MEASURED** — the counts are stable across 141 traces, two days, four worker processes, with
**zero** tid-window shifts. The tid namespace is **per worker process, not per device**: two
traces whose device events carry *different* device labels use the *identical* tid block. That is
why the windows are reliable, and it is also why they are not a property of "device 0".

**MEASURED** — every device event also carries Synapse's own engine label in
`args["HW event name"]`. Coverage is **98.85 %**, and it agrees with the tid windows on **100 %**
of device work across **21.9 M events**. The 1.15 % without a label are exactly the host spans,
which correctly have no engine. **Use this field** — it is the runtime's own answer, and §1.8
explains why a *name heuristic* is not a substitute.

> **Reading trap.** A node's `guid` (the string you pass to `synNodeCreate`) selects an
> **operation / node type**, not an engine. `gemm` is a GC-native node-type name and does land on
> MME — but that is the compiler's node-type→engine mapping, not the GUID naming hardware.
> Corroboration: `matrix_multiply_dram_f32` exists as a **TPC kernel**, so "the matmul GUID" is
> not unique. **MEASURED**

### The tid windows

| engine | tid range | marker |
|---|---|---|
| TPC | `7021–7046` (24 tids; `7027` is skipped) | `TPC_SPU_START_TO_SPU_HALT` |
| MME | `6050, 6060, 6070, 6080` | `MMEH_WB1_MON_TS_BIT1` / `MMEH_WB0_MON_TS_BIT1` |
| DMA | `4052, 4053, 4062, 4063, 4072, 4073, 4082, 4083` | `DBG_DMA_TRC_WR_DATA_LAST` |

**These windows are a convenience, not the ground truth.** They are stable on every trace we
have, but the DMA window in particular is **correct by accident**: its upper bound truncates a
stride-10 channel series one channel before tids that are *already in use* by another path. If
the hardware ever uses the next channel, a window cannot separate the two, whereas
`args["HW event name"]` plus "is this event in a compiled recipe" can. **MEASURED**

---

## 1.2 The DMA subsystem: eight tids, four cores, five DRAM/SRAM lanes

This is the part of the machine that cost us the most time, so it gets its own section.

The DMA tids are **8, not 5**, and they are laid out as **4 cores × 2 lanes**:

| tid | queue |
|---|---|
| 4052 / 4053 | `[D0] Internal DMA 0` / `[D0] Internal DMA 1` |
| 4062 / 4063 | `[D1] Internal DMA 0` / `[D1] Internal DMA 1` |
| 4072 / 4073 | `[D2] Internal DMA 0` / `[D2] Internal DMA 1` |
| 4082 / 4083 | `[D3] Internal DMA 0` / `[D3] Internal DMA 1` |

**MEASURED** — the trace's own `thread_name` metadata enumerates all eight by name.

**Of the eight, five are DRAM/SRAM lanes** (`4052, 4053, 4062, 4063, 4072`) and **three are
HCL-dedicated** (`4073, 4082, 4083`) — HCL being the inter-device communication path.
**INFERRED / under test:** that split comes from the queue names, and it is what makes
"**5 DMA queues**" and "**8 DMA tids**" *both correct statements about different things*:

- **"8 tids"** is the right answer to *"how many DMA queues does the profiler show me?"* — the
  full enumeration, including the HCL-dedicated lanes.
- **"5 queues"** is the right answer to *"how many lanes carry HBM↔SRAM traffic?"* — the
  DRAM/SRAM pool that your `DmaMemcpy`-shaped traffic actually uses.

A separate count discrepancy is a **record-keeping artefact, not a mechanism**: the compiler
library exposes `GAUDI_QMAN_ID_DMA0..7` — **eight queue-manager ids**. **MEASURED**

> **Caveat added after the vendor taxonomy below was checked.** Those eight QMAN ids **coincide
> numerically** with the profiler's eight DMA tids, but the two are **different numberings**, and
> the relationship between them is **not established**. The profiler tids are a **per-process**
> thread numbering (see §1.1); the queue ids are the hardware's own enum, which as the next
> subsection shows contains **32 EDMA queues**, not 8. So do not read "eight QMAN ids" as
> confirming "eight DMA queues" — that agreement is a coincidence of small integers, and the
> question of which physical queues the profiler's tids correspond to is open (§1.12, item 4).

### The vendor's own queue taxonomy — three families, not two

The hardware's queues are grouped into families **named by purpose**, and there are **three** of
them — not the two ("host" vs "device") we had been using informally:

| family | what it is | enum prefix |
|---|---|---|
| **PDMA** | **host PCIe** (H2D / D2H) | `GAUDI2_QUEUE_ID_PDMA_*` |
| **EDMA** | the **device's internal DMA** — the DRAM/SRAM lanes that carry `DmaMemcpy`-shaped traffic | `GAUDI2_QUEUE_ID_DCORE{n}_EDMA_*` |
| **NIC** | **RoCE v2 / HCL** — inter-device communication | `GAUDI2_QUEUE_ID_NIC_*` |

- **Source (public, checkable):** the upstream Linux kernel's habanalabs driver — the
  `GAUDI2_QUEUE_ID_*` enum in `include/uapi/drm/habanalabs_accel.h` (lines **200–463** in current
  upstream `master`; line numbers move between kernel releases, so check the symbol, not the line).

The driver separates them with a **predicate**, not a convention:

```c
static bool gaudi2_is_edma_queue_id(u32 queue_id)
{
	switch (queue_id) {
	case GAUDI2_QUEUE_ID_DCORE0_EDMA_0_0...GAUDI2_QUEUE_ID_DCORE0_EDMA_1_3:
	case GAUDI2_QUEUE_ID_DCORE1_EDMA_0_0...GAUDI2_QUEUE_ID_DCORE1_EDMA_1_3:
	case GAUDI2_QUEUE_ID_DCORE2_EDMA_0_0...GAUDI2_QUEUE_ID_DCORE2_EDMA_1_3:
	case GAUDI2_QUEUE_ID_DCORE3_EDMA_0_0...GAUDI2_QUEUE_ID_DCORE3_EDMA_1_3:
		return true;
	default:
		return false;
	}
}
```

- **Source (public, checkable):** the upstream Linux kernel's habanalabs driver —
  `drivers/accel/habanalabs/gaudi2/gaudi2.c:2661-2673` (`gaudi2_is_edma_queue_id`).

**The EDMA geometry is `2 engines × 4 channels` per DCORE** (`EDMA_{0,1}_{0..3}`), across four
DCOREs — **eight queues per DCORE**, of which the profiler shows us a subset. **This is a design
parameter, and it is exactly what our "five DRAM/SRAM lanes" should be checked against**: five is
a *measured subset* of a *documented* eight, not a contradiction of it.

**What this does to our earlier reading — it SUPPORTS it, and sharpens it, but does not settle
it.** The HCL-dedicated tids are neither host PCIe nor `compute_edma`, and the hardware genuinely
has a **third** family (NIC) for that role — so the three-way split is not something we invented
to explain an anomaly. But the driver's own queue enum gives the family *taxonomy*, not the
**tid→queue mapping**, so **which** family tids `4073/4082/4083` belong to remains open (§1.12).
Read the claim as "supported", not "settled".

> **Why this cost us time.** We had been treating "the DMA engine" as a single serial resource
> with a single bandwidth number, and had been attributing the host-PDMA tids to "host PCIe"
> on a rule that did not survive contact with the lane layout. The lane split above **splits**
> that earlier ruling: it covered tids `3090/3091/4082/4083` and called them all host; on the
> lane layout **`3090/3091` remain host**, while **`4073/4082/4083` are device lanes** (the
> HCL-dedicated ones). **Not fully settled** — see §1.12.

### One transfer at a time per lane — but five lanes in parallel

- **One lane runs one transfer at a time.** **MEASURED**
- **The subsystem as a whole is a parallel pool.** Measured **Σ/union = 2.79–2.97×** over lanes
  `4052/4062/4072` across **all 14 traces** we have (2.97× on the most loaded one), with
  **symmetric** pairwise overlap — no lane is special. **MEASURED**
- ⇒ **the aggregate DMA ceiling is up to 5× a single lane**, because the DRAM/SRAM pool has five
  lanes. **INFERRED** (the 2.79–2.97× figure was measured over three lanes; the 5× ceiling is
  the extrapolation to all five).

Read the 2.79–2.97× correctly: it is the **achieved** parallelism on our traffic, not the
ceiling. If your transfers serialise — because they depend on each other, or because you issue
them from one stream in an order that forces it — you get one lane's throughput and the pool buys
you nothing.

### There is no op → queue mapping

This surprises people, so state it plainly: **the queue a transfer lands on is not a property of
the op, and not a property of the node.**

- All five device DMA families (`DmaMemcpy`, `DmaTranspose`, `DmaBroadcast`, `DmaMemset`,
  `StridedDmaViaTranspose`) use **all five lanes, with counts identical to the digit** — e.g.
  `DmaMemcpy` appears **1,185,712** times on *each* of `4052/4053/4062/4063/4072`.
- The same holds at the **execution** level: **160 of 160** `DmaTranspose` executions touched all
  five tids. **MEASURED**

⇒ The lane is a **per-transfer round-robin choice** made by the driver. **MEASURED.** The only
residual question is the exact round-robin rule, which lives inside the runtime library and is
**not** something you can control from a graph. **UNKNOWN** (would be settled by reading the
driver's queue-selection path).

**Practical consequence:** you cannot steer a copy to a particular lane, so you cannot hand-tune
lane balance. What you *can* do is give the pool enough independent transfers to keep several
lanes busy — which is a property of your graph's **dataflow**, not of any op you write.

---

## 1.3 The op → engine map

| op / pattern | engine |
|---|---|
| `DmaMemcpy`, `DmaBroadcast`, `DmaMemset`, `DmaTranspose`, `StridedDmaViaTranspose` | **DMA** |
| `fp4_dequant_fwd_gaudi2`, `gather_nd_mxnet_fwd_*`, `gather_fwd_*`, `gather_elements_fwd_*`, `scatter_nd_onnx_fwd_bf16`, `bitshift_fwd_*` | **TPC** |
| `fused_kernel_0x<HASH>_<IDX>_<dtype>` (the compiler's fused TPC nodes) | **TPC** |
| casts, reduces, softmax, `mult_*`, `add_*`, `sub_*`, `div_*`, `clamp_*`, `where*`, `exp/log/sqrt/rsqrt_*`, `constant_*`, `range_i32`, `memcpy_*`, `bitwise_*`, `generate_bitonic_chunks_*` | **TPC** |
| `GEMM` | **MME** |
| `synchronizeEvent`, `enqueueWithExternalEventsExt`, `compileGraph`, `eventRecord`, `streamWaitEvent`, `hcclAllReduce`, `allocateDeviceMemory` | **HOST — no engine** |

**MEASURED.** Note the last row: these are the *only* events that carry no `HW event name` at
all, which is why the marker's 98.85 % coverage is not a limitation — the missing 1.15 % is
exactly the host bookkeeping, and "no engine" is the correct answer for it.

### A single compiled recipe routinely contains all three engines

This is the single most important structural fact in the chapter, because it means
"which engine does this recipe use?" is usually the wrong question.

Of **4,765** distinct recipe handles across the corpus:

| engines present in the recipe | handles |
|---|---|
| exactly 1 | **3,301** |
| exactly 2 | **475** |
| **all 3** | **989** |

**MEASURED.** One observed example, a recipe built from two independent ops in one
`torch.compile(fullgraph=True)` region — nine nodes, **DMA ×3, MME ×3, TPC ×1**, plus two
logical wrappers:

```
id   engine  logical  type         guid                          exec  bundle  origin_nodes
453  None    True     Split        split_logical                 0     0       [394]
455  DMA     False    DmaMemcpy    memcpy_dma                    1     0       [394]
447  MME     False    GEMM         gemm                          2     0       [394]
456  DMA     False    DmaMemcpy    memcpy_dma                    3     0       [394]
457  DMA     False    DmaMemcpy    memcpy_dma                    4     0       [394]
448  MME     False    GEMM         gemm                          5     0       [394]
449  MME     False    GEMM         gemm                          6     0       [394]
454  None    True     Concatenate  concat_logical                7     0       [394]
444  TPC     False    fused_kernel_0xDD31D959_1_f32  fused_..._1  8     None    [391,392,393]
```

**MEASURED.** See [`examples/01-basics/dump_post_graphs.py`](examples/01-basics/dump_post_graphs.py)
for how to produce this yourself.

> **Do not count nodes naively.** Two traps, both measured: (a) `is_logical=true` marks **only
> the wrappers** (`Split`, `Concatenate`, `StaticReshape`, `Identity`) — all with
> `engine=None`; (b) the real multiplication goes the *other* way — `origin_nodes` is
> **many-to-one in both directions** (one origin node expands into 9 graph nodes, while a TPC
> node carries three origins), and **bundle slicing** turns one IR node into several physical
> engine tasks. **One IR node ≠ one engine task.** **MEASURED**

### How a recipe gets its boundaries

A compiled region is split into recipes at **placement** boundaries, not at op-name or
namespace boundaries:

> **One recipe per maximal contiguous run of `hpu_cluster`-placed nodes.** A node the
> shared-layer op validator rejects is placed **eager**, is not clusterable, and **splits the
> run into two recipes**.

The rule is therefore ***"cannot be clustered"*, not *"is custom"*** — a **native** op that fails
the same validator splits identically. Verified on 24/24 arms, from both the trace and the live
graph. **MEASURED** (Chapter 2 develops what this does to concurrency.)

---

## 1.4 The compilation model — static shapes, specialized integers

> **This section develops [Chapter 0, facts 1 and 2](00-tldr.md).** The TLDR states them in one
> line each; here is the mechanism. Read it **before §1.6 (bandwidth)** and **§1.7 (cost models)**
> — both assume a compiled graph, and both are easy to misread without this.

Everything that runs on the device runs as a **compiled graph**, and **the graph is compiled for
the exact values it saw**. There is no runtime interpretation step and no dynamic dispatch.

### 1.4.1 There are no dynamic shapes

Every tensor shape is **baked into the compiled graph**. Asking for dynamic shapes does **not**
produce a shape-polymorphic graph — the request is **turned into static compilation**. So you
cannot pass a dynamic slice bound or a dynamic shape and have one graph serve several cases.

**ASSUMED (vendor).** Stated here as the toolchain's behaviour; it is consistent with everything in
§1.6–§1.9, and it is the standard account of this class of accelerator.

**Read it as a design constraint, not a limitation.** The consequence is that you work with a
**shape set**: you decide which shapes the program must serve, and you accept that **each new shape
is a new program**. That is a decision you make once, deliberately — and it is *why* padding and
bucketing are normal engineering techniques here rather than hacks.

### 1.4.2 Integers are specialized exactly like shapes

**This is the fact that surprises people, and it is the most consequential one in this chapter.**

A Python `int` used inside traced code — a **layer index**, a **per-layer byte offset**, any
constant — becomes part of the compiled graph's **guard set**. Change the value, the guard fails,
and **a new graph is compiled** — exactly as a new shape would.

> **There is no dynamic-argument escape hatch for integers.** There is no "pass this int at
> runtime" mode that keeps one graph. An `int` in traced code is a compile-time constant in the
> same sense a shape is.

**The consequence, stated as a rule:**

> **Code that computes an offset from a loop index has written one program per iteration.**

A loop that walks layers and computes `offset = layer * stride` *inside* the traced region does
not produce one graph with a runtime offset. It produces **one graph per layer**, each with its own
baked-in offset. The loop is unrolled **into the compilation**, whether or not it was unrolled in
the Python.

**INFERRED** — this follows directly from the guard-set mechanism above. *(We have not, in this
chapter's sources, isolated a single integer and counted the graphs it produces; that is recorded
in §1.12.)*

### 1.4.3 The graph count multiplies: layer × constant × shape

Put the two facts together and the arithmetic follows immediately:

```
graphs  ≈  (number of distinct shapes)  ×  (number of distinct int constants in guards)
```

and the int term is usually **the layer count**, because layer-indexed code is the normal shape of a
transformer.

> **An N-layer model is an O(N × buckets) specialization case BY CONSTRUCTION, not by bug.**
> **A 43-layer model is an O(10²)-graph model before anything is done wrong.**

**This is the practical value of the section.** A large graph count is **normal**, and it is **not
evidence of a mistake** until it has been checked against this arithmetic. The two questions worth
asking:

- **Is the count near the prediction, or far above it?** Far above means something is specializing
  on a value you did not intend — a data-dependent constant, or a shape that varies when you
  believed it was fixed.
- **Did a change meant to coalesce actually reduce the count?** Coalescing regions is a lever
  (Chapter 2) — but a coalesced region spanning G layers **still needs G graphs** if it guards on
  each layer's constants. **Check the count; do not assume it fell.**

**INFERRED** from §1.4.1–§1.4.2. *(The 43-layer figure is this manual's reference workload, not a
general law.)*

> **Two different counts — do not conflate them.** §1.3's rule (*one recipe per maximal contiguous
> run of clusterable nodes*) is about **placement**: how many recipes a region splits into. This
> section is about **specialization**: how many compiled *variants* exist. They are independent
> axes, and they multiply. A region that splits into 3 recipes and guards on 43 layer constants
> does not have "3 graphs" — and a *recipe count* is not a *graph count*.

### 1.4.4 Where this bites in practice

Two mechanisms, and they compound:

1. **Regional compilation.** A compiler may compile **each decoder layer as its own function**
   rather than one function for the whole stack. That is a reasonable trade for compile time and
   locality — but it makes the per-layer specialization **structural**: the layer boundary *is* the
   graph boundary.
2. **Per-layer constants computed in-graph.** When a layer computes something from **its own
   index** inside the traced region — an address, a page offset, a rotation angle — that value
   enters the guard set and **specializes that layer's graph to that layer**. Two layers running
   *identical* code with different constants do **not** share a graph.

**Together, these mean the O(N × buckets) count is not something you can compile away by writing
the Python differently.** It is the natural consequence of per-layer constants in a
statically-compiled system. The available levers are to **reduce the number of distinct constants**
(make a value uniform across layers where you legitimately can) and to **reduce the number of
distinct shapes** (bucket deliberately) — not to hope the compiler finds the sharing.

**ASSUMED (vendor) / INFERRED.** Stated at the level of *what the compiler does*; deliberately not
tied to any particular implementation.

> **A caution about measuring it.** Count **distinct compiled graphs actually executed**, and do
> not identify a graph by a **per-run handle**: handles are per-process and are not stable across
> runs or ranks, so comparing two runs by handle identity shows you differences that are not
> specialization. Compare **counts and stable graph identities** instead.

---

## 1.5 Memory

| level | size | scope | basis |
|---|---|---|---|
| **HBM2E** | **96 GiB per HPU** | per card | **MEASURED** |
| **SRAM** | **48 MiB** (Gaudi 2) | on-die, **shared** — MME/DMA staging | **source read** |
| **TPC local memory** | **1 KB scalar + 80 KB vector** | **per TPC core, private** | **ASSUMED** (vendor documentation) |
| **L1 / L2 data cache** | — | **there is none** | **ASSUMED** (vendor documentation) |

*(Rows tagged **ASSUMED** are taken from documentation or a vendor constant; where the number comes
from a **named source file with a path and line** it is tagged **source read** instead, as the SRAM
row is. Neither is a measurement on our hardware. Each is cited to a public source — a URL or an
upstream file path — so you can check it. See §1.11.)*

**HBM is 96 GiB per card**, with an idle baseline of ~768 MiB. **MEASURED**

**SRAM is 48 MiB on Gaudi 2.** `SRAM_SIZE = 0x3000000`. For scale across the family the same
constant is **Gaudi 1 = `0x1400000` (20 MiB)** and **Gaudi 3 = `0x10000000` (256 MiB)**.

- **Source (public, checkable):** the upstream Linux kernel's habanalabs driver —
  `drivers/accel/habanalabs/include/gaudi2/gaudi2.h:41` (`SRAM_SIZE`).

The *mechanism* was already confirmed from the compiler's own output — recipe-dump tensors carry
`"allocation": "SRAM"` with a distinct `sp_index` for MME operand slices — and now the *capacity*
is sourced too.

> **A number we carried unverified turned out to be right.** Our own notes had *"48 MB on-die
> SRAM"* marked **"Unverified here; not load-bearing"** by its own erratum. It is correct. That is
> worth one line because it is the *opposite* of the three errors the UNKNOWN note at the top of
> this manual describes: an inherited number that **survived** checking. Two things follow.
> **(a)** Marking it unverified was still right — the marking is what made it checkable.
> **(b)** Building on an unverified number *before* checking it would still have been wrong, even
> though it would have happened to work. The habit is what is being taught, not the outcome.

**There is no L1/L2 cache.** This is a *positive* finding, not a gap. The documented memory
hierarchy is **HBM2E → 48 MiB shared SRAM → per-TPC local memory**, with **no hardware-managed
cache level between them**. The vendor architecture page names three subsystems (compute, memory,
networking) and two compute engines (MME + TPC); searches of the documentation, the architecture
overview and the whitepaper pages return the same three levels and never a cache.

**What the TPC does have is *local memory*, and it is a scratchpad, not a cache:**

> "Each TPC processor has its own instance of local memory. Each TPC can only access its own local
> copy. That is, TPC A cannot access TPC B local memory. … **Scalar local memory: Size is 1 KB.**
> … **Vector local memory: Size is 80 KB. If the program utilizes special functions such as tanh,
> sin, or cos, only 16 KBs are available.**"

- **Source (public, checkable):** the vendor's TPC user guide —
  `https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/Processor_Architectural_Overview.html`

> ### ⚠ Do not conflate the SRAM with the TPC local memory
>
> They are different levels, with different sizes, different owners and different lifetimes. A
> bandwidth model that merges them will be wrong.
>
> | | **48 MiB SRAM** | **TPC local memory** |
> |---|---|---|
> | **scope** | one on-die pool, **shared** by every engine | **private to one TPC core** (24 cores ⇒ ~1.92 MB aggregate, but **not addressable across cores**) |
> | **who schedules it** | the **compiler** — MME/DMA staging, SRAM slicing | the **kernel programmer**, explicitly |
> | **size** | 48 MiB | 1 KB + 80 KB **per core** |
> | **is it a cache?** | no | no |
> | **visible how** | `"allocation": "SRAM"` in the recipe dump | only from inside a TPC kernel |
>
> **Neither is a cache, so there is no automatic reuse anywhere in this hierarchy.** Data reuse
> must be **engineered** — tiled into local memory, or staged through SRAM — or it does not
> happen. That is the single most important consequence of this section for a bandwidth model, and
> it is why the figures in §1.6 are *stream* bandwidths rather than cache-resident ones.

---

## 1.6 Bandwidth — and the size each figure is valid at

**Bandwidth on this hardware is size-dependent. A bandwidth number without its size is not a
number.** This section is therefore a table with a size column, and it is the section most likely
to save you from a wrong answer.

| figure | value | **valid at** | basis |
|---|---|---|---|
| spec nominal | 2,450 GB/s | spec | **ASSUMED** |
| **TPC stream envelope** | **2,147 GB/s** (86 % of nominal) | **≥ 134 MB** | **MEASURED** |
| production tile | 2,200 GB/s | 33.55 MB burst | **MEASURED** |
| **DMA engine** (flat 1-D copy) | **1,775.1 / 1,776.6 / 2,021.3 / 2,028.4 GB/s** | **16 / 64 / 256 / 1024 MB** | **MEASURED** |
| TPC `memcpy_u8` D2D copy | 1,014 / 1,704 / 1,888 GB/s | 64 / 256 / 1024 MB | **MEASURED** |
| single DMA node, C++ bench | **2,023 GB/s** *(1,646 retracted — see below)* | 512 MiB operand, flat 1-D (2-D control agrees to 0.03 %) | **MEASURED** |

### The DMA engine is a two-regime curve

| size | DMA engine | vs the TPC stream envelope (2,147) |
|---|---|---|
| 16 MB | **1,775 GB/s** | −17.3 % |
| 64 MB | **1,777 GB/s** | −17.2 % |
| 256 MB | **2,021 GB/s** | −5.9 % |
| **512 MiB** | **2,023 GB/s** | −5.8 % |
| 1024 MB | **2,028 GB/s** | −5.5 % |

Every point is **DMA-engine**, measured as `2·bytes / device union`, and every point **passed its
precondition**. **MEASURED.**

**The curve is monotone in size, as it should be**, and the summary is two-regime:

> **~1,776 GB/s at 16–64 MB (82.7 % of the TPC envelope) · ~2,025 GB/s at ≥256 MB (94.3 %).**

> ### ⚠ RETRACTION — the 512 MiB point used to read 1,646 GB/s
>
> The 512 MiB row was **published as 1,646 GB/s** and has been **replaced by 2,023 GB/s**. The old
> figure was **wrong for two independent reasons** (both below). It is recorded here rather than
> silently swapped, because a corrected claim should show its history — see *"how a wrong curve
> point survives"* below.
>
> **What is unchanged:** every other point, the two-regime shape, and the direction of every
> comparison.

Two things follow, and **you must say which comparison you mean**:

- **Against the TPC *stream* envelope**, the DMA engine is **~17 % slower at ≤64 MB** and
  **~5.7 % slower at ≥256 MB**. This is a real, size-dependent penalty, and the ≤64 MB band is
  the band a small decode step actually lives in.
- **Against the TPC *memcpy* path**, the DMA engine is **faster at every size measured**
  (1,776 vs 1,014 @64 MB; 2,021 vs 1,704 @256 MB; 2,028 vs 1,888 @1024 MB).

A blanket "routing bytes through the DMA engine costs 14–24 %" was written down during this work
and is **withdrawn** — it was neither size-scoped nor direction-scoped. **MEASURED**

### ⚠ The trap we fell into: `torch.copy_` is NOT the DMA engine

> **An eager `y.copy_(x)` lowers to a TPC kernel (`memcpy_u8`), not to the DMA engine.** Measured:
> **768 TPC kernel events, zero `compute_edma` events.** Under `torch.compile` the same op *does*
> go to the engine. **MEASURED**

A row reading *"D2D copy (DMA engine) | 1,014 / 1,704 / 1,888 GB/s"* was carried in two of our
notes on the strength of its label. Those numbers are **TPC** numbers. Any "DMA engine" figure
taken from `torch.copy_` in eager mode is TPC data, and the error is invisible from the label
alone — which is exactly the failure mode §1.8's reading trap describes.

**How to tell them apart:** count events by `args["HW event name"]`, not by the op name you
called. A TPC copy shows up on the TPC tids; a real DMA transfer shows up on the DMA tids. See
[`examples/01-basics/trace_engines.py`](examples/01-basics/trace_engines.py).

### Why the 512 MiB point was wrong, and how it survived

The old 512 MiB figure (**1,646 GB/s**) was wrong for **two independent reasons**. Either one alone
would have made it wrong; neither was checked.

1. **A per-repetition wall time was read as a device time.** The C++ bench times **`launch → sync`
   inside the repetition loop**, so its **622.1 µs** median is a *host* figure that carries the
   per-repetition launch + sync cost. The device work is **530.5 µs** — and the tell is the
   statistic this manual keeps insisting on: the DMA **union is identical (530.5 µs) whether you
   sync once or per repetition**, because the union is blind to the host's sync pattern. The
   ~90 µs difference is exactly the overhead the bench needs (530.5 + ~90 ≈ 620 µs ≈ its 622.1 µs
   median). **That an overhead of this order is real was measured separately**, in a torch harness
   that does the same copy: **322.9 µs per rep** on *that* path. Note the size — it is **3.5×**
   the C++ bench's, so it **bounds the effect's existence without pinning the bench's number**, and
   the two must not be quoted as the same quantity. **MEASURED**
2. **An arithmetic slip in the record.** `1,073,741,824 B / 0.6221 ms` is **1,726 GB/s, not
   1,646** — a further **4.6 %** error. 1,646 corresponds to **652.3 µs**, which is none of the
   bench's min (608.1), median (622.1) or mean (635.2) — it was never a bench figure at all.

*Source: `Archive/by-date/2026-10-11/dma_open_items/OPEN-ITEMS-c2-and-512MiB.md` §2 — the
four-cell arm (`d2b.py`), the overhead harness (`d2b_ovh.py`) and the pre-committed falsifier
(`D2b_PRE_COMMITTED_FALSIFIER.md`).*

**A hypothesis that died with it.** The point had been attributed to the bench's copy being **2-D**
where the other measurements are flat **1-D**. Measured directly: flat 1-D **2,022.9 GB/s** vs
shaped 2-D **2,023.5 GB/s** — **0.03 % apart**. **The compiler flattens a contiguous 2-D memcpy.**
So the *shape* hypothesis is **dead**, and the *sync* hypothesis is **live and sufficient**.

> **How a wrong curve point survives — worth reading once.** This figure sat in the record through
> several passes. It survived because:
>
> - **a timing artefact looked like a physical result.** "This size is slower" is a *plausible*
>   finding, and the point arrived with a ready-made mechanism to blame — the bench's copy was
>   **2-D** where every other point is flat **1-D** — so nothing about it invited doubt. **The
>   mechanism was false** (0.03 % apart), and the anomaly it "explained" was not real.
> - **the number was never re-derived.** The division was written once and copied. A **4.6 %**
>   arithmetic slip is invisible to anyone who does not redo the division, and it is the **second
>   time this session a figure survived because nobody re-derived it.**
> - **two independent errors cancelled into one plausible-looking story.** The timing artefact
>   produced a low number; the arithmetic error made it lower still; and the low number was then
>   *explained* by a shape hypothesis that turned out to be false. **An explanation is not
>   evidence for the thing it explains.**
>
> The manual's rule follows: **state a number's derivation, not just the number.** A figure whose
> derivation is written down can be re-checked by a reader in a minute; a figure that is only
> quoted cannot be re-checked at all.

### The only remaining interpolation is below 16 MB — and it is not a DMA curve

Every DMA-engine point we have is **≥ 16 MB**, and the curve is now monotone across them. **Below
16 MB we have no DMA-engine measurement at all.**

The only evidence in that range is a **TPC / GEMV** cliff — small-tile matmuls lose bandwidth
sharply. **That is a TPC phenomenon and must NOT be read as a DMA-engine curve.** The two engines
have different behaviour at small sizes, and the DMA engine passes its precondition at **16 MB**
with **1,775 GB/s** — i.e. there is **no measured small-size collapse on the DMA engine**.

> **The trap, stated plainly:** if you need a DMA-engine number for a transfer **below 16 MB**, we
> do not have one. Do not substitute the TPC cliff for it. Either measure it, or bound the
> conclusion to "≥ 16 MB".

### Consequence for any bandwidth-bound floor

A floor that costs bytes at **2,147 GB/s** is using the **TPC stream** envelope. Any byte actually
carried by the **DMA engine in the 16–64 MB band** must be costed at **~1,776 GB/s — about 17 %
slower**. If you build a roofline and then route the traffic differently from how you costed it,
the roofline is wrong in the direction that flatters the change.

### Why there is no vendor bandwidth table in this section — a deliberate choice

**Every per-engine bandwidth figure above is ours.** We looked for a vendor table to import and
found none: the public vendor material — the `docs.habana.ai` documentation set, the Gaudi 2
whitepaper and the published datasheet — gives the **aggregate** HBM figure (2.45 TB/s, and see the
unit-typo warning below) and **no per-engine split**: no share percentages, no bytes/cycle per
engine, no per-TPC table. The libraries shipped with the runtime (`libscal`,
`libGraphCompilerPlugin`, `libSynapseMmeReference`, the habanalabs driver) do not expose one either.

**Read that as a negative result from our own search, not as a vendor statement** — we may simply
not have found the table, and a vendor per-engine figure would be worth having. Until one turns up,
"we measured it ourselves" is not a limitation of this manual; **it is the only option.** If you see
a per-engine Gaudi 2 bandwidth figure anywhere, ask where it came from.

### ⚠ Warning: a 1000× units typo you will probably find

A vendor datasheet in circulation reads:

> "Gaudi2 integrates 96GB of HBM2e memories on chip, at **2.45GB/sec** bandwidth"

That is **GB**, where the unit should be **TB**. The figure is **wrong by 1000×**. The vendor's
documentation, the whitepaper and the launch fact sheet all say **2.45 TB/sec**. **Do not propagate
the datasheet's row.**

**Why this deserves a warning box:** it is the single most likely number a reader will grep for
when looking up Gaudi 2 bandwidth, it is wrong by three orders of magnitude, and it is *in a
datasheet* — the genre a reader trusts most. **If a bandwidth figure looks 1000× off, check the
unit before you check the physics.**

---

## 1.7 The cost models — and their error signs

The compiler emits per-node *estimates*. They are useful, but **they are not neutral references**,
and the two we have looked at fail in **opposite** directions.

### `mme_expected_compute_cycles` — accurate

- The MME cost model emits a **cycle count** per node: **131,072** cycles for one sliced GEMM
  node, **524,288** for the unsliced node.
- It converts cycles to time using a clock it reads from its **own** HAL reader:
  `libSynapseMme.so` → `Gaudi2::MmeHalReader::getClkFreqMHz()` returns **1650 MHz** (a hardcoded
  immediate). **source read** (read out of the shipped binary).
- **1650 MHz is exactly the SoC clock the vendor tool reports**, so the agreement is a
  *validation*, not a restatement — the value comes from the binary, not from a measurement.
- **The prediction is accurate to 0.5 %.** 524,288 cycles ÷ 1650 MHz = **317.8 µs**, against a
  measured **319.6 µs** on a precondition-passing arm. Independently: 262,144 FLOP/cycle ×
  1.65 GHz = **432.5 TFLOP/s** against a measured **430.0–432.7**. **MEASURED**

*For the record, the whole clock family found in the two libraries:* base `gaudi::MmeHalReader`
**1800**, `Gaudi2` **1650**, `gaudi3` **1600**; and the **general** `Gaudi2HalReader` — a
**different library** — **2000**. **MEASURED** — *artifact:
`Archive/by-date/2026-10-11/synapse_cpp/GC_COST_MODEL_CLOCK.md` §8 (the family table, with the
`mov` immediates quoted and the `readelf -rW` vtable resolution to the MME cost model).* Note the
implication: the general reader is **21 % high** against the real SoC clock.

> **OPEN.** *If* the **TPC** cost model consumes that **general** reader rather than an
> engine-specific one, it would be **~21 % optimistic**. **Whether it does is unchecked.** The
> **DMA** model is **excluded** — a disassembly of it found **no clock call at all** — but the TPC
> model's constructor is not exported, so which reader it is given is undetermined.
> *What would settle it:* disassemble the TPC cost model and find which vtable slot supplies its
> clock — the method that settled the MME's. *(Artifact:
> `Archive/by-date/2026-10-11/synapse_cpp/GC_COST_MODEL_CLOCK.md` §10.)* **UNKNOWN / OPEN.**

### `dma_expected_duration` — a LOWER BOUND

`dma_expected_duration` is a per-node **execution-time estimate** at an **assumed ~2,080 GB/s**
peak (85 % of the 2,450 nominal) plus a small per-node constant. It contains **no queue wait, no
contention, and no launch latency beyond that constant**. **MEASURED** (recovered from the field
across four recipes).

Because it assumes an optimistic peak and excludes every source of loss, it is a **lower bound**,
and the *ratio* you compute from it is:

```
measured window  ÷  an OPTIMISTIC execution estimate
```

That framing is the reusable part, and it explains a result that otherwise looks arbitrary: a
**1.15×** reading **validates** (the estimate was nearly right) while a **5.53×** reading does
**not** (an optimistic lower bound cannot be 5.5× above reality without something else going on).

**How optimistic, measured:**

| size | model | measured | measured ÷ modelled |
|---|---|---|---|
| 64 MB | 65.111 µs | 75.55 µs | **1.160×** |
| 256 MB | 258.032 µs | 265.61 µs | **1.029×** |

⇒ **~16 % optimistic at 64 MB, ~3 % optimistic at ≥256 MB — a two-regime error, not a single
factor.** The model's error and the engine's own two-regime behaviour (§1.6) **agree in shape**,
which is a good sign for both. **MEASURED**

An earlier single figure ("1.21× optimistic") rested on a mislabelled TPC row **and on the
since-retracted 1,646 GB/s point** (`2,080 / 1,646 = 1.264`). With the corrected 512 MiB value
(**2,023 GB/s**) the model is only **~1.03× high** at that size — consistent with the ≥256 MB row
above. The single figure is **replaced** by the size-dependent pair. **MEASURED**

### The TPC cost model

The TPC side has a cost field (`tpc_tpc_cycles`) and a working-engine count
(`tpc_working_engines`, e.g. `[24,24,24]`). **The clock it uses to convert cycles to time is
UNKNOWN**, and the "general HAL reader is 21 % high" note above is the specific worry. *(Refined
2026-10-13; mechanism **corrected** the same day. The field is **gated**, not absent, and the gate is
**two-part**: `DUMP_TPC_COST_MODEL_DATA=1` **and** `ENABLE_EXPERIMENTAL_FLAGS=1` — without the latter
the config is **silently ignored** (libSynapse: *"is internal config. cannot be set without
ENABLE_EXPERIMENTAL_FLAGS"*; a sibling `DUMP_*` config writes no file without it and a 51 KB JSON with
it). **But the kernel-side half cannot be satisfied on this compiler:** the gate's own message names
`--kernel-cost`, and `tpc-clang` **2.7.33 has no such flag** — rejected as a driver option, as
`-mllvm`, and through `CCC_OVERRIDE_OPTIONS`, with the string occurring **0** times in the binary. The
nearest real cost-json flags are **inert**: no pass in the codegen pipeline at any `-O` level writes
the cost model, so they set a variable nothing reads. **MEASURED:** a dumped node carries **21 keys
whose only `tpc_*` key is `tpc_working_engines`**, and **11** stock elementwise nodes in the same dump
emit no `tpc_tpc_cycles` either. So **"not obtained"** is the accurate state. The companion field is
better understood: `tpc_working_engines` is the **dispatch width**, and it is now **exactly
predictable** from the glue's declared index space —
[Chapter 4 §4.2.8](04-execution-model.md).)*

**Be precise about which kind of unknown this is.** We tried to settle it by looking the clock up
in the shipped cost-model libraries and **could not complete the check** — so this is
*"not checked"*, **not** *"checked and absent"*. §1.12 records the distinction explicitly, and it
matters: a failed lookup must not be allowed to harden into a vendor statement. *What would settle
it:* symbol access into the cost-model libraries, or the same disassembly method that settled the
MME clock.

> **If you are writing your own TPC kernel, the unit of cost is the VLIW *bundle*, not the
> instruction** — an *issue group*, **not** the graph `bundle_index` sense of §1.3.
> [Chapter 4](04-execution-model.md) prices TPC code that way.

### The rule that follows

> **Two cost models, same field family, different error signs.** The MME model is **accurate
> (0.5 %)**; the DMA model is **optimistic, size-dependently (3–16 %)**; the TPC model's clock is
> **unknown**. **Neither may be used as a neutral reference without saying which one you mean.**
> **MEASURED**

---

## 1.8 Markers, and what a duration actually means

Every device event carries `args["HW event name"]` — Synapse's **own** per-event label. Coverage
**98.85 %**; agreement with the tid windows **100 %** over 21.9 M events. **MEASURED.** Use it.

**But the `dur` field on an event is not automatically an execution window.** The markers are not
all the same kind of thing:

| engine | marker | is `dur` an execution span? |
|---|---|---|
| **TPC** | `TPC_SPU_START_TO_SPU_HALT` | **YES** — a real start→halt span (`union ≈ Σ span`, self-overlap 0.000) |
| **MME** | `MMEH_WB*_MON_TS_BIT1` | **NOT directly** — a *monitor timestamp*; the marker self-overlaps **~4×** its *global* union (an earlier trace gave 2.7×) |
| **DMA** | `DBG_DMA_TRC_WR_DATA_LAST` | **NOT directly** — a *debug trace* beat; self-overlaps **~14.5×** |

**MEASURED.** Consequence: **every MME or DMA `ms` figure taken from raw marker durations is a
*marker* figure, not an occupancy figure.**

### The MME's inflation is entirely *cross*-tid

The MME caveat is **weaker than it first looks**, and this is worth getting right because it is
the difference between "MME timings are useless" and "MME timings are usable":

- MME marker self-overlap is **~4×** on the *global* union — i.e. across all four MME tids. (An
  earlier trace measured 2.7× on the same statistic; the 4.11–4.14× figures below are from a
  later, better-conditioned pair of arms. The conclusion is the same either way.)
- **Within each MME tid**, the markers are **serial: 1.03–1.04×**. (S=3072: global 4.14×,
  within-tid 1.04×; S=4096: global 4.11×, within-tid 1.03×.) **MEASURED**
- ⇒ the overlap is **between** tids, not within one, so **the MME global union IS a defensible
  duration** — and the marker route **is a usable rate measurement**. What it measures is the
  **achieved rate in a given recipe context**, not a hardware constant. **MEASURED**

The **DMA** caveat (**14.5×**) stands unchanged. **MEASURED**

### The correct per-queue seriality statistic

If you want to know whether a queue's record is a serial execution record, **merge per node
first**:

```
seriality  =  Σ_nodes | merge(node's events on this tid) |  ÷  | union(all events on this tid) |
              1.0  =  a serial execution record
```

Measured: `hpu_op` **1.0067 / 1.0010**; kernel `DmaTranspose` **1.0294 / 1.0300**; TPC control
**1.0000**; **MME 1.695 / 1.795 — the only non-serial channel.** **MEASURED**

> **Marker-beat packing is NOT engine seriality.** The naive statistic (summed durations ÷ union)
> measures how densely a node's marker *beats* pack inside its own window — a property of the
> **recording**, not the engine. It produced a plausible-looking ~4.5× contrast that was entirely
> an artefact, which is why it survived so long. Also note the converse trap: a **queue-occupancy
> span is serial per queue too**, so passing a seriality test does not prove you are looking at
> an execution window. **MEASURED**

### The validity gate: never quote a rate from a starved arm

Across five shapes, the achieved MME rate **tracks device busy**:

| device busy | achieved rate |
|---|---|
| 97.5 % | 430.0 TFLOP/s |
| 91.6 % | 427.6 |
| 35.7 % | 396.0 |
| 7.1 % | 332.3 |
| 2.0 % | 159.0 |

**MEASURED.** The collapse is **queue starvation**, not a hardware property. ⇒ **an arm whose
queue was empty cannot produce a negative result — it produces INCONCLUSIVE.** Assert
`pending ≥ 2` **and** `device_busy ≥ 80 %` in every timing arm, and self-test the assertion. This
rule was earned: a naive reading of the lowest point would have scored a false "the hardware is
slow" conclusion.

---

## 1.9 Profiling setup

**Device events require a host-profiler shim.** Without it, `synProfilerStart` fails *silently*
and your trace contains only `cpu_op` / `privateuse1_runtime` events — **no `kernel`, no
`hpu_op`** — which looks like "the device did nothing" rather than like an error.

- The shim is loaded from the path in **`HABANA_PLUGINS_LIB_PATH`**.
- Point it at the path the image actually ships (commonly
  **`/opt/habanalabs/habana_plugins`**, also set in `/etc/profile.d/habanalabs.sh`).
- **The failure mode is a stale path.** A path such as `/usr/lib/habanatools/habana_plugins`
  that no longer exists makes the profiler start fail silently — you get a trace, it parses, and
  it is missing every device event. **MEASURED** (this cost us real time).

**Pre-flight every trace.** Before trusting any trace, run a ~20-line probe that does a known
small matmul under the profiler and prints a histogram of event categories. If you do not see
`kernel` and `hpu_op` in the histogram, the shim is wrong and nothing downstream is worth
reading. See [`examples/01-basics/devprobe.py`](examples/01-basics/devprobe.py) — it is short
enough to read in full.

**One more trap, same family:** collecting **CPU** activity alongside HPU activity inflates the
trace's host span by roughly 3× (measured: 23.3 → 1.67 ops outstanding at the host's end), which
makes a device-bound arm look host-bound. **Profile HPU-only** when the question is about the
device. **MEASURED**

---

## 1.10 Device identity and pinning

There are **two distinct notions of device identity**, and conflating them produces a failure that
looks like a hardware fault:

| notion | what it is | how you see it |
|---|---|---|
| **local device index** | a **dense** `0..N-1` numbering over the devices *visible to your process* | what `torch.hpu.device_count()` reports and what `torch.hpu.set_device(i)` indexes — **not** the vendor tool's index order |
| **physical module id / AIP** | the **fixed** per-card identity, stable across processes | the vendor tool's per-AIP listing (Module ID / PCI bus); its **index order is a third numbering** — neither the module id nor the local index (Chapter 3 §3.7) |

**`*_VISIBLE_DEVICES` is a container-runtime input, consumed *outside* the container.** Setting it
*inside* a container does nothing — nothing in the image reads it; the container's own device flags
are what select the cards. **INFERRED**
**`*_VISIBLE_MODULES` is not a filter in this stack** (**INFERRED**). It is a **local-index →
module-id table**, and it can differ from the vendor tool's index order. **MEASURED**

**The failure:** two processes that each pick the same **physical** module under different
**local** views do **not** both get it. Module acquisition is an **exclusive lock** — one module,
one process. The second requester **fails loudly** at device-acquire, and if it is a worker that
surfaces as a worker-init failure that reads like a device fault. There is **no silent sharing and
no state in which both run**. **MEASURED**

**Rules that follow:**

1. **Check what you can see before choosing anything.** Count the devices you have, and read the
   module-id list, before you decide how to pin.
2. **Use the view you were given.** If the environment already gives you a clean dense view, use
   **relative** numbering — and pin by **module id**, not by the vendor tool's index order, which is
   a third numbering (Chapter 3 §3.7).
3. **Pinning is the first device call you make — or an environment variable you export.** Setting
   the device as the first HPU call writes the module-id variable and establishes the mapping; you
   can also **export that variable yourself** to pin a process, which is what a launcher or a
   per-rank worker does. It is read **lazily** at device-acquire time, so it may be set *after* the
   imports as long as it precedes the first HPU tensor; a non-numeric value fails at **allocation**,
   not at import. **MEASURED**
4. **No self-report API is reliable** — the obvious "which device am I on?" calls can all report
   `0` regardless. **Read the module-id environment variable** (the one the runtime writes, or the
   one you exported) rather than asking the tensor. **MEASURED**
5. **Pin to an idle device; a device above baseline is *held*, not shared.** Acquisition is
   exclusive (the failure mode described above), so a device showing much more than the idle
   footprint has a process **holding** it — ordinarily **another HPU job** (another agent's, another
   user's, or yours), and only secondarily a non-HPU holder or a stale/interrupted run. A pin set
   must therefore be a **partition**: one distinct module per job.

This section is deliberately general: it is about the *mechanism*, and the specific environment
variables are named per-deployment. The manual does not assume any one container layout.

**What this section does not know.** Whether the module-id table's own value also restricts the
candidate set the driver picks from when *nothing* is pinned is **not established** — the binary
analysis finds the PT-bridge's by-type function does not consult it (**INFERRED**), but the
driver/thunk level is not settled (`hpu_isolation` §10.2), and the two-process arm we *have*
(two unpinned processes, watching which modules they take) used the container's **full** table,
so it cannot separate the table from the container's `--device` cgroup. A **discriminating** arm
is still open. The deployment-named treatment, the three numberings, and the per-rank rule are in
**Chapter 3 §3.7**.

---

## 1.11 Where to look things up yourself

Three introspection routes, all of which we used to produce this chapter:

| route | what it gives | example |
|---|---|---|
| **`DUMP_POST_GRAPHS=<dir>`** | one JSON per compiled graph with a **literal per-node `"engine"` field**, plus `guid`, `type`, `is_logical`, `origin_nodes`, `exec_order_idx`, `bundle_index`, and the cost fields (§1.7) | [`dump_post_graphs.py`](examples/01-basics/dump_post_graphs.py) |
| **`LOG_LEVEL_ALL=2` alone** | per-recipe arc jobs naming the engines: `(for 0[TPC])`, `(for 1[MME])`, `(for 2[DMA])` — the cheapest independent confirmation | same example, second half |
| **the SynapseAI C++ API** | build and launch a graph yourself; enumerate the recipe's launch tensors; read the GC's own node-type GUID table | [`synapse_cpp_minimal.cpp`](examples/01-basics/synapse_cpp_minimal.cpp) |

Two gotchas worth knowing before you try them:

- **`DUMP_POST_GRAPHS=<path>` must point at a directory that already exists.** If it does not, the
  compiler treats the path as a *file*, and concurrent writers race and fail the compile
  (`synStatus 26`). `mkdir -p` first. **MEASURED**
- **`**`-globbing skips hidden directories.** The dumps land in a dot-directory, so a shell glob
  can silently see a small fraction of them (one recon saw **24 of 1,141** files). Use
  `os.walk`/`find`. **MEASURED**

### A note on where the vendor claims in this chapter come from

The vendor rows cited above — the SRAM size, the TPC local memory, the queue taxonomy, the fp8
datatype description — are cited to **public sources**: the upstream Linux kernel's habanalabs
driver, the vendor's published documentation (`docs.habana.ai`) and the vendor's published PDFs.
Every citation in this chapter is a URL or an upstream file path that a reader can fetch.

**The manual does not depend on those sources being reachable.** Every claim is stated in full in
the text — the citation is for *checkability*, not for access. A reader who cannot fetch a vendor
page still gets the same chapter, and those rows are the ones least likely to change.

---

## 1.12 What this chapter leaves UNKNOWN

Stated here so the gaps are visible, and repeated inline above where they matter.

| # | UNKNOWN | status | what would settle it |
|---|---|---|---|
| 1 | **The TPC cost model's clock** | ⚠ **not successfully checked** — the symbol lookup did not complete on **any** attempt (timeouts), and a text search of the shipped libraries found nothing. **This is NOT "the answer is no".** | retry the symbol lookup on `TPCCostModel` / `getClockFreqMHz`; or disassemble the TPC cost model |
| 2 | **Whether the TPC cost model uses the 21 %-high general HAL reader** | ⚠ **not successfully checked** (same cause). The **DMA** cost model is **excluded** — a disassembly found **no clock call in it at all** (`Archive/by-date/2026-10-11/synapse_cpp/GC_COST_MODEL_CLOCK.md` §10). | the same retry, or the same disassembly |
| 3 | **The DMA driver's exact round-robin queue-selection rule** | open — the queue *families* are now documented (§1.2); the *rule* is not. Note the raw thunk path takes a **caller-supplied queue index**, so the round-robin we measured must be chosen **above** the driver | symbol access into the runtime library that assigns the queue |
| 4 | **Which queue family tids `4073/4082/4083` belong to** | **supported, not settled** — a third family (NIC) exists (§1.2), which is consistent with our split; the **tid→queue mapping was not given** | the profiler's `thread_name` for those tids, against the queue-name table |
| 5 | **Are the per-TPC AXI rate limiters programmed?** — `mmDCORE0_TPC0_CFG_RD_RATE_LIMIT` `0x400BDA8` / `WR_RATE_LIMIT` `0x400BDAC` exist; likewise the QMAN `HBW_RD_RATE_LIM_CFG_0/1` and `LBW_WR_RATE_LIM_CFG_0/1`. We found **no programmed values**. *(Source for those names and addresses, public and checkable: the upstream Linux kernel's habanalabs driver — `drivers/accel/habanalabs/include/gaudi2/asic_reg/dcore0_tpc0_cfg_regs.h:185,187` and `drivers/accel/habanalabs/include/gaudi2/asic_reg/dcore0_tpc0_qm_regs.h:1031-1037`, in current upstream `master`; line numbers move between kernel releases, so check the symbol, not the line.)* | **NEW, open.** If they *are* programmed they would cap per-TPC bandwidth and could explain part of the **2,147 GB/s measured vs 2,450 GB/s nominal** gap | read the registers on a live device, or find the firmware init sequence that writes them |
| ~~6~~ | ~~**The 512 MiB C++ bench point (1,646 GB/s)** — outlier, two untested explanations.~~ | **CLOSED — RESOLVED 2026-10-11.** The figure was **wrong for two independent reasons**: a **per-repetition `launch → sync`** wall time read as a device time (bench median **622.1 µs** against a device union of **530.5 µs**, a ~90 µs per-rep overhead; the **union is identical** whether you sync once or per rep), **and** a **4.6 % arithmetic slip** (`1,073,741,824 / 0.6221 ms` = **1,726**, not 1,646). Corrected value: **2,023 GB/s**, and the curve is **monotone in size**. The 2-D-vs-1-D shape hypothesis is **refuted** (0.03 % apart). See §1.6. | — **done.** *The remaining DMA gap is **below 16 MB**, where we have **no DMA-engine measurement** — and the TPC/GEMV cliff must not be substituted for it.* |
| 7 | **The MME rate's absolute value — a 2× unit ambiguity** | open | the artifact's own label reads "215 TFLOPS" where the arithmetic gives 430; settling the unit convention settles the number |
| 8 | **The MME rate at shapes other than the 4096³ family** | open | a rate sweep across shapes, each arm asserting `device_busy ≥ 80 %` |
| 9 | **`per_op_cost`** — the per-op fixed cost term | open; **we found no definition of one** | a direct measurement; four candidate meanings exist, three orders of magnitude apart |
| 10 | **Whether one process driving several devices keeps the tid windows** | open | one short run on a multi-device view, checking the three tid sets |
| 11 | **The measured graph count vs the O(N × buckets) prediction** — §1.4.3 states the arithmetic; we have not checked a real model's executed-graph count against it in this chapter's sources | open | count **distinct compiled graphs executed per step** on a real model and compare against `shapes × layer constants`. This is the check that turns §1.4.3 from an arithmetic into a diagnostic |
| 12 | **Whether the module-id table also restricts the driver's unpinned choice** (§1.10). The binary analysis finds the PT-bridge's by-type function does not consult it, but that is **INFERRED**, not measured, the driver/thunk level is **not settled**, and the unset path is otherwise documented as "first free module". | open — the two-process arm we have (two unpinned processes, watching which modules they take) used the container's **full** table, so it cannot separate the table from the container's `--device` cgroup | a **discriminating** arm: a per-job `HABANA_VISIBLE_MODULES` that *excludes* a cgroup-exposed module, with **no** pin, watching which modules the two processes take (`hl-smi -Q index,module_id,memory.used -f csv,nounits`) |

**Two kinds of negative, and they are not the same.** Items **1** and **2** are *"we could not
complete the check"* — the lookup that would answer them failed, so they carry **no information
either way** and must not be read as a vendor statement. The rest are *"we looked, and the answer
is not in the sources we have"*.

**Collapsing those two is how a tool outage becomes a false finding** — and it is the same class of
error as the UNKNOWN note at the top of this manual: a claim that was never checked acquires the
authority of one that was. When you record a negative, record **which kind** it is.

---

## What to carry forward

1. **Three engines — 24 TPC, 4 MME, 8 DMA tids (4 cores × 2 lanes; 5 DRAM/SRAM lanes).** "5
   queues" and "8 tids" are both correct about different things.
2. **One transfer at a time per lane; five lanes in parallel (measured Σ/union 2.79–2.97×).**
3. **No op → queue mapping** — the lane is a per-transfer round-robin choice.
4. **One compiled recipe routinely contains all three engines** (989 of 4,765 handles carry all
   three). Do not count nodes naively.
5. **Bandwidth is size-dependent.** 2,147 GB/s for the TPC stream at ≥134 MB; the DMA engine is
   ~1,776 GB/s at 16–64 MB and ~2,025 at ≥256 MB. **Eager `torch.copy_` is TPC, not the DMA
   engine.**
6. **The cost models have different error signs** — the MME model is accurate to 0.5 %; the DMA
   model is a size-dependently optimistic lower bound; the TPC model's clock is unknown.
7. **A marker duration is not automatically an execution window.** Only TPC is a true start→halt
   span; the MME's inflation is cross-tid (so its union is usable); the DMA's is not.
8. **Never quote a rate from an arm that failed its precondition.**
9. **Pre-flight every trace** — a stale profiler plugin path silently yields a trace with no
   device events at all.
10. **Two notions of device identity** — conflating them mis-pins a process with no error at pick
time; and a module is acquired **exclusively**, so a second requester of a held module **fails
loudly** rather than sharing it.
11. **Everything is compiled for the exact values it saw** — shapes *and* integers. An N-layer
    model is an O(N × buckets) graph count **by construction**; check a count against that
    arithmetic before treating it as a bug.

**Next:** [Chapter 2 — Optimizing small programs and regions](02-regions.md) takes the engine model
above and derives the rules for making independent work actually run at the same time.
