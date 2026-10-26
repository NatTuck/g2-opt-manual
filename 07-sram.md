# Chapter 7 — The on-die SRAM

**The 48 MiB SRAM is real, the compiler already slices into it, and you cannot place a tensor in it
from your own code.** Those three statements are the whole chapter, and they are not in tension: SRAM placement
is a decision the graph compiler makes, on criteria that are mostly about **graph structure**, and
the useful work is to make your tensors *eligible* and then **verify what it did** rather than to
request a buffer.

This chapter is worth reading before you plan anything that assumes SRAM, because **every obvious
route to it is closed and most of them fail without an error** (§7.4). It is also worth reading if you
have already decided SRAM is irrelevant: **it reads measurably faster than HBM on the TPC path —
10.2 % with non-overlapping distributions at 4 MB, and 1.18× at the highest rate we saw (32 MB, one rep
against the best of three) — and the gain is small because both arms land within ~10 % of the same
per-core band, which points at the TPC array rather than the memory (INFERRED)** (§7.6). It is still the
only lever in this manual that changes *where bytes live* rather than how they are moved, and the
chapter gives you the two ways to put bytes there yourself (§7.3.2) and the ladder for proving what the
compiler placed (§7.5.4).

> **Provenance for this chapter.** The measured results were taken on a **parallel port of the same
> model, on the same 4-AIP Gaudi 2 hardware**, by the project that shares this manual's measurement
> campaign; its archived artifacts are not published, so they are **provenance pointers, not checkable
> citations**. Where a claim could be checked against something you *can* open, it is: the address map
> is read from the device at runtime, the driver constants are cited to the upstream Linux kernel, and
> the graph-compiler knobs are quoted **from the `libSynapse.so` this manual was written against**
> (`strings` on the shipped binary — the same route §1.11 uses for `GCFG_DUMP_POST_GRAPHS`).
>
> **A version caveat, stated once.** The compiler *source* reads below are against a **1.16-era
> Synapse snapshot** that this project's engine runs; the shipping compiler on our stack is
> **1.24.2**, and every knob and message quoted here was **confirmed present in the 1.24.2 binary**.
> Treat the source reads as a map of *where a decision is made*, not as line-exact citations for your
> build. Where the two could differ, the chapter says so.
>
> **The §7.6 bandwidth comparison is ours, end to end** — our own TPC kernel, our own Synapse harness,
> the same box — and its raw artifacts are unpublished for the same reason as the rest. Its **gate** is
> not: it is a byte-valued `workspace_size` in the post-graph dump, so any reader can re-derive which
> arm they are in on their own install.

---

## 7.1 What it is: one 48 MiB pool, immediately below DRAM

The aperture is a **raw device address range** in the same "driver device map" address space that
`synDeviceMalloc` hands out. Read from the device at runtime via `synDeviceGetAttribute`:

```
SRAM_BASE_ADDRESS      = 0x1000fffffd000000
SRAM_SIZE              = 50331648        (48 MiB, exactly)
TPC_ENABLED_MASK       = 0xffffff        (24 TPCs)
ADDRESS_ALIGNMENT_SIZE = 128
CLK_RATE               = 1650 MHz
MAX_RMW_SIZE           = 16777216
```

**MEASURED.** The arithmetic closes exactly, and it closes against the **driver's** constant rather
than against every address the runtime reports: `SRAM_BASE + 48 MiB = 0x1001000000000000`, which is
`DRAM_PHYS_BASE` in the public header cited below. **A one-byte overrun at the top walks into DRAM**,
and copying past the aperture faults the card (§7.7) — that fault is measured, and it is the reason
the bounds rule exists.

> **⚠ There are two different "DRAM base" numbers, and they are not interchangeable.**
> `synDeviceGetAttribute` reports a `DRAM_BASE_ADDRESS` of `0x1001001800000000`; the **physical** base
> in the driver header is `0x1001000000000000`. They are different constants from different spaces,
> and **subtracting one from the other does not give a hole** — an earlier revision of this chapter
> did exactly that and published a "384 MiB hole" that matches neither figure. What is *measured* is
> narrower and sufficient for the safety rule: **copies past the aperture fault the device.** The size
> of any unmapped gap between them is **UNKNOWN** here.

The base and size are also in the public driver:

- **Source (public, checkable):** the upstream Linux kernel's habanalabs driver —
  `drivers/accel/habanalabs/include/gaudi2/gaudi2.h` (`SRAM_BASE_ADDR`, `SRAM_SIZE`). Line numbers
  move between releases; check the symbol, not the line. This is the same citation §1.5 uses for the
  48 MiB figure.

**Reserved low.** The HCL reservation (320 KiB) and the per-graph non-bundle reservation (§7.2.2) come off the top of
the budget. The 2 KiB SCAL reservation (`SCAL_RESERVED_SRAM_SIZE_H6 = 0x800`) is **inside** the
HAL's SRAM size rather than subtracted again — the measured budget identity in §7.2.2 has no term
for it, which is how this was caught. In practice **the graph compiler places the first
tensor it puts in SRAM at `SRAM_BASE + 0x80000`** (512 KiB) — **MEASURED**, and stable across
unrelated graphs. **An address scan that finds values at exactly `SRAM_BASE + 0x80000` has found a
real placement, not a coincidence**; an audit that reports "SRAM is in use" from one such hit alone
has not distinguished a placed operand from this fixed first-allocation offset.

**One pool, many clients.** `NUM_OF_DCORES = 4`, `NUM_OF_TPC_PER_DCORE = 6` → **24 TPCs**;
`NUM_OF_MME_PER_DCORE = 1`, `GAUDI2_NUM_MME = 4` → **4 MMEs**; one central SRAM. So SRAM is a
**28-client contended resource** — there is no per-TPC and no per-MME SRAM. Any claim that a tensor
"is in SRAM" is therefore a claim about a contended pool, and the useful measurement is a contention
curve, not a single peak number.

> **The vendor block diagram is wrong about the MME count** — it labels the right-hand block
> "2×MME", while the driver headers give **4** and a trace shows **4 MME tids** (§1.1). The same
> diagram is right about the 6 HBM stacks and the single central 48 MB SRAM. **A vendor block diagram
> can carry a wrong engine count while being right about everything else** — confirm a load-bearing
> layout fact against a header constant before building on it.

**On bandwidth, and the negative it leaves.** HBM spec is **2.45 TB/s**; **2.147 TB/s** is the
achieved rate (87.6 % of spec), not a second envelope — §1.6's framing is correct and this chapter does
not change it. Across ~150 gated measurement points in the campaign behind this chapter, the maximum
bandwidth observed **anywhere** was **2,306 GB/s = 94 % of spec**, and nothing in that set exceeded it.
On-die SRAM slower than off-die HBM is not credible, so a ceiling that never rises above HBM was
consistent with **nothing in that programme being SRAM-resident at a size that mattered** — which §7.2
explains.

**That negative has since been tested, and it moved by 1.4 %.** The one **SRAM-resident** arm measured
since (§7.6) reaches **2,338 GB/s = 95.4 % of spec** — above the old 2,306 GB/s ceiling, and still
below spec. §4.1.2's independent HBM-side measurement brackets it too: six well-packed TPC arms sit
dead flat at **2.26–2.43 TB/s**. **MEASURED.** **The size of that move is the finding**: it points at the
**TPC array**, not the memory, as what limits a TPC read — so putting the bytes in SRAM buys a per cent
or two at that limit rather than the multiple the two memories' specifications would imply. **That last
step is INFERRED rather than measured** — one SRAM point inside an HBM-fed band does not distinguish the
array's cap from SRAM's own supply to a TPC (§7.6).

---

## 7.2 The compiler already slices into it

This is the part most readers do not know, and it inverts the usual intuition. SRAM placement is not
something you opt into; **it is a graph-compiler pass that runs on every multi-node graph**, and the
question is only whether your tensors qualify.

### 7.2.1 There are two SRAM subsystems, and one of them is dead code

**The dead one.** `Gaudi2Graph`'s constructor hardcodes the allocation mode:

```c
getGraphAnnotation().memoryStrategyParams.allocatinMode = ALL_IN_DRAM;   // sic: "allocatinMode"
```

and the compiler *contains* a complete SRAM memory-strategy optimizer —
`MemoryStrategyManager::runOptimizationTrials` — which trials `ALL_IN_SRAM_PERSISTENT`, then
"activations in SRAM + static tensors prefetched", then falls back to DRAM, with capacity accounting,
fragmentation and prefetch-buffer sizing.

**Nothing calls it.** Its only call site is inside its own file, and its header is included by no
other file in the tree. **The memory-strategy optimizer is orphaned dead code in the 1.16-era tree we read** — it was not
checked against the 1.24.2 binary. **source read.** Most likely the static-tensor *prefetch* path is
Gaudi 1-era and was never validated on Gaudi 2 — that is the source's own guess, and it is **open**.

**The live one.** `passes/sram_management/` — **18,074 lines** across ~90 files — is a different
system and it *does* run. Its entry point is `SRAMSlicingManager::sliceGraph()`, which is gated on one
knob and then builds and solves slicing strategies:

```c
if (GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.value() == 0UL) return true;   // 0 = DISABLED
snapshotPreSlicingSizes(m_graph);
DuplicateMmeInputsHandler::handleDuplicateMmeInputs(m_graph);
m_normsHandler.findAndRemoveSliceNormNodes();
generateInitialBundles();
generateInitialStrategies();
expandAllBundles();
preSlicingOptimizations();
sliceAllBundles();
```

`generateInitialBundles()` produces five typed bundle classes — `m_mmeBundles`,
`m_tpcScalarPipeBundles`, `m_tpcBundles`, `m_rmwSectionBundles`, `m_dmaTransposeBundles`.
**`m_mmeBundles` exists specifically to slice MME operands so they fit in SRAM and to bundle nodes so
data stays resident**, and `flatten_mme.cpp` carries the comment *"If operand is in sram, flattened
tensor should also be in sram"*. **The compiler is designed for the MME to read SRAM.** **source read**

### 7.2.2 The budget is the device's size minus a graph-dependent reservation

The arithmetic, in `SlicingBrain::initKnobsValues()`:

```
nonBundleSramCap   = NonBundleSramTensorComp::getGraphNonBundleTensorSramSize(graph);
calculatedSRAMSize = halReader->getSRAMSizeInBytes() - nonBundleSramCap;   // aligned down to 256 KiB
knobs.maxSRAMCapInBytes = std::min(GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES, calculatedSRAMSize);
```

Measured byte-exact on one graph: `maxRange.size = 49,805,312` = `50,331,648 − 526,336`, where
526,336 = 327,680 (HCL) + **198,656** (non-bundle). **The non-bundle reservation varies with the
graph**, so treat ~194 KiB as a floor, not a constant. The log line reads "available SRAM 47.25 MB";
**trust the byte value**, because the log's MB unit does not reconcile exactly against either decimal
or binary MB.

At defaults **neither knob is responsible for the budget** — it is simply the device's SRAM size minus
the reservation. **The one user-settable lever is `GCFG_SET_SRAM_SIZE`, and it can only *lower* the
budget** (`min`).

### 7.2.3 The knobs, quoted from the shipping binary

These are all present in the **1.24.2** `libSynapse.so` this manual was written against, with the
help text below — **source read**, checkable on any install with
`strings -a /usr/lib/habanalabs/libSynapse.so | rg GCFG_SRAM`:

| knob | help text in the shipped binary |
|---|---|
| `GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES` | *"Maximal SRAM capacity to use when slicing operation to fit in SRAM **(0: Slicing disabled; -1: Use platform default)**"* |
| `GCFG_SET_SRAM_SIZE` | *"Set SRAM size in bytes **(0 == default)**"* |
| `GCFG_SCRATCHPAD_SRAM_MAX_SIZE` | *"This flag controls the max size for auxTensor without data to be in SRAM"* |
| `GCFG_SRAM_SLICER_COST_MODEL_ENABLED` | — |
| `GCFG_SRAM_SLICER_GRAPH_SIZE_OPTIMIZATION_ENABLED` | — |
| `GCFG_MIN_SLICE_SIZE_FOR_SRAM_FILLING` | *"Minimal slice size eligible for fill directive (fetch) from HBM to SRAM. Smaller slices will be read from HBM regardless of the solver directives."* |
| `GCFG_NON_BUNDLE_SRAM_ALLOCATION_FACTOR` | *"Factor to allocate non bundle tensors in SRAM — factor from max live capacity. <1 for disable"* |

> **Read the help text, not the name — it is where this subsystem's two traps live.**
> **(a)** `SRAM_SLICER_MAX_CAPACITY_BYTES = 0` **disables slicing**; `UINT64_MAX` (the default) means
> "no user cap", i.e. the **full** budget with slicing **on**. A value of `UINT64_MAX` observed
> anywhere means *"slicing enabled, no cap"* and must **never** be read as "SRAM disabled".
> **(b)** `GCFG_SET_SRAM_SIZE` clamps with `min` against an **`unsigned`**, and the next step
> **subtracts** the non-bundle reservation from it. Set it **below** that reservation and the
> subtraction **underflows** to a huge budget — the exact opposite of the clamp, and it looks
> identical to "the slicer ignored my setting". **To force a fallback, use a value comfortably above
> the reservation** (4 MiB, not 1 MiB), and **gate on `maxRange.size` in the log**, never on
> `inSram: false`.

Private (`MakePrivate`) knobs are refused with *"is internal config. cannot be set without
ENABLE_EXPERIMENTAL_FLAGS"* — but they **are** settable once `ENABLE_EXPERIMENTAL_FLAGS=true` is set
in the environment **before Synapse init**, in both directions. That gives you a working control arm:

```bash
ENABLE_EXPERIMENTAL_FLAGS=true SRAM_SLICER_MAX_CAPACITY_BYTES=0   # slicing OFF  <- the control
ENABLE_EXPERIMENTAL_FLAGS=true SRAM_SLICER_MAX_CAPACITY_BYTES=46137344   # 44 MiB cap
```

**MEASURED:** with `=0`, `inSram: true` drops **3 → 0** on the same graph and the `EPOCH_ALLOC`
"Allocate N Tensors in Sram" lines drop 2 → 1. Note `synConfigurationSet` is **context-dependent**: **without** experimental flags it returns
`synInvalidArgument` and the env path is silently ignored; **with** them it works at runtime (rc = 0,
exact readback) — but the capacity is **latched before the first compile**, so a runtime change does
not affect a graph compiled afterwards in the same process (§7.5.3). Set it in the environment, and
give each arm its own process.

### 7.2.4 `ALL_IN_DRAM` is the default mode — and it does not mean nothing is in SRAM

The `ALL_IN_DRAM` default (§7.2.1) invites the conclusion *"therefore no ordinary graph uses SRAM"*.
**That conclusion is wrong, and it was wrong in this project's own notes first.** `ALL_IN_DRAM` is the
default *allocation strategy*; specific tensor classes are placed in SRAM regardless of mode, by the
`setTensorInSram()` call sites and by the live slicer.

An empirical scan of the engine's own recipe cache refutes it directly: **≥ 31 % of that port's decode
recipes already contain SRAM-window tensor addresses** (92 of 299, 13 distinct offsets, up to
`+25.31 MiB`). **MEASURED**, and the fraction is a **lower bound**: a serialized recipe stores **one** SRAM base at
a fixed offset and encodes per-node placement as base-relative patch data, so an absolute-address
scan of a recipe **under-reports**. The same class of evidence exists here — **32 of our own archived
post-graph dumps carry `"allocation": "SRAM"`** (plus 40 more in working scratch, which is
disposable and will not survive a cleanup). Re-runnable: `rg -l '"allocation"\s*:\s*"SRAM"'`.

So SRAM **is** in use in production decode graphs — for **small scratch tensors at 0.3–1 MiB scale,
not for activations or weight tiles.** That is the honest reading, and it is a statement about
**capacity**, not about absence.

---

## 7.3 What decides placement

### 7.3.1 Graph structure, not execution mode

The determinant is **whether a tensor has a producer chain inside the graph**. A tensor that arrives
as a bare graph input, with no producer, stays in DRAM even though the slicer is running — because a
slicer with nothing to slice has nothing to place.

**MEASURED**, two runs, same shapes, `torch.compile(backend="hpu_backend", fullgraph=True,
dynamic=False)`:

| run | graph shape | result |
|---|---|---|
| A | weights passed in as **function arguments** | slicer runs; both weights `inSram: false`; only a 27 KB activation placed |
| B | weights **created inside** the compiled region | both MME weight operands `inSram: true`, sliced 3×, double-buffered |

```
[SRAM_SOL_GEN] Operand [1] 6 : 4608x5120, Sliced : 2048x5120, Num of slices: 3, Buffers: 2, inSram: true
```

**But "has a producer" is not enough — only a real compute node qualifies:**

| producer on a persistent parameter | weight placed in SRAM? |
|---|---|
| `reshape` (a view) | **no** — `inSram: true` = 1 (activation only) |
| `split` + `cat` (views) | **no** |
| elementwise `* 1.0` | **YES** — `inSram: true` = 2 |

**Pure views are treated as trivial and are never sliced.** A `nn.Parameter` **is** placeable, but only
behind a genuine compute node. **MEASURED**

**⇒ This is the lever, and it is a graph-structure lever.** A model whose weights are produced by a
real op — a dequant, a cast, a scale — is **structurally in the placeable category**; a model that
passes them in as parameters is not. If you want SRAM for a weight, **give it a producer that costs
nothing**.

> **Execution mode is a separate question, and the old answer was wrong.** The requirement is that
> **Synapse receives a multi-node graph**; lazy mode is only one way to get one. Raw eager ops compile
> to **one node per op** and the slicer has nothing to bundle. `torch.compile(backend="hpu_backend")`
> — what vllm-gaudi uses — gives it the whole captured region, and **does** get SRAM slicing. Lazy
> and `torch.compile` are **mutually exclusive** in this bridge, so forcing lazy mode would disable
> the Dynamo path a modern port is built around: **not a viable lever.**
>
> The binary makes this explicit — it contains the refusals **"Unsupported SRAM usage in Eager-mode"**
> and *"Node {} has {} {} in SRAM. SRAM is not supported at Eager mode"*. **A measurement taken in raw
> eager mode therefore says nothing about SRAM slicing**; an experiment that produced five GC lines
> and zero `SRAM_SLICE` lines did so because each op compiled to a one-node graph.

**⇒ Placement is a property of the graph's CONSTRUCTION, not of its shapes.** This is the one that
silently costs you the whole lever. The same logical span, the same shapes, built two ways:

| construction | what the slicer does |
|---|---|
| a single `bmm` | bundles it and **places the tile** — 6 slices, 2 buffers, epoch 13,117,440 (26.3 % of `maxRange`) |
| **6 per-expert GEMMs plus a `cat`** | picks the `5120x1` *activation* as master operand (`numberOfUsesA: 2`, `numberOfUsesB: 1`) and **gives up**: `Cannot slice master operand 15 on dim 2 - too small` → `No slicing dims were found` → `Allocate 0 Tensors in Sram, epoch size: 0` |

**The second graph still runs — just from HBM, silently, with no error.** So **"placement is proven"
must be re-gated on the exact graph every time**, and a span that looks equivalent to one that was
placed can lose placement entirely.

**⇒ For a real port that makes this an audit, not a footnote:** if the MoE path expands a batch GEMM
into per-expert GEMMs plus a `cat`, it will **not** get SRAM placement even though the same shapes do
when built as a single `bmm`. **Check the construction, not the shape.** **MEASURED**

### 7.3.2 Using SRAM directly: the two routes that work

Placement is the compiler's decision (§7.3.1, §7.3.3). But SRAM is not sealed off: there are **two
verified routes** by which you can put bytes there yourself and read them back. **Neither of them
places a tensor** — they populate and observe, while the compiler decides what lives where. Keep that
distinction, because it is the one that gets lost.

#### (a) Host DMA to a raw address

The aperture is in the same "driver device map" address space that `synDeviceMalloc` hands out, so
`synMemCopyAsync` reaches it at a raw address. **MEASURED:** with every host buffer from
`synHostMalloc`, it round-trips **byte-exactly** into `SRAM_BASE + 1 MiB`, and at **five** separate
in-window offsets (1 / 2 / 4 / 8 / 16 MiB), **each reading back its own distinct pattern** — so it is
the real aperture, not one shadow page. Pre-existing content at `SRAM+1 MiB` was non-zero and
structured (`0x40000e38, 0x40000dbc`); it was overwritten and **restored exact**.

> **This retracts an earlier conclusion, and the retraction is the useful part.** A previous probe got
> `synMappingNotFound (34)` and concluded *"SRAM is not DMA-addressable."* **Code 34 is returned when
> the *host* side of the copy is ordinary heap rather than `synHostMalloc` memory** — it says nothing
> about the device address. The old test passed a Python ctypes array. **A return code is a statement
> about the call, not about the address.**

**The host staging buffer must come from `synHostMalloc`.** That is the whole of the trap.

#### (b) A TPC via an aux tensor

`AuxTensor.sramAlloc` is documented as *"This flag instruct GC to allocate the aux tensor in SRAM
memory"* and is a **dead field** — one declaration, no readers anywhere, and three GUIDs differing
only in `(sramAlloc, noInit)` produced an **identical** recipe address. **Do not use it.**

What actually decides aux placement is whether the buffer has data, and its size:

```
pData == nullptr  AND  write-space < GCFG_SCRATCHPAD_SRAM_MAX_SIZE (10 MiB by default)   => SRAM
```

`noInit` controls **initialisation**, not placement. **MEASURED**, same ELF and same glue, only
`bufferSize` changed:

| aux `bufferSize` | placement | workspace |
|---|---|---|
| 4 KiB | SRAM `0x1000fffffd080000` | 256 B |
| 1 MiB | SRAM `0x1000fffffd080000` | 256 B |
| 12 MiB | **no SRAM address** | 25,166,080 B |

**The DRAM arm's workspace has a computable size, and that is what makes it a usable gate: `2 × the
buffer + 256 B`.** The 12 MiB row above is `2 × 12 MiB + 256 B` **exactly**, and the 64 MB fallback
further down is `2 × 64 MiB + 256 B` **exactly** — **MEASURED**, two independent witnesses to the same
rule. So a 4 MiB buffer should show **8,388,864 B** and a 32 MB buffer **67,109,120 B**. **Check the
number you observe against the one your own buffer predicts** — a remembered figure from someone else's
run is not a gate.

So at the **default** the usable ceiling is **~10 MiB per aux tensor** — *per tensor, not per node* —
against a 48 MiB aperture shared with every TPC and every concurrent kernel.

**The gate is the knob, not the pool, and it is worth being precise about which binds.**
`GCFG_SCRATCHPAD_SRAM_MAX_SIZE` *is* that ~10 MiB, so raising it moves the gate: with it set to 48 MB,
a **32 MB** aux lands in SRAM (workspace **256 B**, §7.6). What binds next is the **48 MiB aperture**
itself, and it binds **silently** — a **64 MB** aux exceeds it and falls back to the DRAM workspace
**in the arm you believe is the SRAM arm**, with `workspace = 134,217,984 B` as the only evidence
(`2 × 64 MiB + 256 B`, exactly the rule above). **Read the workspace size, not the intent.**
**MEASURED.**

**And it works end to end. MEASURED:** the aux tensor lands at `0x1000fffffd080000` inside the
aperture, the kernel writes `0x5A5A1234` **through its descriptor**, and the host DMAs those exact
bytes back from that address. Reproduced **3x**. Host -> SRAM -> compute engine, all three legs. Two
nodes in one graph, each with its own 4 KiB SRAM aux, get addresses `SRAM+0x80000` and `SRAM+0x81000`
— delta exactly `0x1000`, each signature found exactly once. **No collision.**

**What this buys, stated plainly:** a declared **scratch buffer** up to the **default ~10 MiB**, or as
large as `GCFG_SCRATCHPAD_SRAM_MAX_SIZE` allows (§7.6 used **32 MB**). It is **not** a way to hold model
weights, and it is **not** a way to place a tensor. It is how you get a small hot buffer somewhere you
can prove, and it is one of the instruments for proving that the compiler's own placement is real
(§7.5.4).

**And the neighbouring header comment is wrong too:** `auxiliaryTensorNr` is documented as *"will be
set by GC to MAX_AUX_NUM"* and arrives as **0** on first instantiation. A glue that treats it as a
capacity and returns `GLUE_FAILED` surfaces as **`synGraphCompile -> synFail (26)` with a 0-byte build
log**. The array is a real `std::array<AuxTensor, MAX_TENSOR_NR>`; index 0 is always writable. **Do
not gate on the incoming value.**

### 7.3.3 Graph tensors — the mechanism that matters, and no route to it that we could find

A graph tensor *can* be SRAM-resident, and this is the only route that can serve an **MME** (an aux
tensor is a TPC-only mechanism — it arrives as an extra kernel parameter, so an MME cannot read one).

`Tensor::setTensorInSram()` sets the tensor's memory section ID to `getSramMemoryID()` and
`memory.location = TENSOR_IN_SRAM`; `SramTensorsEpochAllocator` then allocates every tensor carrying
that ID into the SRAM heap. **source read**

**And that ID is the problem.** `getSramMemoryID()` returns `getMaxMemorySectionID()` — the **maximum**
section ID, a sentinel encoded into the tensor's virtual address. `synSectionCreate` only hands out
ordinary low-numbered IDs, so **no user can produce it**. That is precisely the missing request, and
there is no supported way around it. **source read**

**⇒ What is left for you is influence, not control:** shape your graph so the compiler's own passes
place the tensor (§7.3.1), and **verify** (§7.5). Do not write code that tries to place a tensor.

### 7.3.4 Over-subscription aborts; it does not spill

`SramTensorsEpochAllocator::allocateTensorInDram` is `HB_ASSERT(0)`. A graph that asks for more
SRAM-resident tensors than the aperture holds does **not** quietly fall back to DRAM — **it aborts the
process.** Any change that exposes SRAM placement must enforce the capacity itself. **source read**

---

### 7.3.5 What there is to place: the fp8 dequant tile

The candidate tensor this topic keeps returning to is the fp8 tile the fp4→fp8 dequant materialises
for the MoE experts. Its size belongs on the page, because **"would it even fit?" is answerable
without a device.**

The op's output formula is `{6, q.size(2)/32, q.size(1)*128}`, and both prepared checkpoints resolve
to the same totals:

| | gate+up | down | per layer |
|---|---|---|---|
| fp8 tile | **37.50 MB** | **18.75 MB** | **56.25 MB** |

- **Per decode step** (40 MoE layers): **2.20 GB** materialised.
- **Against the ~47.25 MB budget** (§7.2.2): gate+up (37.50 MB) **fits**, and down (18.75 MB)
  **fits**. Both together do not, but they are sequential anyway — so **each tile is individually
  placeable with roughly 10 MB of headroom.**
- **Two premises worth correcting, because both are easy to get wrong.** The prepared op's **dim0 is
  the 6 routed experts; there is no token dimension**, so **BS = 2 does not double the tile.** And the
  prepared op is **top-6 only** (its contract requires `ids.size(1) == 6`), so an `experts 12`
  variation is **invalid**, not merely slow. *(Only the `n256` variant has `slots = tokens × experts`
  and would scale with batch — and that variant is currently unrunnable in the harness, §7.8.)*

**MEASURED / source read** — checkpoint safetensors headers plus the op's own output formula. *(The
in-code fastpath guard still tests `(384, 18, 163840)` / `(384, 40, 36864)`, which matches neither
checkpoint. It is a stale guard, not a size oracle.)*

> **What this does *not* give you is a time.** Summed write+read for 2.20 GB at 873 / 1,889 GB/s is
> ~3.7 ms, and that figure is **summed work** — the recoverable amount is the un-overlapped part of
> the union, and it has to be measured (§2.9 rule 1). **Capacity is what this section answers; cost is
> §7.6's, and it is open.**

---

## 7.4 The routes that do not work — and one nobody has tried

Each of these was tried and refused. Two of them **fail silently**, which is why they are here.

| route | result |
|---|---|
| `synDeviceMalloc` at a requested SRAM address | returns **`synSuccess`** and hands back a **DRAM** address. The header says the request is *"a hint"* and that malloc *"will succeed regardless"* — **the returned-address comparison is the only thing that catches it.** |
| `synTensorSetDeviceAddress` / `synSectionSetDeviceAddress` | **not declared in the shipped headers** — a search, not a proof of absence |
| `MEMORY_ATTRIBUTE_SRAM = 4` | **untried, not refuted** — the only support for this row is a header comment (`// currently not supported`), and a header comment is not evidence (§7.3.2). Nobody has tried it. |
| a TPC kernel computing the address | **impossible, and silently wrong** — see below |
| `AuxTensor.sramAlloc` | **dead field** (§7.3.2) |
| asking the orphaned memory-strategy optimizer | **dead code** (§7.2.1) |

### The TPC cannot address SRAM, and it fails without an error

`tpc-clang -march=gaudi2` has **no 64-bit integer type** — it rejects `unsigned long long` outright —
and reports `__SIZEOF_POINTER__ 4`, `__POINTER_WIDTH__ 32`. **Both `void*` and `__global void*` are
32-bit.** A TPC pointer cannot hold a 48-bit device address.

Global memory is reachable **only through a tensor descriptor** (`gen_addr` + `ld_g`/`st_g`), whose
base register Synapse patches. Write `*(volatile uint*)0x1000fffffd100000 = x` and you get:

```
ld_l  %S0, -0x2f00000      # LOCAL load
st_l  -0x2f00000, %S1      # LOCAL store
ld_l  %S1, -0x2f00000
```

`-0x2f00000` is the signed 32-bit reading of `0xFD100000` — the low half of the address. `0x1000ffff`
was **silently discarded**, and the emitted ops are **local-memory**, not global. Because TPC local
memory is 1 KiB scalar + 80 KiB vector, the offset **aliases** (`0xFD100000 mod 0x20000 == 0` → local
offset 0, the kernel's own scalar state). **The round trip genuinely holds — at the wrong address.**
In the run behind this section the readback returned a value a *previous process* had left there, so
the probe reported success. It then faulted the engine asynchronously **while every Synapse call in
the run returned `synSuccess (0)`**. **MEASURED**

> **Rule: "it compiles" is not evidence that a TPC kernel emits the access you think.** Read
> `tpc-clang -S` and look for `ld_g`/`st_g` (global, tensor-relative) versus `ld_l`/`st_l` (local). A
> hardcoded address that compiles to `ld_l`/`st_l` is not an address at all. This is the same class as
> §4's VLIW-bundle point: the listing is the ground truth, not the source.

---

## 7.5 How to observe it

You cannot control placement, so **observing** it is the skill. There are two instruments and both
have a trap that makes silence look like a result.

### 7.5.1 The loggers

`hl_logger` resolves levels in this precedence:

```
LOG_LEVEL_<LOGGER_NAME>      highest
LOG_LEVEL_ALL_<prefix>
LOG_LEVEL_ALL                lowest
```

Values are **integers, not strings**: `TRACE=0 DEBUG=1 INFO=2 WARN=3 ERROR=4 CRITICAL=5 OFF=6`. There
is a **parallel `LAZY_LOG_LEVEL_*` set**, and **which one is live depends on the execution mode** —
so **set both forms**. Logger names come from `HLLOG_DEFINE_MODULE_LOGGER(...)`:

```bash
LOG_LEVEL_SRAM_SLICE=0 LAZY_LOG_LEVEL_SRAM_SLICE=0        # slicer strategy dumps
LOG_LEVEL_SRAM_SOL_GEN=0 LAZY_LOG_LEVEL_SRAM_SOL_GEN=0    # solution generation  <- the inSram lines
LOG_LEVEL_EPOCH_ALLOC=0 LAZY_LOG_LEVEL_EPOCH_ALLOC=0      # why a tensor went to SRAM or DRAM
LOG_LEVEL_COST_MODEL=0                                    # what slicing is thought to be worth
LOG_LEVEL_GC=2                                            # "Memory strategy params are:"
```

The binary also carries a metrics line worth grepping for:
*"Metrics : SRAM Capacity : {}[MB], MME Utilization : {}, HBM Bandwidth: {}[GB/s], SB Reuse : {},
Double-Buffer : {}"*.

> **⚠ The compiler is ASYNCHRONOUS, and the log is truncated twice — two independent ways to read
> silence as a result.**
>
> 1. **`torch.hpu.synchronize()` returns while the graph compiler is still working.** Its placement
>    lines land **seconds later** (measured: TPC-binary extraction at 04:32:54, `SRAM_SOL_GEN` at
>    04:32:58, against a synchronise that returned in 0.2 s). Reading `graph_compiler.log` straight
>    after the first call is therefore **a race, and it is usually lost**. **Poll until the log's
>    delta contains `epoch size:`** — or the slicer's explicit give-up line `No slicing dims were
>    found` — before reporting anything.
> 2. **The log is truncated at process start and again at exit**, so a shell `cp` taken after the
>    process ends reliably captures **0 bytes**. That is the real mechanism behind "every log in this
>    batch was empty", and it is **not** the same as "logging was off". **Read the log in-process**,
>    and print its size in the harness so a failed capture is visible as a failed capture.
>
> **Every placement reading in the campaign behind this chapter was exposed to (1)**, including
> readings reported as facts. Treat any placement number not gated on a completed log delta as
> provisional. GC may also write to a *separate* file (`GRAPH_COMPILER_SEPARATE_LOG_FILE`).

### 7.5.2 The gates that make an A/B an A/B

Four gates, and an arm that fails any of them is not the arm you think it is:

1. **The placement line for the operand you are actually measuring — not a count of `inSram: true`
   lines.** A count can be non-zero for an operand that is not the one under test; that happened
   (§7.6), and it reads as a pass. Not "the run succeeded" either.
2. **`maxRange.size` identical across arms.** If the clamp did not take effect, any timing under it
   is meaningless (§7.2.3).
3. **`hl-smi` uncorrected-event count before and after** (§7.7).
4. **The graph's `workspace_size`, when the arm is supposed to materialise a DRAM copy.** A gate whose
   expected value is arithmetic: **256 B** means nothing was materialised, **`2 × buffer + 256 B`**
   means a DRAM copy was allocated (§7.3.2 derives the rule, §7.6 applies it). It is also what catches a
   tile that silently fell back.

### 7.5.3 Interleave the arms

`A,B,A,B` — not `A,A,B,B`. Sequential blocks confound drift with effect.

**But the interleaving has to be at PROCESS level, and that is not obvious.**
`synConfigurationSet` for the capacity knob **does** work at runtime — rc = 0, exact readback,
accepts `0` / `4194304` / `33554432` / `UINT64_MAX` — yet **it does not change placement for a graph
compiled afterwards in the same process**: compile #1 at the default and compile #2 with the capacity
set to `0` both report `inSram: true`. **The capacity is latched before the first compile**, so an
**in-process A/B is dead** and the two arms must be separate processes. Set the knob in the
environment and give each arm its own process.

**And interleaving does not replace measuring the floor.** This project's own first attempt *was*
interleaved — `A,B,A,B`, two rounds — and still produced a confident **12.9 % "slower"** result. The
error was not the ordering; it was that **no noise floor had been measured yet**. Ordering removes
drift; only a floor tells you whether the delta is real.

---

### 7.5.4 Verifying that a tensor is actually in SRAM

**Placement fails silently** — the graph still runs, just from HBM — so "I think it's in SRAM" is
worth nothing. There is a ladder of instruments; each proves something different, and each has a trap.

| instrument | what it proves | its trap |
|---|---|---|
| `[SRAM_SOL_GEN]` **`inSram: true`** plus the slice/buffer line | the slicer placed **this** operand | valid only after a **completed log delta** — the compiler is asynchronous (§7.5.1) |
| `[EPOCH_ALLOC]` `Allocate N Tensors in Sram` / `epoch size:` | how much actually went to SRAM | 0 bytes is not "no slicing" until the give-up line is present |
| **`No slicing dims were found`** | placement was **attempted and refused** | this is the *positive* evidence of a silent failure — the most useful line in the log |
| `maxRange.size` | the budget actually in force | must be identical across arms, or the clamp did not take (§7.2.3) |
| post-graph dump `"allocation": "SRAM"` | a tensor is section-backed in SRAM | a **graph-node** fact, not runtime occupancy — and it needs a cold cache, since a hit builds no graph |
| a scan of the recipe for 64-bit values inside `[SRAM_BASE, SRAM_BASE + 48 MiB)` | a recipe carries SRAM addresses | **under-reports**: one base at a fixed offset, with per-node placement as base-relative patch data |
| **`synRecipeSerialize`** → scan the written file for in-window 64-bit values | the aux tensor's **actual device address** | the check that separates real placement from a silent fallback — one in-window value at a fixed offset here, none in the DRAM window |
| **the graph's `workspace_size`** — 256 B against `2 × buffer + 256 B` | whether the compiler **materialised a DRAM copy** of the buffer | it is a **graph-object** field, so it appears in only a minority of dump files (224 of 21,763 in one census, [§3.6.6](03-porting.md)); read it for *the* graph that ran, and prefer it to any count of placement lines |
| **a runtime signature scan** — write a known pattern into the aperture and DMA it back | the aperture is real and yours to populate | needs `synHostMalloc` staging; a plain heap buffer returns 34 and tells you nothing (§7.3.2) |
| re-gating the placement line for the operand under test on the **exact graph** | *this* graph placed *that* tensor, not one that looked equivalent | construction decides placement, not shape (§7.3.1); and a **count** of `inSram: true` lines is not the gate (§7.5.2) |

**The principle under all of them, and it is the transferable part:** *require a property the wrong
explanation cannot produce.* An address inside the aperture. A byte-exact host DMA match. `st_g`
rather than `st_l` in a TPC listing. A fault that the alternative explanation would also have produced
proves nothing.

## 7.6 What it is worth — measured, and why the gain is small

**This section used to take a position instead of quoting a number:**

> **SRAM should be significantly faster than HBM for a compute engine that reads from it. We have not
> yet demonstrated that.**

**That is out of date. It has been demonstrated on the TPC path**, and this section states the size of
the effect and — the more useful half — what can and cannot be said about why the gain is small.

**The measurement, and the conditions that decide what it means.** One **custom TPC kernel**
(`sram_bw_gaudi2`), built with `tpc-clang` on the same glue and entry-point pattern as `fp4_dequant`,
driven from a C++ Synapse harness. The buffer is declared as an **aux tensor** with `noInit = 1` — the
one route to a guaranteed SRAM buffer (§7.3.2). **Same kernel, same buffer size, same launch in both
arms**; the only difference is placement, selected by `SCRATCHPAD_SRAM_MAX_SIZE` — at the default the
buffer is in SRAM, at 1 MiB the compiler sends it to the DRAM workspace instead. *(This is a plain
size comparison, and **not** the knob whose unsigned-underflow trap §7.2.3 warns about — that one is
`GCFG_SET_SRAM_SIZE`; and neither arm is gated on the knob's name, only on the workspace size below.)*

**The placement gate is the workspace size, and its expected value is computable — which is what makes
it unbreakable.** Nothing materialised is **256 B**; a DRAM copy is **2 × the buffer + 256 B**, the rule
§7.3.2 derives from two independent buffers. So the gate is arithmetic you can check for *your* buffer:

| arm | workspace, expected | what it means |
|---|---|---|
| SRAM | **256 B** | nothing materialised; the buffer is in the aperture |
| DRAM | **2 × the buffer + 256 B** — 4 MB → 8,388,864 B, 32 MB → 67,109,120 B | the compiler allocated a DRAM copy of the buffer |

**One loose end, stated rather than smoothed over.** The DRAM gate recorded in the source run behind
these rates reads **92,274,944 B**, which by that rule is `2 × 44 MiB + 256 B` — a **44 MiB** buffer.
**Neither of the two configurations tabulated below is 44 MiB**, so that figure is not this
measurement's gate, and which run it belongs to has not been recovered: treat its configuration as
**UNKNOWN**. It does not weaken either arm — the SRAM arm's **256 B** proves nothing was materialised,
and any non-trivial DRAM workspace proves a copy was — but it is exactly the hazard `README.md` names,
a figure that recomputes to something plausible and therefore reads as right.

| configuration | SRAM | DRAM | ratio |
|---|---|---|---|
| **4 MB × 16 nodes = 64 MB read** | **1,141.7 GB/s** (1154.6 / 1123.7 / 1146.8) | **1,036.4 GB/s** (1046.1 / 1032.2 / 1031.0) | **+10.2 %** |
| **32 MB × 16 nodes = 512 MB read** (highest rate) | **2,338 GB/s** | **1,974 GB/s** | **1.18×** |

**MEASURED** — same box, one process per arm, gate read on every run. *(The per-config workspace values
are the ones the rule predicts, 8,388,864 B and 67,109,120 B; the source record's DRAM value is the
44 MiB one flagged above, so the numbers to reproduce are the rates and the SRAM arm's 256 B.)*

**The 4 MB row is the one that carries the direction, and it is the headline.** Three reps per arm, and
the distributions do not overlap: SRAM's **minimum** (1,123.7) is above DRAM's **maximum** (1,046.1). A
confirmation run gave 1,151.6 against 1,022.8 — **12.6 %**.

**The 32 MB row is the highest rate we saw, and it is weaker evidence than its precision suggests.** It
is **one SRAM rep against the best of three DRAM runs**, and the DRAM arm alone spanned **1,573–1,974
GB/s across those three identical runs** — so 1.18× is **best-vs-best**, and that arm's own spread is
wider than the effect it is being used to show. That comparison is **conservative for SRAM** (its single
rep against DRAM's best), and it is still not an effect size: quote it as the configuration that reaches
the highest rate. The two rows are also **different kernel versions** (8 accumulators at 4 MB, 16 at
32 MB), so they are two A/Bs, not a size sweep of one.

**Why the gap is small — and the honest answer is that we can say where each arm lands, not fully
why.** `DUMP_POST_GRAPHS` reports **`tpc_working_engines: [24,24,24]` on all 16 nodes**: the full TPC
array is **engaged**, in both arms. **Engaged is not saturated** — that field counts working engines,
and it is not a bandwidth measurement.

| | aggregate | per core (÷ 24 TPCs) | against the measured ceilings |
|---|---|---|---|
| SRAM arm | 2,338 GB/s | **97.4 GB/s** | **+9 %** on §1.6's 89.5 GB/s-per-core stream envelope; **103 %** of §4.1.2's 2.26 TB/s floor |
| DRAM arm | 1,974 GB/s | **82.3 GB/s** | **−8 %** on §1.6's envelope; **87 %** of §4.1.2's 2.26 TB/s floor, i.e. **~13 % below** it; 81 % of the 2,450 GB/s spec |

**Where each arm lands, with no mechanism attached.** The SRAM arm's 2,338 GB/s is **inside** §4.1.2's
**HBM-fed** band of 2.26–2.43 TB/s. The DRAM arm's 1,974 GB/s is **~13 % below that band's floor**. **Why
a DRAM-resident read of this kernel falls short of a rate that HBM demonstrably supplies in §4.1.2 is
NOT established** — and it cannot be a supply limit, because §4.1.2's six arms are themselves HBM-fed
and they exceed this arm. The distance between the two arms here is **~18 %** (97.4 against 82.3 GB/s
per core), which is the room SRAM had to win in at this configuration.

**⚠ "The engines bind, not the memory" is INFERRED, not measured, and one point cannot separate the two
explanations that fit it.** What is measured is that **one** SRAM point (2,338 GB/s) lands inside a band
measured **from HBM**. That is equally consistent with **(i)** the TPC array capping at ~2.3–2.4 TB/s
and **(ii)** SRAM's own supply to a TPC happening to be ~2.3–2.4 TB/s. **Nothing in this measurement
distinguishes them**; the discriminating test is §7.8 item 2(b) — push the SRAM arm past the array's
rate, or use an engine with more appetite per core.

*(The per-core column in the table above is a division of the aggregate by the 24 TPCs in
`TPC_ENABLED_MASK` (§7.1); the same division on §1.6's **2,147 GB/s** aggregate gives **89.5 GB/s per
core**. §1.6 quotes the aggregate — the per-core form is spelled out here because it is what makes the
comparison to the array a comparison.)*

> **The honest framing, in one sentence.** *This is a real, repeatable SRAM advantage over HBM on the
> TPC path — **+10.2 % at 4 MB with non-overlapping distributions, and 1.18× at the highest rate we saw
> (32 MB, one rep against the best of three)** — and it is small because both arms land within ~10 % of
> the same per-core band, ~2.3–2.4 TB/s, which **points at** the array rather than the memory as the
> binding constraint (**INFERRED**, above — one measurement does not settle it).*

**Reaching SRAM's own bandwidth needs an engine with more appetite, and that engine is the MME.** The
design intent is explicit — the vendor's architecture page says the SRAM has *"sufficient bandwidth to
allow MME, TPC, DMAs and RDMA NICs to operate in parallel"* — and the compiler is built for it
(`m_mmeBundles` slices MME operands to fit SRAM; `flatten_mme` propagates SRAM placement onto them,
§7.2.1). **ASSUMED** for the design-intent quote, **source read** for the passes. **What stops it is the
compiler, not the silicon — MEASURED:** an MME operand marked `inSram: true` with `Num of slices: 1`
does **not** shrink the workspace, so that placement **adds a staging copy rather than replacing one**.
That is §7.8 item 1, and it is the open half of the question.

**The levers that produced these numbers, in the order they paid** — kernel-level, and transferable to
any TPC kernel that streams:

1. **Loads in flight 8 → 16 per loop iteration, with independent accumulators.** The largest single
   gain. The original `acc = acc + x` chain was latency-bound; independent accumulators keep many loads
   outstanding. **32 accumulators did not help** (2,303 against 2,339 GB/s) — **16 is the knee.**
2. **A bigger tile per node:** 4 MB × 16 → 32 MB × 16, with `SCRATCHPAD_SRAM_MAX_SIZE` raised so a
   32 MB aux still lands in SRAM (§7.3.2).
3. **Chunk 2,048 → 4,096 bytes** per index-space unit, so the inner body is 16 loads rather than 8.

**And the order matters more than the list: the same instrument reported 431.5 GB/s (SRAM) against
429.5 GB/s (DRAM) — identical — before the kernel was fixed.** At five times below the envelope it
could not distinguish memories at all. **A null from a slow kernel is a null about the kernel.**

**Three traps that produced confident nonsense first, recorded as method:**

- **A rate above the hardware's supply is a bug, not a result.** An early version reported **3.3 TB/s**,
  above HBM's 2.45 TB/s spec and therefore impossible. The 1-D index space covered only the innermost
  dimension, so each node read **4 KB instead of 4 MB** and the rate was inflated ~1,000×. **Do that one
  division against the supply before believing the finding** — it is the cheapest check in this chapter.
- **Gate on the specific tensor, never on a count of `inSram: true` lines.** A run reported two operands
  "in SRAM" while the one that mattered was not. The workspace size cannot be fooled this way (§7.5.2,
  §7.5.4).
- **64 MB tiles silently fall back to DRAM** — they exceed the 48 MiB aperture, and the *SRAM* arm then
  reads `workspace = 134,217,984 B`. The gate caught a mislabelled arm, which is exactly what the ladder
  in §7.5.4 exists for.

**What this measurement does not establish, stated so it is not built on:**

- **Nothing about the MME path.** This is a TPC result; §7.8 item 1 is untouched by it.
- **Nothing about SRAM's bandwidth above the TPC band.** 2,338 GB/s is where our **instrument** stopped
  — whether that is the array's cap or SRAM's own supply to a TPC is not separated by one point (§7.6).
- **The harness's own per-rep overhead is not measured**, so the two ratios are **lower bounds** on
  anything: a fixed additive overhead depresses both arms and compresses the ratio (§7.6.1 item 4).
- **Nothing about a production span.** This is a standalone kernel, not a span A/B, so it does not price
  SRAM for a real decode step. **It routes around the reasons this chapter previously quoted nothing; it
  does not repair them**, and they still stand for any span-level measurement:

  - **The span sample is small and the arms overlap.** Five to ten gated runs per arm, and the pooled
    ranges of the SRAM and control arms overlap — the best control run beat the worst SRAM run.
  - **Those gates are log reads, and log reads are racy** — the compiler is asynchronous and its
    placement lines land seconds after `synchronize()` returns (§7.5.1). The source's own later note is
    blunt: *"every placement reading in this project was exposed to this, **including readings reported
    as facts**."*
  - **The arms cannot be interleaved in one process** — capacity is latched before the first compile
    (§7.5.3), so any interleaved run needs one process per arm.
  - **The measured span may not be the span that matters.** One arm is a `mul`-producer bf16 GEMM graph;
    another is a compute-bound dequant span where slicing adds double-buffer overhead and there is
    nothing to win. **Neither is a production decode step**, and neither result transfers to the other.

**So a number is quoted now, and its scope is the measurement's scope**: a TPC kernel streaming from a
guaranteed-SRAM buffer, at the two configurations above, gated on the workspace size. **It is not a span
speedup and must not be carried into a decode-step model** — the span A/B that would license that is
still the open item, and §7.6.1 is what it has to satisfy.

### 7.6.1 What a valid measurement must satisfy

This is the part of the work that **is** solid — several of these were learned by getting them wrong —
and any future measurement of this effect has to meet all of them:

1. **One process per arm.** Capacity is latched before the first compile (§7.5.3).
2. **Interleave at process level** (`A,B,A,B`), and **measure the noise floor for the graph you are
   measuring.** It is workload-dependent: ±3 % on a 2.1 ms graph, ~10 % / ~30 % on a 0.4 ms one. **A
   floor carried across graphs is not a floor.**
3. **Gate every run on a completed log delta**, never on a log read taken after `synchronize()`
   (§7.5.1) — and assert the gate's own reading, for the arm you believe you are in.
4. **Time with `torch.hpu.Event(enable_timing=True)` and `elapsed_time()`.** Wall-clock around
   `synchronize()` did not resolve a delta; events did.
5. **Do not try to raise the timed unit by repeating the work.** `--repeat 4` gave 351 us and
   `--repeat 8` gave 345 us — identical — because `torch.compile` CSEs identical subgraphs. **Vary**
   the work instead.
6. **Quote min and median together, and compare distributions, not single pairs.** A single pairing
   can differ by 2 % while the arms differ by 20 % typically.
7. **Re-gate placement on the exact graph every time.** Whether a tensor is placed at all depends on
   the graph's **construction**, not its shape (§7.3.1) — so a span that looks equivalent to one that
   was placed can lose placement entirely, silently.

**How §7.6's measurement scores against this list**, because the list exists to be applied rather than
admired:

| item | §7.6's measurement |
|---|---|
| 1, one process per arm | **met** — separate process per `SCRATCHPAD_SRAM_MAX_SIZE` value |
| 2, process-level interleave and a floor for *this* workload | **partly met — and the floor is what fails** — a floor was measured and it is **larger than the 32 MB effect** (three identical DRAM runs spanned **1,573–1,974 GB/s**, a **25 %** spread against that row's 18 % ratio), which is why that row is a **rate** and not an effect size; **process-level `A,B,A,B` interleaving is not evidenced** by anything we have |
| 3, gate on a completed log delta | **exceeded** — the gate is a **byte value in the graph dump** (256 B against `2 × buffer + 256 B`), not a race with an asynchronous log |
| 4, time with `torch.hpu.Event` | **UNKNOWN, and it is the item to check before reusing these numbers** — this is a C++ Synapse harness, so it has no `torch.hpu.Event`; §1.6 records exactly this trap (a per-repetition `launch → sync` read as a device time: bench median 622.1 µs against a device union of 530.5 µs). The per-rep host overhead of *this* harness was not measured. **If it is non-zero it depresses both arms equally and therefore compresses the ratio**, so 10.2 % / 1.18× are lower bounds rather than inflated |
| 5, vary the work rather than repeat it | **met** — the work was varied (buffer size, and later kernel form) rather than the timed unit being raised by repetition; where reps exist they are reported as a distribution |
| 6, min and median together, distributions not single pairs | **met for the 4 MB pair** (three reps, non-overlapping); **explicitly not met for 32 MB**, which is flagged as one SRAM rep against the best of three DRAM runs |
| 7, re-gate on the exact graph | **met** — one kernel, one graph per arm, gated every run |

**The rule that follows is the general one: a measurement is only as good as its weakest item, and
naming which item it is makes the number usable.** Here **four and a half of the seven** are met or
exceeded, item 2 is **partly met**, item 4 is **UNKNOWN**, and the consequence is stated rather than
assumed: the **4 MB** figure is usable as an effect size at its stated scope, and the **32 MB** figure
is usable as a *rate*, not as an effect size.

## 7.7 Safety: how to brick a card, and how not to

**`synMemCopyAsync` does not validate device addresses.** It returns **`synSuccess`**, issues the
transaction, and an unmapped destination surfaces **asynchronously as a DFA**.

**MEASURED — we did this.** A read-only sweep that walked past `SRAM_BASE + SRAM_SIZE` into the
unmapped space above the aperture, plus a read of a raw DRAM-window address outside any allocation,
produced two DFAs.
Uncorrected events went **3 → 450**; the device dropped to 0 °C / 50 W / Perf N/A, then recovered to
Operational but **required a hard reset** (`hl-smi -r -i <pci-bus-id>`) to clear the counters.

**Rules:**

1. Only ever copy to an address **inside `[SRAM_BASE, SRAM_BASE + SRAM_SIZE)`** or inside a range you
   allocated yourself.
2. **Bound every sweep by `SRAM_BASE + SRAM_SIZE`, read from `synDeviceGetAttribute` at runtime and
   asserted** — not hardcoded twice.
3. **If you cannot establish an address's provenance, do not copy to it.**
4. `hl-smi -Q bus_id,module_id,ecc.errors.uncorrected.volatile.total -f csv` **before and after**;
   abort on any rise.
5. **Host staging buffers must come from `synHostMalloc`.** A plain heap buffer returns
   **`synMappingNotFound (34)`**, which is a statement about the **host** side and says nothing about
   the device address. This exact confusion produced a retracted conclusion — *"SRAM is not
   DMA-addressable"* — in this project's own notes.

> **This is the same error class as chapter 1's "a return code is not a measurement."** Here it costs
> a card rather than a number: `synDeviceMalloc` returned success and relocated, `synMemCopyAsync`
> returned success and faulted, and `synGraphCompile` returned success on a kernel writing to the
> wrong address. **Every Synapse call in the fatal run returned `synSuccess`.**

**Device numbering — four orderings that disagree.** `HABANA_VISIBLE_MODULES` id, PCI bus id,
`/sys/class/accel` node, and the `hl-smi` display row are **four different numberings**. On the box
behind this chapter:

| Module ID | PCI bus | accel node | hl-smi display row |
|---|---|---|---|
| 0 | `0000:43:00.0` | `accel2` | `[2]` |
| 1 | `0000:44:00.0` | `accel6` | `[6]` |
| 2 | `0000:19:00.0` | `accel0` | `[0]` |
| 3 | `0000:1a:00.0` | `accel3` | `[3]` |

**Resolve every device with `synDeviceGetPCIBusId` before touching it, and print the bus id in every
run log.** Never by index, never by row order. (§3.7 has the pinning rules that follow from this;
this table is the raw mapping.)

---

## 7.8 What this chapter does not establish

| # | open | what would settle it |
|---|---|---|
| 1 | **Whether the MME can read SRAM at the hardware level.** The compiler evidence is strong (`m_mmeBundles`, and `flatten_mme` propagating SRAM onto MME operands) — but a clean isolated MME-from-SRAM bandwidth test has not been run, and §7.6 is a **TPC** result that does not bear on it. What §7.6 does add is the compiler half: an MME operand marked `inSram: true` with `Num of slices: 1` does **not** shrink the workspace, so the placement adds a staging copy rather than replacing one. | A kernel pair differing only in operand placement, with `inSram: true` gated, at a size large enough to be bandwidth-bound — **and the workspace shrinking in the SRAM arm**, which is the part that has never been observed. |
| 2 | **Whether SRAM is faster than HBM for a compute engine, and by how much — PARTLY ANSWERED on the TPC path (§7.6).** **MEASURED:** **+10.2 %** at 4 MB × 16 with three reps per arm and **non-overlapping** distributions (SRAM min 1,123.7 above DRAM max 1,046.1), and **1.18×** at 32 MB × 16 (2,338 against 1,974 GB/s — **one SRAM rep against the best of three DRAM runs**, so best-vs-best), both gated on the graph's `workspace_size`. The old per-TPC figure (1.24 GB/s SRAM against 1.02 GB/s DRAM, 1.21×) came from a single TPC doing scalar accesses with index-space parallelism held at 1 — **not a bandwidth figure**, and superseded as a bandwidth statement. (Host DMA into SRAM is PCIe-bound at ~19 GB/s and says nothing about the pool either.) **Still open: (a) the MME path**, the only engine with the appetite to reach the pool's bandwidth, where the compiler adds a staging copy rather than replacing one (item 1); **and (b) which constraint binds above ~2.3–2.4 TB/s** — one SRAM point landing inside an HBM-fed band cannot separate "the array caps here" from "SRAM's supply to a TPC happens to be here" (§7.6), and 2,338 GB/s is where our instrument stopped, not where the pool stopped. | **(a)** an MME arm reading an SRAM-resident operand with the **workspace shrinking**; **(b)** the discriminating test for the binding constraint — a TPC kernel pushed **past** ~2.4 TB/s aggregate (which nothing in this project has reached), or an engine with more appetite per core, which is (a) again. |
| 3 | **Whether the orphaned memory-strategy optimizer works on Gaudi 2.** It has never executed in 1.24.2. If `STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM` worked, **all activations** would leave DRAM — a much larger prize than any single tensor. | Call it behind a flag on a small graph and check `maxRange.size` and `inSram` counts against the slicer's. |
| 4 | **Whether production model weights actually land in SRAM.** They are persistent parameters (§7.3.1's *run A*), but production MoE weights are **produced by the fp4→fp8 dequant**, which is arithmetic — so they are *structurally* placeable (a **derived** inference, not a measurement). §7.3.1's construction finding then decides it: a single `bmm` places, per-expert GEMMs plus a `cat` does not. So this is **plausible but unproven**, and it depends on how the model builds the graph. | `inSram: true` gating on the **real** engine graph, per layer, with the construction named. |
| 5 | **How much of the 48 MiB a real decode step can hold.** The aux route caps at **~10 MiB per tensor at the default knob** (and is raiseable — §7.3.2 puts a 32 MB aux in SRAM), the slicer's budget is ~47.5 MB, and §7.3.5 shows the one tile class we can size fits with ~10 MB of headroom — but what fraction a *production* graph can use before fragmentation binds is unmeasured. | Sweep `SRAM_SLICER_MAX_CAPACITY_BYTES` on a real graph and read the placement the slicer achieves at each cap. |
| 6 | **Why the control arm is heavy-tailed.** CV 50 % against SRAM's 7 %, with two 2–3× outliers in 20 runs (the span A/Bs, §7.6). If intrinsic, part of SRAM's value is **tail removal** rather than mean improvement — which is a different and possibly larger claim. | More control runs and a distribution test rather than a range comparison; then a **p99 step** measurement rather than min. |
| 7 | **The n256 dequant path is unmeasured — it is currently unrunnable in the harness.** Its signature is `(ids, q16, s16, lookup, bool normal)` with **no channel tensor**; after that argument was fixed it fails *"V4.1 prepared MXFP4 requires contiguous inference inputs with matching dtype/device"*, not yet diagnosed. Note the n256 tile, unlike `prepared`, **has a token dimension** and would scale with batch. | Diagnose that check, then measure the n256 tile at BS = 1 and 2. |

---

## What to carry forward

1. **SRAM is 48 MiB at `0x1000fffffd000000`, one pool shared by 24 TPCs and 4 MMEs**, and
   `SRAM_BASE + 48 MiB` is the driver's `DRAM_PHYS_BASE`. **Copying past the aperture faults the
   card** — that is the measured fact; the size of any gap above it is UNKNOWN.
2. **You cannot place a tensor in it.** `synDeviceMalloc` relocates, the attribute is unsupported,
   a TPC has 32-bit pointers (a 64-bit address compiles to a *local* access and appears to work), and
   `AuxTensor.sramAlloc` is dead. **Placement is the compiler's decision** — but a **host DMA into the
   aperture** and a **TPC writing through an aux tensor** are both verified working (§7.3.2), which is
   how you *populate* or *observe* it, not how you place a tensor.
3. **The live slicer is `passes/sram_management/`, and it runs on every multi-node graph.** It is
   not the orphaned `optimize_memory_strategy` optimizer, which is dead code **in the 1.16 tree we
   read**.
4. **`ALL_IN_DRAM` is the default allocation *mode* and does not mean nothing is in SRAM** — ≥ 31 %
   of one port's decode recipes carry SRAM-window addresses.
5. **Eligibility is graph structure: a tensor needs a real compute producer.** A view does not
   qualify; an elementwise op does. **Give a weight a producer that costs nothing *extra*** — one
   elementwise pass, not a view.
6. **Observe, do not request.** Gate on the placement line for the operand under test, on
   `maxRange.size`, and — for an aux arm — on `workspace_size` against `2 × buffer + 256 B`; treat a
   0-byte `graph_compiler.log` as a capture failure; interleave the arms.
7. **SRAM reads measurably faster than HBM on the TPC path — +10.2 % at 4 MB with non-overlapping
   distributions, and 1.18× at the highest rate we saw (32 MB, one rep against the best of three) —
   and the gain is small because both arms land within ~10 % of the same per-core band, ~2.3–2.4 TB/s.**
   2,338 GB/s over 24 TPCs is **97.4 GB/s per core** against §1.6's **89.5 GB/s per core** envelope, and
   §4.1.2 measures the same band from the **HBM** side. That **points at** the array rather than the
   memory as the binding constraint — **INFERRED, not measured**: one SRAM point inside an HBM-fed band
   cannot separate "the array caps here" from "SRAM's supply to a TPC happens to be here" (§7.6).
   Reaching the pool's own bandwidth needs the **MME**, which the compiler cannot currently feed from
   SRAM without adding a staging copy. **§7.6.1 is what a *span-level* measurement still has
   to satisfy** — this one is a kernel, not a span.
8. **A return code is not a measurement, and here it costs a card.** `synMemCopyAsync` accepts any
   address and faults asynchronously.
