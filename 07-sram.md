# Chapter 7 — The on-die SRAM

**The 48 MiB SRAM is real, the compiler already slices into it, and you cannot place a tensor in it
from your own code.** Those three statements are the whole chapter, and they are not in tension: SRAM placement
is a decision the graph compiler makes, on criteria that are mostly about **graph structure**, and
the useful work is to make your tensors *eligible* and then **verify what it did** rather than to
request a buffer.

This chapter is worth reading before you plan anything that assumes SRAM, because **every obvious
route to it is closed and most of them fail without an error** (§7.4). It is also worth reading if you
have already decided SRAM is irrelevant: on a batched MME workload it is worth roughly **20 % on min
iteration time** (§7.6), and it is the only lever in this manual that changes *where bytes live*
rather than how they are moved.

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
bandwidth observed **anywhere** was **2,306 GB/s = 94 % of spec**, and nothing exceeded it. On-die
SRAM slower than off-die HBM is not credible, so a ceiling that never rises above HBM is consistent
with **nothing in that programme being SRAM-resident at a size that mattered** — which §7.2 explains.

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

### 7.3.2 A TPC's aux tensor: the one route you can drive directly

`AuxTensor.sramAlloc` is documented as *"This flag instruct GC to allocate the aux tensor in SRAM
memory"* and is a **dead field** — one declaration, no readers anywhere, and three GUIDs differing only in
`(sramAlloc, noInit)` produced an **identical** recipe address. **Do not use it.**

What actually decides aux placement is whether the buffer has data, and its size:

```
pData == nullptr  AND  write-space < GCFG_SCRATCHPAD_SRAM_MAX_SIZE (10 MiB)   ⇒ SRAM
```

`noInit` controls **initialisation**, not placement. **MEASURED**, same ELF and same glue, only
`bufferSize` changed:

| aux `bufferSize` | placement | workspace |
|---|---|---|
| 4 KiB | SRAM `0x1000fffffd080000` | 256 B |
| 1 MiB | SRAM `0x1000fffffd080000` | 256 B |
| 12 MiB | **no SRAM address** | 25,166,080 B |

So the usable ceiling is **~10 MiB per aux tensor** — *per tensor, not per node* — against a 48 MiB
aperture shared with every TPC and every concurrent kernel. Two SRAM aux tensors in one graph get
distinct contiguous buffers (`+0x80000` and `+0x81000`); **no collision**. **MEASURED**

**What this buys, stated plainly:** a declared **scratch buffer** up to ~10 MiB. It is **not** a way
to hold model weights. It pays off only for a span whose working set is small, hot and
bandwidth-bound.

**And the header comment on the neighbouring field is wrong too:** `auxiliaryTensorNr` is documented as
*"will be set by GC to MAX_AUX_NUM"* and arrives as **0** on first instantiation. A glue that treats
it as a capacity and returns `GLUE_FAILED` surfaces as **`synGraphCompile → synFail (26)` with a
0-byte build log**. The array is a real `std::array<AuxTensor, MAX_TENSOR_NR>`; index 0 is always
writable. **Do not gate on the incoming value.**

### 7.3.3 Graph tensors — the mechanism that matters, and it has no user-facing request

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

## 7.4 The routes that do not work — do not spend time on them

Each of these was tried and refused. Two of them **fail silently**, which is why they are here.

| route | result |
|---|---|
| `synDeviceMalloc` at a requested SRAM address | returns **`synSuccess`** and hands back a **DRAM** address. The header says the request is *"a hint"* and that malloc *"will succeed regardless"* — **the returned-address comparison is the only thing that catches it.** |
| `synTensorSetDeviceAddress` / `synSectionSetDeviceAddress` | **do not exist** |
| `MEMORY_ATTRIBUTE_SRAM = 4` | `// currently not supported` |
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

Three gates, and an arm that fails any of them is not the arm you think it is:

1. **`inSram: true` count, for the arm you believe you are in.** Not "the run succeeded".
2. **`maxRange.size` identical across arms.** If the clamp did not take effect, any timing under it
   is meaningless (§7.2.3).
3. **`hl-smi` uncorrected-event count before and after** (§7.7).

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

## 7.6 What it is worth

**The claim to carry, stated so it can be defended:** *SRAM placement of an MME weight operand
lowers min iteration time on a batched MME GEMM, typically by **~15–25 %**. The effect is real, but
it is **not a hard separation**.* **MEASURED** (a parallel port; artifacts unpublished)

The evidence — **10 gated runs per arm** at 50 iterations, interleaved, same graph, same work, only
placement differing, **pooled across 20 runs**:

| arm | min µs range | CV |
|---|---|---|
| **SRAM** | **353.0 – 423.4** | ~7 % |
| **control** (slicing off) | **411.3 – 1096.3** | ~50 % |

**Typical min improvement is ~15–25 %, and the pooled extremes OVERLAP** — the best control run beat
the worst SRAM run. So this is **a real effect, not a hard separation**; on median the distributions
overlap in some pairings and separate in others. *(An earlier, smaller sample — 5 runs per arm —
showed no overlap on min at all. The larger sample does not support that, which is why the smaller
one is not quoted here.)*

> **Do not quote a single percentage as a floor.** "~24 %" was one pairing in the small sample and
> "~20 %" was that sample's typical; the larger sample's typical is **~15–25 %**, and its worst
> pairing sits inside the overlap. **Quote the range, and say the distributions overlap.**

**Budget pressure is real, and the case that matters was exercised.** The test weight was **47.2 MB**
against a **~47.25 MB** budget — it fits **only** by 3-way slicing with double buffering (epoch
40.0 MB), and placement did not degrade. That is the configuration worth knowing about: a tile that
*just* fits.

**An unexplained finding, and it may be the real value.** The control arm is **heavy-tailed** — CV
50 % against SRAM's 7 % — with two outliers of 560 µs and 1096 µs across 20 control runs while SRAM
never left 353–423 µs. If that tail is intrinsic, SRAM's benefit is partly the **removal of
occasional 2–3× slow iterations**, which would matter far more for tail latency than for the mean.
**Not established.**

**The instrument matters, and two things about it were learned the hard way:**

- **`torch.hpu.Event(enable_timing=True)` works and has `elapsed_time()`.** Recording events around
  the call and synchronising afterwards cut the spread enough to resolve a delta; **wall-clock around
  `synchronize()` did not.**
- **Repeating identical work inside a compiled region does not raise the timed unit.** `--repeat 4`
  gave 351 µs and `--repeat 8` gave 345 µs — identical — because `torch.compile` CSEs the copies.
  To raise the timed unit you must **vary** the work, not duplicate it.

**And the floor is workload-dependent — re-measure it per graph.** On a 2.1 ms workload it was ±3 %
on min; on this 0.4 ms one it is **~10 %** for the SRAM arm and **~30 %** for the control. **A floor
carried across graphs is not a floor.**

### 7.6.1 The negative result: none of this transfers to the dequant span

**Read this before planning around SRAM, because it closes the obvious next idea.**

The same A/B, run on the **fused fp4→fp8 dequant span** — the span that actually materialises the
expert weights — both arms gated (`inSram: true` = 3 vs 0):

| arm | min µs (5 runs) | mean | range |
|---|---|---|---|
| **SRAM** | 316.88, 327.65, 304.22, 302.94, 297.81 | **309.90** | 297.8–327.7 |
| **control** | 284.31, 319.89, 295.34, 255.58, 280.62 | **287.15** | 255.6–319.9 |

**SRAM is 7.9 % SLOWER on mean min, and the ranges overlap.** The delta is inside the control's own
floor **and it is the wrong sign**; medians are not resolvable either — one outlier flips the sign
between 6.2 % faster and 12.2 % slower.

**The best explanation, and it is derived rather than measured:** that span is dominated by the
dequant **compute**, not by HBM traffic on the tile, so relocating the tile removes little and adds
slicing and double-buffer overhead. The arithmetic agrees — writing 37.50 MB at 873 GB/s is ~43 µs
and reading it at 1,889 GB/s ~20 µs, i.e. **~63 µs of a ~300 µs span: roughly 21 % traffic, 79 %
compute.** There is no large traffic win to collect.

> **⇒ The consequence, plainly: avoiding HBM materialisation of the fp8 tile is NOT the lever for that
> span. The dequant compute is.** The ~15–25 % above was measured on a `mul`-producer bf16 GEMM
> graph; **it does not transfer**, and a reader who carries it over will spend a week on a span where
> the effect has the wrong sign.

**A second negative result, about method.** An earlier A/B showed **no** measurable benefit — 1.0 %
faster on min, inside the floor — and the reason was **the graph, not SRAM**: it regenerated its
47 MB weight with `torch.randn` *inside* the compiled region every iteration, so the timed region was
dominated by weight generation and the saving on the weight *read* was diluted below resolution.
**The instrument was fine; the graph was wrong.** The fix was a producer chain that costs nothing
*extra* — one elementwise pass, not a view, since a view does not qualify (§7.3.1) — which is also
the production-relevant shape.

---

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
| 1 | **Whether the MME can read SRAM at the hardware level.** The compiler evidence is strong (`m_mmeBundles`, and `flatten_mme` propagating SRAM onto MME operands), and §7.6's result is consistent with it — but a clean isolated MME-from-SRAM bandwidth test has not been run. | A kernel pair differing only in operand placement, with `inSram: true` gated, at a size large enough to be bandwidth-bound. |
| 2 | **Whether SRAM is actually faster than HBM for a compute engine, as a bandwidth figure.** Unmeasured. The one per-TPC number on record (1.24 GB/s SRAM vs 1.02 GB/s DRAM, 1.21×) came from a single TPC doing scalar accesses with index-space parallelism held at 1 — **not a bandwidth figure**. Host DMA into SRAM is PCIe-bound at ~19 GB/s and says nothing. | Keep the kernel body fixed and vary TPC parallelism; report GB/s per engine, not an aggregate. |
| 3 | **Whether the orphaned memory-strategy optimizer works on Gaudi 2.** It has never executed in 1.24.2. If `STATIC_TENSORS_PREFETCHED_ACT_IN_SRAM` worked, **all activations** would leave DRAM — a much larger prize than any single tensor. | Call it behind a flag on a small graph and check `maxRange.size` and `inSram` counts against the slicer's. |
| 4 | **Whether production model weights actually land in SRAM.** They are persistent parameters (§7.3.1's *run A*), but production MoE weights are **produced by the fp4→fp8 dequant**, which is arithmetic — so they are *structurally* placeable (a **derived** inference, not a measurement). §7.3.1's construction finding then decides it: a single `bmm` places, per-expert GEMMs plus a `cat` does not. So this is **plausible but unproven**, and it depends on how the model builds the graph. | `inSram: true` gating on the **real** engine graph, per layer, with the construction named. |
| 5 | **How much of the 48 MiB a real decode step can hold.** The aux route caps at ~10 MiB per tensor and the slicer's own budget is ~47.5 MB, but what fraction a production graph can *use* before fragmentation binds is unmeasured. | Sweep `SRAM_SLICER_MAX_CAPACITY_BYTES` on a real graph and read the placement the slicer achieves at each cap. |
| 6 | **Why the control arm is heavy-tailed.** CV 50 % against SRAM's 7 %, with two 2–3× outliers in 20 runs (§7.6). If intrinsic, part of SRAM's value is **tail removal** rather than mean improvement — which is a different and possibly larger claim. | More control runs and a distribution test rather than a range comparison; then a **p99 step** measurement rather than min. |
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
6. **Observe, do not request.** Gate on `inSram: true` and on `maxRange.size`; treat a 0-byte
   `graph_compiler.log` as a capture failure; interleave the arms.
7. **What it is worth: ~15–25 % on min iteration time, and the distributions OVERLAP** — a real
   effect, not a hard separation. **It does not transfer to the dequant span, where it is 7.9 %
   slower** (§7.6.1). The noise floor is workload-dependent; re-measure it per graph.
8. **A return code is not a measurement, and here it costs a card.** `synMemCopyAsync` accepts any
   address and faults asynchronously.
