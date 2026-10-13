# 4. The TPC execution model

**Status: draft.** Tags follow the manual's convention — **MEASURED** / **source read** / **ASSUMED** /
**INFERRED** / **UNKNOWN** ([README](README.md#how-to-read-this-manual)). Every number below is
restated from the named archived artifact beside it, not from memory, and each carries the conditions
it is valid under.

**What this chapter is.** The layer *below* chapters 1–3: how one TPC core issues instructions, what a
TPC kernel's cost is actually denominated in, and what the vector registers, dtypes and source forms
do. Chapters 1 and 2 stop at "24 TPC cores, and here is how to make them overlap with the MME and DMA".
This chapter opens the TPC and answers the three questions a porter hits as soon as they write a kernel
of their own:

1. **How many kinds of concurrency are there inside a TPC, and what is each one's unit?** (§4.1)
2. **What is the cost currency of TPC code?** — and the answer is *not* the instruction. (§4.2)
3. **What do the vector registers, the dtypes and the source forms actually do?** (§4.3–§4.4)

It closes with the recipe for checking any of this yourself from a listing (§4.5) and with its own
UNKNOWN register (§4.6).

> **Why this is chapter 4 and not chapter 1.5.** The material is the layer *under* chapter 1, but
> renumbering the three existing chapters would rewrite their cross-references and their history. The
> manual appends; it does not renumber. **Read it after chapters 1–2** — it assumes their vocabulary
> (engines, recipes, regions, the `MAX` rule).

**One naming trap, stated up front.** The word **"bundle"** is overloaded. A **VLIW bundle** (this
chapter) is one issue group of up to four instructions inside a TPC. A **`bundle_index`** in a
`DUMP_POST_GRAPHS` dump ([§1.3](01-basics.md), [§2.10](02-regions.md)) is *graph* bundle slicing — a
grouping of IR nodes, unrelated to issue. Do not carry a fact from one to the other.

---

## 4.1 Three levels of concurrency — each with its own unit

There are **three** independent kinds of concurrency in TPC code. They compose, they are easy to
confuse, and each has a different unit:

| level | the unit | the mechanism | where |
|---|---|---|---|
| **intra-instruction** | the **VLIW bundle** — 4 slots | up to 4 instructions issue in one cycle, one per slot | §4.1.1 |
| **intra-TPC** | the **TPC** | the vector ALU pipe and the memory system drain **asynchronously** | §4.1.2 |
| **inter-node** | the **recipe** | TPC ∥ MME ∥ DMA execution windows intersect | [§2.2](02-regions.md) — not restated here |

**Getting the level wrong is the classic mistake.** "It didn't overlap" at one level is not evidence
about another: a kernel can be perfectly packed at the bundle level and still be memory-bound, and a
recipe can carry a flawless bundle schedule and have zero cross-engine overlap.

### 4.1.1 Intra-instruction: `{...}` is a 4-slot VLIW bundle

A `{...}` group in a `tpc-clang -S` listing is **not a textual grouping — it is the machine's issue
group**, a VLIW bundle with exactly **four slots**, named **`load`, `scalar`, `vector`, `store`**.
The four slots and their names come out of the compiler's own interface, not off the device. The
evidence:

* `tpc-clang` has four asm formats. `-mllvm --tpc-asm-format=1` prints **one line per instruction, all
  slots**; `=2` prints **one line per bundle, four `;`-separated fields** in a fixed slot order
  (read off 134 lines of one kernel's `=2` listing: field 0 = loads/`set_indx`/`event`, field 1 =
  scalar ALU/branch, field 2 = the vector op, field 3 = stores).
* The compiler's own diagnostic vocabulary names the four slots verbatim — `load scalar vector store`,
  next to *"Too many slots"* — and its verifier messages are phrased in bundles: *"THREAD_SYNC must be
  scheduled alone in a bundle"*, *"Only one immediate field is allowed in a VLIW"*.
* The grouping is produced by the **packetizer**, not the printer: `-mno-packets`,
  `-disable-instruction-scheduling` (*"Disable instruction scheduling and bundling"*),
  `-mllvm --tpc-single-packets` and `-mllvm --disable-tpc-packetizer` each **change the listing**.

**`source read`** *(`/opt/habanalabs/bin/tpc-clang`, TPC C/C++ compiler **2.7.33**, clang 14.0.5: its
asm-format help, its slot vocabulary, its verifier strings, and the listings its own flags produce.)*

**The decisive identity, and the one to remember.** The *default* (`--tpc-asm-format=0`) listing prints
each bundle's **real** instructions, and **a single `NOP` for a bundle that has none**. So

> **instruction lines in a default listing == `Σ_bundles max(#real instructions in bundle, 1)`
> == bundles + (#bundles carrying ≥2 real instructions).**

Exact on four kernels: `vecop_i8_copy` 94 = 94, `vecop_i8_wide7` 169 = 169, `vecop_i8_sel_K8`
225 = 225, `vecop_i8_decbody` 394 = 394. **MEASURED.** *(This identity is also a trap. An earlier fit
in this project regressed device time on the default listing's line count and concluded "the device
charges every emitted instruction, NOPs included". The regression was right; the mechanism it named
was wrong — it was regressing on the **bundle** count all along. NOPs inside an occupied bundle are
free. To count bundles exactly, use `--tpc-asm-format=2` (§4.5).)*

**How many instructions ride in a bundle?** At most **4** — one per slot — and in the shipping build,
almost never more than one. Whole-kernel census over 2,815 default-build bundles:

| real instructions per bundle | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| bundles | 1192 | 1487 | 120 | 12 | **4** |

and **95.2 % of the shipping build's bundles carry 0 or 1 real instruction** — **2,679 of 2,815**
(`(1192+1487)/2815`). No bundle in the 8,483 censused across three builds ever exceeded 4.
**MEASURED.** *(The 4-wide case needs a load + a scalar op + a store + a register move. The compiler
**never packs two vector ops** — there is only one `vector` slot.)*

*(**Provenance note, 2026-10-13.** The source report's prose says **93 %** in three places (§0 item 2,
§9, §10 item 2) and its own census does not reproduce it — the histogram above *is* that census, and
it gives **95.2 %**. 93 % is derivable from no population this census names (occupied bundles only:
91.6 %; all three builds pooled: 94.1 %; default + `vliw-td` only: 91.2 %). The only `0.93` in that
report is a **different quantity** — the fp4 body's real-instructions-**per-bundle** ratio
`70.75/76.00 = 0.9309` (§3.1, §7), i.e. the achieved *co-issue rate*, not a bundle population share.
That report now carries a dated annotation at each site. **Do not restore 93 %.**)*

**Consequence — the arithmetic floor.** A vector op's floor is **one bundle**, so

> `bundles/iteration ≥ #vector ops/iteration`, and the best possible wall is `#vector ops × 0.556 ns`.

The fp4 `dec()` body is **61.5 vector ops**, so its floor is **61.5 bundles** — not 4× less. Co-issue
can hide *load/store/scalar* work behind vector work; it **cannot** multiply vector throughput.
**INFERRED** (the floor is arithmetic; the 0.556 ns is §4.2.1).

### 4.1.2 Intra-TPC: the ALU pipe and the memory system overlap — a `MAX`, not a sum

The second level is *within* a single TPC and needs **no same-cycle issue at all**. The load/store
slots hand requests to the memory system, which drains them **asynchronously** while the vector pipe
keeps issuing. The measured relation is a **max**:

```
t_ALU = 6.436 + 0.8189 × (instr/iteration)   µs
t_mem = 15.334                               µs     (at the shape measured)
t     = MAX(t_ALU, t_mem)
```

**MEASURED.** *(Artifact: the four-arm device family at the production tile `[4096,2048]`, `span`
statistic — `Archive/by-date/2026-10-09/fp4_arith/README.md` §1.6; the re-fit and the `max` form are
in `Archive/by-date/2026-10-12/arith_review/FINDINGS.md` §3 and restated in `tpc-coissue/REPORT.md`
§5. **`t_mem = 15.334 µs` is the `varc` arm's span at that tile, not a hardware floor** — the
`5.954 µs` floor is the *copy* arm's, at a different shape, and belongs to `tpc-issue-rate-ceiling/`.
The `6.436 + 0.8189` coefficients are the `span` re-fit; the archived fit reads `6.471 + 0.8183`.)*
The 0.81 µs-per-instruction slope is the same family as chapter 3's "~0.81 µs per
instruction/iteration" — [§3.4.5](03-porting.md).

**This is a different mechanism from §4.1.1's co-issue, and one experiment separates them.** The
evidence is a **hard memory floor** that six different well-packed arms sit on exactly, at
`[16384,2048]`:

| arm (three of the six) | bundles/iter | real instr/iter | `dur med` | traffic | rate |
|---|---:|---:|---:|---:|---:|
| `sel_K1v` | 4.25 | 6.75 | 44.24 µs | 100.66 MB | **2.275 TB/s** |
| `sel_K8v` | 11.75 | 13.75 | 44.48 µs | 100.66 MB | 2.263 TB/s |
| `copyv` (2 tensors) | 2.75 | 4.75 | 27.60 µs | 67.11 MB | **2.431 TB/s** |

Six arms spanning **6.75–13.75 real instructions** all land at **44.2–44.5 µs** — dead flat — because
the memory system became the binding resource at **2.26–2.43 TB/s**. Arms **above** that floor
(`sel_K16v` 1.66 TB/s, `decbodyv` 0.52 TB/s) are **issue-bound**. **MEASURED.**

> **So: co-issue shrinks the issue term; memory overlap caps the total at the memory term.**
> Independent levers, additive where applicable. On the fp4 kernel the memory term is irrelevant
> (490–590 GB/s against a 2.3 TB/s floor — §4.2.4) and the issue term is everything.

### 4.1.3 Inter-node: TPC ∥ MME ∥ DMA

The third level is across engines, and the manual already has its law: the **four-condition overlap
law** — one Synapse submission, different engines, no dataflow edge, comparable durations; violating
any of the first three gives `CROSS` exactly `0.0000`, and the fourth is a **detectability** condition
whose violation gives a *small, non-discriminating* `CROSS` rather than zero
([§2.2](02-regions.md)). The reason it matters *here* is that it is a
**different kind of statement** from §4.1.1–§4.1.2: it is about **submission structure**, not
instruction scheduling. A perfectly packed kernel (§4.2) does nothing for it, and vice versa. For how
much overlap is *worth*, and the occupancy law that says the wall tracks the **union** and not the sum,
see [§2.6](02-regions.md) and [chapter 0](00-tldr.md#the-one-measurement-rule-that-changes-what-you-optimise).

---

## 4.2 The cost currency is the **bundle**, not the instruction

This is the chapter's most actionable lesson, so it gets the most evidence.

### 4.2.1 The device retires one bundle per cycle

`hl-smi` reports `clocks.limit.tpc = 1800 MHz` — and **no** `clocks.current.tpc` key exists at all, so
the instantaneous TPC clock is not observable on this stack (§4.6). At 1.8 GHz one cycle is
**0.5556 ns**, and the model

```
t = (bundles per iteration) × (iterations per TPC) × 0.5556 ns
```

predicts five arms over four independent kernels to **0.6–1.7 %**:

| kernel | bundles/iter | iters/TPC | predicted | measured `dur med` | meas/pred |
|---|---:|---:|---:|---:|---:|
| `decbody` @16384 | 75.00 | 5504 | 229.35 µs | 231.20 µs | 1.008 |
| `decbodyv` @16384 | 63.00 | 5504 | 192.66 µs | 194.08 µs | 1.007 |
| `sel_K16` @16384 | 67.25 | 5504 | 205.65 µs | 206.96 µs | 1.006 |
| `sel_K16v` @16384 | 19.50 | 5504 | 59.63 µs | 60.64 µs | 1.017 |
| `sel_K8` @16384 | 35.25 | 5504 | 107.80 µs | 109.12 µs | 1.012 |

**MEASURED.** All four slots of a bundle issue in that one cycle — nothing else explains 1.008 at 75
bundles/iteration carrying 70.75 real instructions. *(A sixth arm at that shape, `sel_K8v`, is
**omitted from the table on purpose, and the omission is the point**: it sits on the memory floor of
§4.1.2 — 44.48 µs at 2.263 TB/s — so an issue-only model does not describe it, and its `meas/pred` is
**1.238**. The model is validated on the five arms that are issue-bound, not on all six.)*

**This also reconciles two figures that look like different constants.** An earlier measurement of this
same hardware reported **0.544–0.621 ns per *emitted slot*** (the fp4 band within it is
0.578–0.586); this one reports **0.5556 ns per *bundle***. They are the same measurement, because in
the shipping build the default listing's line count *is* the bundle count to within the small number
of bundles that carry ≥2 real instructions (§4.1.1). The correct statement to carry is:

> **0.556 ns per VLIW bundle per TPC at 1.8 GHz, with up to four instructions riding in it.**
> *(Source for the slot-rate figures: the issue-rate-ceiling study — 0.544 ns/slot marginal best,
> 0.578–0.586 ns/slot for the fp4 kernel, i.e. within 6–8 % of the best observed. Artifact:
> `Archive/by-date/2026-10-12/tpc-issue-rate-ceiling/`.)*

### 4.2.2 The shipping scheduler leaves most of the bundle count on the table

The compiler **can** pack — the fp4 prologue carries a genuine 4-wide bundle, `mov.i8 %V13, 0x0 |
add.i32 %S12, %S35, 0x1 | mov.i8 %V12, 0x0 | set_indx %I2, b00010, %S35` — but by default it does not.
**The default pre-RA scheduler for this target is `source`**: `--pre-RA-sched=source` is
**byte-identical in `.text`** to passing no flag at all, which is how "the default is `source`" is
properly proved (§4.2.5).

The lever is a **scheduler flag, not a source change**:

```
-mllvm --pre-RA-sched=vliw-td
```

On the ten kernels measured it cuts bundles **1.19×–3.45×** on **the nine that change** — the tenth,
`vecop_i8_copy`, is 1.000 because it has nothing to pack — at **constant or higher** instruction
count:

| kernel | bundles (default) | bundles (`vliw-td`) | ratio | real instr (def → vliw) |
|---|---:|---:|---:|---:|
| `vecop_i8_copy` | 2.75 | 2.75 | 1.000 | 3.75 → 4.75 |
| `vecop_i8_sel_K1` | 7.25 | 4.25 | 1.706 | 5.75 → 6.75 |
| `vecop_i8_mid7` | 16.25 | 8.00 | 2.031 | 9.75 → 10.75 |
| `vecop_i8_sel_K4` | 19.25 | 7.75 | 2.484 | 8.75 → 9.75 |
| `vecop_i8_wide7` | 21.25 | 10.00 | 2.125 | 11.75 → 12.75 |
| `vecop_i8_dep7` | 31.25 | 10.75 | 2.907 | 11.75 → 12.75 |
| `vecop_i8_sel_K8` | 35.25 | 11.75 | 3.000 | 12.75 → 13.75 |
| `vecop_i8_sel_K16` | 67.25 | 19.50 | **3.449** | 20.75 → 21.75 |
| `vecop_i8_decbody` (fp4 `dec()` verbatim) | 75.00 | 63.00 | **1.190** | 66.75 → **71.75** |
| shipped `fp4_dequant_base` | 76.00 | 65.00 | 1.169 | 70.75 → **77.25** |

**MEASURED.** Read the last column: the packed build has **more** instructions. It is not winning by
doing less work.

### 4.2.3 The decisive controlled pair

Byte-identical source (copied to a different basename, nothing else changed), one process, interleaved,
`n = 60` launches, `warm = 5`, output verified **bitwise equal on device before any timing**:

| kernel | bundle ratio | instruction ratio | time ratio @4096 | time ratio @16384 |
|---|---:|---:|---:|---:|
| `decbody` (fp4 `dec()` verbatim) | 1.190 | 0.930 | **1.192** | **1.191** |

The faster build executes **7.5 % MORE instructions** and is **19.2 % faster**. The time ratio matches
the **bundle** ratio to **0.2 %**, and is **opposite in sign** to the instruction ratio. **Time is
charged per bundle.** **MEASURED.**

### 4.2.4 The win on the shipped kernel, and where the flag goes

**Measured on the production fp4 decode kernel** — in-process interleaved pair, `n = 60`, `warm = 5`,
**3/3 repetitions identical**, bit-exact:

| shape | default | `vliw-td` | speedup |
|---|---:|---:|---:|
| `[4096,2048,128]` | 67.760 µs | **56.160 µs** | **1.2066** |
| `[8192,1024,64]` | 67.680 µs | **56.080 µs** | **1.2069** |

Bundle ratio **84.50/69.25 = 1.220**; the 1-bundle/cycle model predicts 1.220; measured 1.207.
Instruction count goes **up** (74.75 → 81.25 non-NOP per iteration). *(The earlier, pre-guard revision
of the same kernel, measured separately, gives 61.281 → 52.240 µs at `[4096,2048]` and 61.201 →
52.160 µs at `[8192,1024]` = **1.173×**, bundle ratio 76.00/65.00 = 1.169 — the same result on the
same kernel's previous revision.)* **MEASURED.**

**Not memory-bound, so this is not a free-memory-shape artefact:** at `[4096,2048]` the kernel moves
33.6 MB per launch; achieved **490 GB/s** default, **590 GB/s** packed, against this box's measured
streaming floor of **2.26–2.43 TB/s** — **20–26 % of it**, a 4× margin. Both builds are
issue/bundle-bound. **MEASURED.**

**Where the flag goes.** The TPC build is a **plain shell script** that calls `tpc-clang` directly —
there is no CMake/Make/ninja for the TPC step. In `src/custom_ops/fp4_dequant/build.sh` (run with cwd
`src/custom_ops/fp4_dequant`) the compile is one line per kernel:

```bash
/opt/habanalabs/bin/tpc-clang -march=gaudi2 -O2 -c kernels/gaudi2/fp4_dequant_fwd_gaudi2.c \
    -o fp4_dequant_fwd_gaudi2.o
```

**The flag goes on that line** — and on the matching `-S` line, or the census disagrees with the
shipped object. A **zero-edit route** also works and produces the **identical `.text`**:

```bash
CCC_OVERRIDE_OPTIONS="+-mllvm +--pre-RA-sched=vliw-td" bash build.sh
```

`src/vllm/` and the vendored `vllm-gaudi` sources **do not call `tpc-clang`** at all — they consume
`GC_KERNEL_PATH` — so these `build.sh` files are the only place a TPC flag can be injected.
**MEASURED.**

### 4.2.5 Safety, and two traps

**Bit-exactness, widened.** The flag was validated across **14 distinct kernel source revisions**,
each built twice from byte-identical sources and compared **on device**. They appear in **22
arm-comparisons** — a body is counted once per family it is exercised in, so the list below does not
sum to 14: the production fp4 kernel + the E3a stub (2), `fp4_dequant_arith` (5 arms),
`fp4_dequant_lut` (3), `fp4_dequant_clamp` (3), `lutprobe` (4), `kchain` (5) = **22**. The domain is
the **exhaustive per-cell sweep** — all 256 input bytes × 64 values of
`k ∈ [−16, 47]` — plus four real shapes at the production distribution: **0 mismatches**. And the flag
**changes every one of their objects** (12 of the 14 get fewer bundles; the two unchanged ones are
trivial copies with nothing to pack), so the pass is **not vacuous**. **MEASURED.**

> ⚠ **Model-level safety is NOT established.** Every one of those is a microbenchmark or a standalone
> build. The pre-RA scheduler changes register allocation and hence the whole schedule; a kernel with
> different pressure could spill differently. **A model re-run with output comparison is the deciding
> experiment, and it is NOT RUN.** Do not read the 14-kernel pass as covering the model.

**Trap 1 — `.o` md5 is not a "same code" test on this compiler.** The object embeds its **whole `-cc1`
command line**, so *any* flag change (even a no-op) changes the `.o` md5. Compare the machine code
instead:

```bash
objcopy -O binary --only-section=.text <obj> <obj>.text && md5sum <obj>.text
```

That is how "the shipping default is `source`" is properly proved, and it explains a fact that
otherwise looks alarming: the frozen `libfp4_dequant_kernels.so` is **not md5-reproducible** from any
source revision, while its **`.text` is byte-identical** to a fresh build of the current tree.
**MEASURED.**

**Trap 2 — the flag's reach is bounded.** `src/custom_ops/*/lib*_kernels.so` is a *wrapper*: it
`dlopen`s `/usr/lib/habanalabs/libtpc_kernels.so` and forwards everything that is not ours. The flag
reaches **only the kernels we compile**; every **stock** TPC kernel is prebuilt by the vendor and is
**untouched**. The model-level win is bounded by our custom ops' share of TPC time. **MEASURED /
structural.**

### 4.2.6 Which schedulers pack — and the maintenance hazard

* **The packed set is `{vliw-td, fast, linearize}`** — 3 of the 7 available. **`list-ilp` does NOT
  pack** on any of the three sources tested (it lands on the default's bundle count), and **`fast` ≡
  `linearize` byte-for-byte in `.text`**. *(This corrects an earlier statement in this project's record
  that `list-ilp` also reached the packed schedule.)* **MEASURED.**
* **Why** these three pack and the others do not is **UNKNOWN**. The hidden-option descriptions do not
  distinguish them — `fast` is "fast suboptimal list scheduling" and `linearize` is "Linearize DAG,
  *no* scheduling", yet they emit identical code — and `--disable-tpc-packetizer` does not remove the
  difference. We did not instrument the packetizer, and **we do not guess**.
* ⚠ **Maintenance hazard.** `--pre-RA-sched=vliw-td` is an **LLVM hidden option** (`-mllvm`), absent
  from the driver's own `--help-hidden`, with no vendor statement that it is a supported interface.
  Whether it survives a compiler or Synapse version bump is **UNKNOWN**. The failure mode is
  **loud, not silent**: an unknown `-mllvm` option is rejected and `tpc-clang` fails the build — so the
  pin is real, but a stale build cannot pass unnoticed.

### 4.2.7 Headroom — and the 4× that was never there

| | bundles/iter | vs arithmetic floor |
|---|---:|---:|
| fp4 `dec()` body, shipped | 75.00 | 1.220× |
| fp4 `dec()` body, `vliw-td` | 63.00 | **1.024×** |
| shipped `fp4_dequant_base` | 76.00 | 1.192× |
| shipped `fp4_dequant_base`, `vliw-td` | 65.00 | **1.028×** |

**The available headroom on the fp4 stage is ≈1.19×, and `vliw-td` captures essentially all of it.**
The fp4 body is **VPU-bound** (61.5 vector ops in 63.00 bundles = 1.024× off its own floor), so **no
2–4× was ever available there** — co-issue can only hide non-vector work, and the fp4 kernel has little
of it. **Where co-issue *is* worth 2–3.4× is latency-bound serial chains** — the `sel_K*`/`dep7` arms,
which ship at 2.4–4.8× their own floor under the default scheduler. **Any kernel with a long serial
vector chain should be re-scheduled before it is believed to be at a ceiling.** **MEASURED.**

> **Cross-reference.** This is why chapter 3's caveat about the fp4 clamp's marginal cost — "the chain
> story explains it, but register pressure or a VLIW bundling change would explain it equally"
> ([§3.4.5](03-porting.md)) — now has a cheap control: count **bundles** in both builds. If the bundle
> count moves with the added instructions, the cost is a scheduling effect; if it does not, it is a
> chain effect.

---

## 4.3 Vector registers and dtype rules

### 4.3.1 One register, 256 bytes — one native lane width per element size

A TPC **vector register is 256 bytes**, and there is exactly **one** native lane count per element size:

| element size | lanes per register | examples |
|---|---:|---|
| 8-bit | **256** | `i8`, `u8`, fp8 `e4m3` (`_Float8_143`), fp8 `e5m2` (`_Float8_152`) |
| 16-bit | **128** | `bf16`, `f16`, `i16` |
| 32-bit | **64** | `f32`, `i32` |

**MEASURED.** There are **no** 128- or 64-lane variants of the 8-bit ops and no 32-lane variants of the
16-bit ops — `v_i8_*` is always a 256-lane type.

### 4.3.2 Cost is per **register**, not per element

Nine element types were timed on an identical skeleton (2 loads + 1 store per iteration, one 256-byte
vector on the contiguous dim, `#pragma unroll 4`), the **same shape in bytes**, the same process, the
same warm-up. **At every instruction count K the nine types are indistinguishable — worst spread
0.094 µs (0.79 %) at K=1, and ≤0.26 % for K ≥ 4.** **MEASURED.**

So the per-element ratios are exact:

> **per element: `i8 = fp8 : bf16 : f32 = 1 : 2 : 4`. per instruction: `1 : 1 : 1 : 1`.**
> **Only the per-element view is useful** — the per-instruction view says "int8 buys nothing", which is
> true and useless. int8/fp8 do **4×** the elements per unit time that f32 does, and **2×** what bf16
> does.

Per-element cost at K=8 (span ÷ iterations/TPC ÷ lanes):

| dtype | lanes | ns/element/TPC | Gelem/s per TPC | Gelem/s per card (24 TPC) |
|---|---:|---:|---:|---:|
| i8 / fp8 e4m3 | 256 | 0.0846 | 11.83 | **283.8** |
| fp8 e5m2 | 256 | 0.0844 | 11.85 | 284.4 |
| bf16 / f16 / i16 | 128 | 0.1688 | 5.93 | 142.2 |
| f32 | 64 | 0.3375 | 2.96 | 71.1 |
| i32 | 64 | 0.3382 | 2.96 | 71.0 |

**MEASURED.** *These are per-card **element** rates, not bandwidth: at K=8 the kernel is ALU-bound, not
memory-bound. Quoting the per-instruction ratio instead would say "int8 buys nothing"; the per-element
ratio says "int8 buys 4×". Both are true and only the second is useful.*

### 4.3.3 fp8: a first-class **non-arithmetic** citizen, a second-class adder

* **fp8 has a real native vector ALU for the non-arithmetic half.** `max`, `min`, `and`, `or`, `xor`,
  `sel_geq`, `cmp_*`, `brev`, `popcount`, `find_first`, `extract_exp` all lower to **one 256-lane fp8
  instruction** (`max.f8_143`, `sel_geq.f8_143`, …). The fp8 `sel` kernel times **identically to the
  i8 kernel** (29.554 vs 29.554 µs). **MEASURED.**
* **fp8 `+` is NOT native, and `*` is only half native.** An fp8 add runs through a **256-lane 32-bit
  accumulator**:
  ```
  convert.f8_143  all_lanes target_type=fp32   %A0, %V4
  add.f8_143                                   %A0, %A0, %V4
  convert.f32     all_lanes target_type=f8_143 %V0, %A0
  ```
  → **3 vector instructions per 256-lane fp8 add**; `*` is **2**. Measured **1.692×** the bf16 cost for
  the same source (`f8_mix_K8` 50.195 µs / 18 instr vs `bf16_mix_K8` 29.674 µs / 12 instr).
  **MEASURED.** *Practical: for an fp8 **add-heavy** inner loop, widening to bf16 and doing the add
  there is cheaper per element. Use fp8 freely for the non-arithmetic half.*
* `v_f8_add_b` / `v_h8_add_b` take **`float256`** (a 256-lane fp32 struct), **not** an fp8 vector —
  calling them with `_Float8_143 × 256` is a hard compile error. The working spelling for an fp8 add is
  the **C `+` operator** on the fp8 vector type. **MEASURED.**

### 4.3.4 Source form: two spellings, one listing — and a folding hazard

* **`v_<dt>_<op>_b(a, b)` and the plain C operator emit byte-identical listings.** Use either.
* **A lane-wise scalar loop does not compile at all** — element access on a vector type is a hard error
  (`error: access to elements of type 'char256' … is not allowed`). **There is no source-level
  scalarisation case on this machine:** the classic "did it scalarise?" failure mode cannot occur,
  because the code will not build. The realistic hazards are the **opposite** — the compiler silently
  **reducing** the op count (folding) and silently **raising** the cost (an op that is not one
  instruction). **MEASURED.**

**The folding hazard — the op count you get is not the op count you wrote:**

| source | intended | emitted | what happened |
|---|---:|---:|---|
| `z = x + x` | 1 add | `shl.i8` | `x+x → x<<1` |
| `z = x & y` ×8 | 8 `and` | **1 `and.i8`** | `and` is idempotent |
| `max` chain ×8, **f32** | 8 `max` | **1 `max.f32`** | `maxnum` folds in InstCombine |
| `max` chain ×8, **bf16 / i8** | 8 `max` | **8** | the builtin is opaque — does **not** fold |
| `z = x+y; z = z+x; …` ×8, i8 | 8 `add` | `1 mul.i8` + 4 `and.u32` + 2 `shl.i8` | reassociation + strength reduction |

**MEASURED.** *The same source folds differently per dtype — `max` chains are fold-proof for bf16/i8
but not for f32; `and` chains fold for every dtype. This is why a listing must be read.*

**The fold-proof idiom** used to build the K-sweeps above — exactly K native ops for all nine dtypes:

```c
z = v_<dt>_sel_geq_<dt>_b(z, x, y, z);   /* (z >= x) ? y : z -- non-linear, cannot fold */
```

### 4.3.5 Structuring: independent work goes **inside** the body

* **4-way ILP inside the body is worth 1.47× at *identical* instruction count.** Two kernels with the
  same dtype, shape, non-NOP count (11) and process, differing only in dependency structure: `dep7`
  (one serial chain) emits **21 NOPs** and runs at **26.713 µs**; `wide7` (4 independent accumulators +
  a combine tree) emits **11 NOPs** and runs at **18.180 µs** — ratio **1.4694**, against an
  **emitted-slot ratio of 32/22 = 1.455**. It reproduces on all four dtypes. **MEASURED.** *This is the
  single biggest lever in this section: the latency spine `z = f(z, …)` costs the compiler ~3 NOPs per
  op; independent accumulators cost nothing extra.*
* **`#pragma unroll` is NOT the lever.** At fixed chain length, `unroll` 4 → 8 → 16 makes every kernel
  **slower** — up to **+10.7 %** (K=8: 29.554 → 30.873 → 32.713 µs). The unrolled bodies are
  independent, but the scheduler does not interleave them and the larger body costs index-space setup.
  **Independent work must be exposed *within* the loop body, not across unrolled iterations.**
  **MEASURED.**

### 4.3.6 Pack / unpack

* `pack.i8` and `unpack.i8` are **one instruction per 256-byte vector** each.
* A `pack` → 8-op ALU → `unpack` body costs **+3 instructions (+25 %)** and **+23.3 % time**
  (36.434 vs 29.554 µs) — i.e. the round trip is worth **about 2 ALU ops** on an 8-op body.
* **Chaining** pack ops costs an extra `mov` each: 4 chained packs = 4 `pack.i8` + **3 `mov`**. Chain
  them through registers rather than nesting if the `mov` matters.
* `unpack.i8`'s operand already carries the structure you usually want
  (`source_group=0 element_stride=2 group_half=0`) — a single instruction with a switch-selected
  window. That is why `pack`/`unpack` at one instruction each is the right primitive.

**MEASURED.**

---

## 4.4 `shuffle.i8` — a correction, recorded with its calibration

This is the chapter's **correction** entry, written the way a correction should be: what was claimed,
what was measured, what is now **refuted**, and what is still **open**.

**What was claimed.** A static reading of the vendor's index encoding plus a 2026-09-18 device sweep
said `shuffle.i8` implements `DEST[i] = a[b[i]]` for index bytes with bit 7 set, so a 16-element window
could be replicated across 256 lanes in **one instruction**.

**What was measured — a mapping-independent test.** A device probe with a **flat ruler**
(`a[L] = L`, so the source lane of every output value is directly readable) and a **constant index**
vector (so any `DEST[L] = f(b[L])` semantics must produce a constant output — and it does not):

```
out[L] = a[ (L & ~3) | (b[L] & 3) ]     if (b[L] & 0x3C) == 0 and bit 7 of b[L] is set
       = income                          otherwise
```

i.e. it **permutes bytes only within each 4-byte-aligned group, and the source group is the
destination lane's own**. The load-bearing evidence is an **impulse** test: with `a` zero except
`a[4] = 1` and a **constant** index `b = 0x80`, the output is **nonzero at lanes 4,5,6,7** — which no
per-lane index map of the documented form can produce. The enabled index bytes are exactly
**`{0x80..0x83, 0xC0..0xC3}`** — bit 7 set **and** bits 2–5 clear; **bit 6 is ignored**; every other
byte yields `income`. A constant index of `0x80..0x83` gives **64 distinct values** in a 256-lane
window — the signature of a group size of **4** (64 = 256/4), not 32 and not 256. **MEASURED.**

**There is no escape hatch.** Compiling the identical probe with `-DSWITCHES=n` for
`n ∈ {0 … 262144}` emits a **byte-identical** `shuffle.i8` — unlike `lookup_1c`, whose `switches` bit 3
(`SW_LUT_PTR`) selects the generic table form, `shuffle` has **no switch that changes its operand
form**. **MEASURED.**

**Refuted:** the documented cross-lane `DEST[i] = a[b[i]]` semantics, and therefore the
16-element-window replication premise that rested on it. **The 2026-09-18 sweep is not reproducible**
with this probe (its observed reach was 32 lanes; ours is 4), and its kernel source is not archived, so
the two measurements cannot be reconciled.

**What this is NOT.** It is **not** "`shuffle` is a 4-byte permute, full stop". **UNRESOLVED:** whether
this is an **ISA property** or **tpc-clang's lowering** — the listing shows the instruction emitted
correctly and only once, so nothing looks mis-lowered, but we did not disassemble the TPC ELF. And the
author's own **bare-kernel control was INCONCLUSIVE**: with the glue's `(256,4)` access pattern it
wrote only ~4 of 1024 output lanes, so it exercised 4 lanes and cannot separate structure from
semantics. **The honest statement is: the documented semantics do not hold under the tested call
structure, and the exact trigger is open.** A purpose-built 1-D access pattern is what would settle it.

*(For completeness, the instruction is genuinely cheap: one `shuffle.i8` per 256 lanes, priced like one
ordinary ALU op — 12.034 µs against the 5.954 µs copy floor, the same +6.08 µs as one `sel`. **This
price is the dtype-sweep study's, not the `lut` studies':** `vector-op-throughput/REPORT.md` §Q4.)*

---

## 4.5 How to verify any of this yourself

Everything above comes out of a **listing**, and reading one is three steps.

**1. Compile it.** The production build uses `-c`; for the listing use `-S`:

```bash
/opt/habanalabs/bin/tpc-clang -march=gaudi2 -O2 -S kernels/gaudi2/<kernel>.c
```

**2. Count *bundles*, not mnemonics.** The default listing's line count is `Σ_bundles max(real,1)`
(§4.1.1), which is the bundle count to within the few bundles carrying ≥2 real instructions; multiply
by `0.556 ns × iterations/TPC` (§4.2.1). To count bundles exactly, and to see the four slots
separately:

```bash
tpc-clang -march=gaudi2 -O2 -S -mllvm --tpc-asm-format=2 <kernel>.c   # 4 ';'-separated fields/line
```

**3. Check the *operands*, not just the count — because a count cannot distinguish a working op from a
dead one.** The canonical instance is in this project's own history: the broken W1 LUT variant and the
corrected LUT variant have **byte-identical instruction histograms** — both **22 instructions per
iteration** (`lookup_1c=4 unpack.i8=4 and.i8=2 ld_tnsr=2 or.i8=2 st_tnsr=2 add.i8=1 max.i8=1 min.i8=1
shl.i8=1 shr.i8=1 set_indx=1`). They differ **only** in the `lookup_1c` operand form:

```
broken:     lookup_1c  BV8_0 %V14, %V10, 0x0              <- ROM form, immediate 0x0
corrected:  lookup_1c  BV8_0 lut_ptr %V14, %V10, %S2     <- table fetched through BASE2_VA
```

plus the corrected kernel's tensor-2 base-address MMIO pair (`ld_l mmio %S2, 0xa0` / `0xa4`;
`st_l mmio 0xcd8, %S2` / `0xcdc`). The broken kernel's table load was **dead-code-eliminated**, so it
never read its input — and no instruction count could show that. **MEASURED.**

**The rule this generalises to, and it is the reason to read operands:**

> **An instruction count is not evidence that an operation happened.** When an input is consumed
> through a side channel (a base address in MMIO rather than an `ld_tnsr`), the only proof that the
> input is read is the **operand** of the instruction that uses it.

An operand-aware gate therefore checks **five** things, not one: **(a)** per-mnemonic emission counts
(catches both folding *down* and over-emission *up*), **(b)** required operand forms, **(c)**
forbidden forms, **(d)** **input reachability** — every declared input tensor must be read, either
as an `ld_tnsr …, 0x<id>` source or by fetching its base address, and **(e)** loads/stores per
iteration against the declared count (one load per element is the scalarisation signature). **(d) is
the check that fails the dead kernel.** **MEASURED.**

**And for builds, compare `.text`, never `.o` md5** (§4.2.5).

*(This chapter has no `examples/04-execution-model/`: the three steps above are the whole instrument,
and they run against any TPC kernel you already have.)*

> ⚠ **The instrument does not exist for the TPC clock.** `hl-smi` on `hl-1.24.0-fw-62.6.2.0` exposes
> **no** TPC engine/cycle/clock counter of any name. Its clock keys are `clocks.current.soc`
> (1650 MHz flat), `clocks.max.soc`, `clocks.limit.soc` and `clocks.limit.tpc` (1800 MHz); its
> utilisation keys are `utilization.aip` and `utilization.memory`. `hl-smi -Q tpc_tpc_cycles`
> and `-Q clocks.current.tpc` both return *"Invalid --query-aip parameter"*. This is **"no such key on
> this build"** (the whole key list was enumerated), **not** "we could not look". **UNKNOWN** whether a
> newer or privileged build exposes them.

---

## 4.6 What this chapter leaves UNKNOWN

Stated rather than omitted, per the manual's convention. Each entry says what would settle it.

| # | UNKNOWN | status | what would settle it |
|---|---|---|---|
| 1 | **Whether 4 slots is architectural or a compiler maximum** | every one of 8,483 censused bundles has ≤4 real instructions and the compiler names exactly four slots — so 4 is *at least* the compiler's maximum; whether the silicon has a fifth resource (e.g. a second scalar pipe) is unchecked | a TPC disassembly, or a vendor statement |
| 2 | **Why `{vliw-td, fast, linearize}` pack and the others do not** | open — `fast` ≡ `linearize` byte-for-byte is unexplained by their descriptions | compiler instrumentation (`-print-after`, a debug build) |
| 3 | **The TPC clock in Hz** | **unobservable** on this build — no counter of any name. Every figure above is in ns; any cycle figure is **INFERRED** | a newer/privileged `hl-smi`, or a firmware route |
| 4 | **Whether `vliw-td` is safe for the whole model** | **NOT RUN** — bit-exact on 14 kernel bodies over the exhaustive domain plus four real shapes, but **no model run** | a model run with output comparison |
| 5 | **`shuffle.i8`: ISA or lowering?** | **UNRESOLVED** — the bare-kernel control was **INCONCLUSIVE** | a probe with a purpose-built 1-D access pattern |
| 6 | **Whether `--pre-RA-sched=vliw-td` survives a compiler/Synapse bump** | **UNKNOWN** — it is an LLVM hidden option with no vendor statement. The failure mode is loud (an unknown `-mllvm` option fails the build), so a stale build cannot pass unnoticed | a vendor statement, or the next compiler release |
| 7 | **The exact fixed per-launch term** | estimates range **0.8–4.0 µs** across fits, and the cross-arm fit and the per-kernel size fits disagree on the marginal slot rate (0.580 vs 0.544 ns) at the ~6 % level | a fit that identifies the second-order term (outer-loop / first-touch), or a clock that can be pinned |
| 8 | **Whether the bundle win generalises in wall time to the other kernel families** | **NOT MEASURED** — bundle ratios exist for 12 more kernels (up to 1.84×); only the fp4 kernel was timed | time the `lutptr` / `lut` arms |
| 9 | **The TPC ELF encoding** | no `tpc-objdump`/`llvm-objdump` in the image. `.text` bytes per bundle are 21.3–23.1 across seven kernels (~22), consistent with a fixed-width bundle word plus a variable-length immediate extension — but that is **INFERRED** from byte counts | a disassembler |

---

## What to carry forward

1. **Three concurrency levels, three units** — the VLIW bundle (intra-instruction), the TPC's async
   memory drain (intra-TPC), the recipe (inter-node, [§2.2](02-regions.md)).
2. **The cost currency is the bundle, not the instruction.** `t ≈ bundles × 0.556 ns`; the default
   listing prints one line per real instruction, plus one `NOP` line per empty bundle.
3. **The shipping scheduler is `source` and packs ~1 instruction per bundle.**
   `-mllvm --pre-RA-sched=vliw-td` cuts bundles **1.19×–3.45×** on the nine of ten kernels that
   change and is worth **1.21×** on the shipped fp4 kernel at **more** instructions, bit-exact — **but
   model-level safety is NOT established (NOT RUN)**.
4. **A 256-byte register, one lane width per element size.** Cost is per register:
   `i8 = fp8 : bf16 : f32 = 1 : 2 : 4` per element.
5. **fp8 is native for the non-arithmetic half and 3 instructions for `+`.** Price fp8 adds.
6. **Expose independent work *inside* the body** (1.47× at zero cost); do **not** reach for a bigger
   `#pragma unroll`.
7. **Read the listing, count bundles, and check operands** — a count cannot tell a working op from a
   dead one (the 22 = 22 LUT case).
8. **`shuffle.i8` does not do a cross-lane gather under the tested call structure** — the 16-element
   replication is refuted; whether it is ISA or lowering is open.

---

**Artifacts this chapter restates** (all under `Archive/by-date/<date>/<task>/` unless noted):

| what | artifact |
|---|---|
| the 4-slot bundle, the listing identity, the 1-bundle/cycle model, the scheduler lever, the decisive pair | `Archive/by-date/2026-10-12/tpc-coissue/` (`REPORT.md`, `tools/vliw.py`, `tools/bundles.py`) |
| the emitted-slot rate, the NOP-inclusive count, the *copy* arm's `5.954 µs` floor, the TPC-clock query absence | `Archive/by-date/2026-10-12/tpc-issue-rate-ceiling/` (`tools/slots.py`) |
| the `MAX(t_ALU, t_mem)` pipe-overlap model (the `t_ALU` fit, `t_mem = 15.334 µs`) | `Archive/by-date/2026-10-09/fp4_arith/` (`README.md` §1.6), re-fit in `Archive/by-date/2026-10-12/arith_review/` (`FINDINGS.md` §3) |
| the nine-dtype sweep, widths, folding, ILP, pack/unpack, the operand-aware gate | `Archive/by-date/2026-10-12/vector-op-throughput/` (`tools/isa_gate.py`, `listings2/`) |
| the `shuffle.i8` refutation, the constant-index rule, the `switches` sweep, the W1 22-instruction case | `Archive/by-date/2026-10-12/lut-build/` (`REPORT.md` §5) and `Archive/by-date/2026-10-12/lut-verify/` |
| the `+6.08 µs` shuffle/`sel` price over the copy floor | `Archive/by-date/2026-10-12/vector-op-throughput/` (`REPORT.md` §Q4) |
| the build path, the zero-edit route, the 14-kernel bit-exactness, the scheduler set, the `.text`-not-`.o` rule | `Archive/by-date/2026-10-13/vliw-buildpath/` (`REPORT.md`) — archived 2026-10-13, **text only**: the `.o`/`.so`/`.pt` build outputs are excluded as regenerable (12.0 MB with them, 3.8 MB without) |
