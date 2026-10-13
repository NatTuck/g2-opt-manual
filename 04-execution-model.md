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

**And there is no fourth level.** The concurrency inside a TPC is not "more work items per core" —
the TPC is a **single-context** core, so there is no wave, no warp and **no occupancy to tune**. The
negative is stated with its evidence in §4.1.4, and it is load-bearing: it is what makes §4.1.2 a
`MAX` rather than a `SUM`.

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
> == `Σ_bundles #real instructions + #(empty bundles)`
> == bundles + `Σ_bundles max(#real − 1, 0)` — the *extra* lines from bundles carrying ≥2, not the
> count of them: a bundle with k ≥ 2 real instructions contributes k lines, so `bundles + #(r ≥ 2)`
> undercounts by `Σ_bundles max(#real − 2, 0)`.**

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
keeps issuing. **This is the only overlap a TPC has** — the core is single-context, so there is no
second work item to fill the pipe while the memory drain proceeds (§4.1.4) — which is why the
relation below is a `MAX` and not a sum. The measured relation is a **max**:

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

### 4.1.4 There is no fourth level: no wave, no warp, no occupancy

The three levels above are all there is, and the negative is load-bearing enough to state with its
evidence: **the TPC is a single-context core.** There is no wave, no warp, no lockstep width below
the register, no work-item notion distinct from the index space — there is no lane id either: the
compiler's own string table carries `v_lane_id_8/16/32` and the call is **rejected**
(`Archive/by-date/2026-10-12/vector-op-throughput/` §5.4) — and **no occupancy to tune**.

**No lockstep width below the register.** A 256-lane `i8` kernel and a 64-lane `f32` kernel with the
**same skeleton, the same inner trip count and the same number of index-space points** take the
**same time** — because one instruction consumes the whole 256-byte register either way (§4.3.1).
Same kernel family (`dep`, `U = 4`), same shapes, `n = 15`, `warm = 4`, and the three rows below are
the ones where **both arms get the same `WE`** so the comparison is like for like:

| `P` | `i8` (256 lanes) `dur med` | `f32` (64 lanes) `dur med` | ratio | tids (`i8`/`f32`) | elements/point |
|---:|---:|---:|---:|---:|---:|
| 192 | 2.400 µs | 2.480 µs | 0.968 | 24 / 24 | 1024 / 256 |
| 2048 | 8.091 µs | 8.481 µs | 0.954 | 24 / 24 | 1024 / 256 |
| **8192** | **26.400 µs** | **28.480 µs** | **0.927** | 24 / 24 | 1024 / 256 |

The `i8` arm moves **4× the elements** per instruction and is the *faster* one at the largest shape.
If the machine had a narrower lockstep width — 64 lanes, say — the 256-lane op would have to issue
four times and would take ~4× as long. It does not. **MEASURED.**

**No occupancy.** The direct experiment is four `torch.compile` graphs, each containing **two
independent TPC nodes** — the compiler's own `blocking_nodes = []` on both, and
`tpc_working_engines = [24,24,24]` on both — i.e. every configuration in which two TPC work items
could be co-resident. On **every** TPC the two kernels run **strictly back to back**:
`cross_name_overlap_pairs = 0` in all four graphs, and a **running-max-end** interval census (a long
event containing two short ones is caught; adjacent-pair comparison is not enough) over **34,702
kernel events** finds **zero** overlapping intervals on any `tid`, anywhere. **MEASURED.**

**ROIs are waves, and they serialise too.** The stock `relu_fwd_bf16` node carries
`num_of_ROIs = 3` with `tpc_working_engines = [24,24,24]`; the trace shows **6 events per TPC per
iteration** (2 nodes × 3 ROIs) and still **zero** overlap. Every level of the hierarchy — points
within a TPC, ROIs within a node, nodes within a recipe — executes **one work item at a time**.
**MEASURED.**

**The consequence, stated once, because it is the whole reason this subsection exists:**

> **Because a TPC has no second context, nothing inside a TPC can hide anything else's latency.**
> The only overlap available inside a TPC is the **asynchronous memory drain** of §4.1.2 — the load
> and store slots hand requests to the memory system and the vector pipe keeps issuing while they are
> *in flight*. That is a throughput overlap between two resources, not co-residency, and it is
> exactly why §4.1.2's relation is `t = MAX(t_ALU, t_mem)` rather than `t_ALU + t_mem`.

It also explains two results the chapter already carries, both of which were previously described
only as measurements: §4.2.7's `vliw-td` win and §4.3.5's 1.47× restructuring win are **pure schedule
edits**, and they are worth exactly what they remove from the schedule — the machine has nothing else
to issue into the NOP slots. **And it is the reason the index space is the only work-division knob
there is**: more points per TPC is not parallelism, it is queue depth (§4.2.9).

*Artifact: `Archive/by-date/2026-10-13/tpc-wave/` — `REPORT.md` §1 and §3, `tools/overlapcheck.py`,
`json/dt_cmp.json`, `json/two_node.json`.*

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

### 4.2.8 How many TPCs does a node get? — `WE` is derived, not chosen

The model in §4.2.1 is **per TPC**, so its last term is the split: how many TPCs the node runs on —
`WE`, the `tpc_working_engines` field. **It is computed exactly, and it is predictable on the host
with no device. MEASURED: 1,579 device-measured `WE` values, 0 wrong** — and, independently, the
field equals the trace's own `tid` count in **14 of 14** dumped shapes. **Observed values over ~55
swept shapes are 4, 16, 22 and 24: "24 TPCs per launch" is shape-dependent, not a machine constant.**
Read `tpc_working_engines` out of the dump ([§2.10](02-regions.md)) before believing a kernel uses
all 24.

**The input is the index-space grid, in the graph compiler's own dim order — and the GC's dim 0 is the
*torch-last*, fastest-varying dim.** For a torch tensor `(d0, d1)` and a glue whose AP is
`a = (lanes, a1)`:

```
grid = [ ceil(d1 / lanes), ceil(d0 / a1), 1, 1, 1 ]    lanes = 256 (i8) / 128 (bf16) / 64 (f32)
```

**1-D closed form** (verified for all `G = 1 … 19 999`):

```
WE = 24                      if 24 | G
WE = ceil(G / ceil(G/24))    otherwise
```

**A 2-D grid is a genuine competition** between two trials on *box size* — a **GCD** trial and a
**naive** trial, over **one** dim-preference permutation, with `numTpcEngs = 24`. That competition
**is** the non-monotonicity: the GCD trial splits a dim by `gcd(24, dim)` and wins whenever its box is
smaller. The worked pair, `i8`, `grid = [g0, 64]`: **`g0 = 3` → 24** but **`g0 = 4` → 22**.

**Two controls, and they are the evidence** — both scored over the **1,556 rows of the six custom
sweeps** (the 23 stock observations are scored separately and are not in that denominator). Take the
GC's dim order away — use the torch order — and the model is wrong on **938 of 1,556** rows. Replace
the single permutation with the *full* permutation search and it is wrong on **167 of 1,556** — and
those 167 are *exactly* the rows where the two modes disagree, with the device agreeing with
single-permutation on **167/167** of them.

**`WE` is not requestable — but the glue's declared index space is the control.** There is no API:
`synUserProgrammability` carries only `userExecOrder` and `userBundling`, so
`synNodeSetUserProgrammability` cannot set a TPC count. The lever is the **grid** — the glue's
`TensorAccessPattern` `a = (lanes, a1)`, i.e. the per-dimension step it declares, together with the
tensor shape — because moving `a` moves the grid and hence both the point count *and* `WE`. A glue
that wants 24 engines can now choose them. *(In the GC's own dim order — the order the `grid` formula
above is written in — `a[0] = lanes` is the step on dim 0, the fastest-varying dim, i.e. torch `d1`,
and **in every glue and arm measured** it is the dtype's native lane count; `a[1] = a1` is the step
on dim 1, i.e. torch `d0`. So `a1` is the **second**-dimension step and `lanes` the **first**, and
those two are exactly the two terms of that formula.)*

**The kernel body does not matter at all** (three kernels at 2.75 / 31.25 / 20.0 bundles per iteration,
same grid → identical `WE`), and neither does the graph context. **The glue's `preferredSplitDim` is not
a lever either — a measured negative** (25 device compiles over 7 shapes × `{0,1,2,3}` give an identical
`WE`, including `(256,1024)` → 22 where a dim-0-first split would have given 24; the glue prints the
value it sets, so this is not a null test), which leaves the declared index space as the only `WE`
control found.

> ⚠ **The "fixed per-TPC cost" is not fixed — and the four figures this note used to carry are NOT
> measurements.** They read **1.26 / 1.80 / 4.12 / 16.4 µs** for `dep U = 1/4/16/64` and
> **0.62 / 2.38 µs** for `copy U = 4/64`. Those are the residue
> `dur − P × (31.25 bundles/iteration) × 0.556 ns / WE` — the **complement of a bundle model**, not a
> fitted intercept — so each is a measurement only where that model happens to be right.
> **Corrected 2026-10-13.** Re-derived with no device from the archived `psweep_dep.json`
> (`Archive/by-date/2026-10-13/tpc-launch-cost-vs-codesize/tools/archived_fit.py`, re-run for this
> text), the **model-free** `dur_med` intercept of that same `dep U=4` sweep is **1.7972 µs**
> (23 rows, x = 2.91–85.33 points/TPC, slope 75.124 ns/pt/TPC, R² 0.99825). It coincides with the
> `1.80` residue **only because at `U=4` the bundle model is nearly right** — its slope 75.1 ns/pt
> against the model's 69.45, a factor of 1.08. At `U = 1/16/64` the model's own slope grows with the
> body (§4.2.10), so those three are complements, not three more measurements of one quantity.
> **Do not quote the four as a measured curve.** **MEASURED** *(the re-derivation; the four residues
> themselves are **inherited from the source report's prose**).* The per-launch term is now measured
> directly, and it is a **bundle law** — §4.2.10.

*Artifact: `Archive/by-date/2026-10-13/we-rule/` — `REPORT.md` §2, and `we_model.py`, which predicts
`WE` for any shape with no device.*

### 4.2.9 How long does a node take? — the index space is the work-division unit

§4.2.8 says how many TPCs a node gets. This is the other half of the same model: **what a node's time
is denominated in.** It is the **index-space point** — not the byte, and not the tensor.

The graph compiler builds a 5-D index space for every TPC node (`indexSpaceGeometry`) and hands each
TPC a *slice* of it. The number of points is

```
P = ∏_d ceil( size_d / a_d )        a = the TensorAccessPattern's elements per index-space unit
```

and the per-TPC time is that slice's points times the per-point bundle cost, plus a fixed term:

```
per-TPC time ≈ (P / WE) × (bundles per point) × 0.556 ns   +   a per-TPC fixed term
```

**The decisive experiment sweeps `P` at fixed per-point work.** One binary
(`wave_i8_dep_U4`, whose inner loop is `#pragma unroll 4`, so one point is exactly 4 inner iterations
= **125 bundles**), shape `(256, 4P)`, `P = 1 … 131072`, only the shape changing. `n = 20`,
`warm = 5`; linear fit over the four largest `P` (2048, 8192, 32768, 131072):

```
dur_med (µs) = 1.9799 + 0.002983 · P          R² = 1.000000
slope × 24   = 71.60 ns per point per TPC     (bundle model 125 × 0.5556 = 69.45 ns → 1.031×)
```

| `P` | `WE` | `dur med` | ns/point/TPC | per-TPC spread | one point is (`24/P`) |
|---:|---:|---:|---:|---:|---:|
| 512 | 24 | 3.5200 µs | 197.5 | 31.8 % | 4.7 % |
| 8192 | 24 | 26.4800 µs | 80.14 | 4.5 % | 0.29 % |
| 32768 | 24 | 99.681 µs | 73.68 | — | 0.07 % |
| 131072 | 24 | 393.041 µs | 72.10 | **0.33 %** | 0.02 % |

**The split is even, and the per-point cost is the bundle cost.** `ev/tid` is exactly `n` at every
`P` — one kernel event per TPC per launch, even at 5,461 points per TPC — and the spread of per-TPC
durations collapses from 31.8 % at `P = 512` to **0.33 %** at `P = 131072`, in step with the point's
shrinking share of the slice. The measured slope sits **3.1 %** above the bundle model's 69.45 ns —
the same small memory-system residual the co-issue study found (§4.1.2), and the reason the `copy`
control below does not follow this law at all. **MEASURED.**

**And the access pattern's `a` is a first-class knob, not cosmetics.** Same kernel binary, same 16 MB
tensor, only the glue's `a[1]` overridden: `a = 2` (`P = 4096`) → **14.24 µs**, `a = 32` (`P = 256`)
→ **2.72 µs** — **5.235×** from the access pattern alone, at a marginal 71–75 ns per point per TPC
against the same 69.45 ns model (`dur_med = 1.9667 + 0.002995·P`, R² = 0.999988). **MEASURED.**
*(Quote `a` with every throughput number — the earlier studies held it fixed and used "identical AP"
as a control, which is correct practice, but the value is part of the result.)*

**Two caveats, both load-bearing.**

* **This is an *issue-bound* law, and it does not describe a streaming kernel.** The `copy` arm
  (2 loads + 1 store, 2.75 bundles/iteration) at the same shapes goes `P = 1` → 0.642 µs and
  `P = 2048` → 1.360 µs, i.e. 28,438 → 23.0 ns per point per TPC against a 6.11 ns prediction: it
  sits on the memory floor instead. That is §4.1.2's `MAX` structure, not a failure of this one.
* **At small `P` you are measuring the fixed term, not the kernel.** At `P = 1` the bundle model
  predicts **0.07 µs** and the measurement is **2.0 µs** — and the fixed term is itself not fixed
  (§4.2.8's ⚠ note: **1.26–16.4 µs**, tracking the kernel's unrolled code size, cause UNKNOWN). Any
  small-shape microbenchmark on this machine is measuring that term.

*Artifact: `Archive/by-date/2026-10-13/tpc-wave/` — `REPORT.md` §2 and §5, `json/psweep_dep.json`,
`json/roi_sweep.json`, `json/confirm.json`, `json/drive.json`.*

### 4.2.10 The per-launch term is a bundle law — and it is SUBLINEAR

§4.2.9's model has a fixed term `a` that was previously only bounded. It is now measured directly:
**the per-launch intercept is a function of the kernel's emitted bundle count**, and that function is
**sublinear**.

**The measurement.** One GC kernel library of **71 TPC kernels**, one torch op per arm, each an
elementwise 2-in/1-out body on a 256-byte `i8` vector register with the convention
`grid (GC order) = [ceil(d_last/256), ceil(d_prev/U)]`. Each arm is swept over a ladder of shapes at
fixed per-point work; each point is 5 warm-up launches then a `torch.profiler` window of **n = 30**
launches, each launch its own single-node graph execution. Population: **425 measurement points over
71 kernels** (sweep 1, 345 points / 59 arms; sweep 2, 80 points / 12 arms), and every row's `WE` is
**read from a `DUMP_POST_GRAPHS` dump**, not assumed. Range covered: **61 → 3,984 bundles**,
**33 → 4,530 real slots**, `WE ∈ {16, 22, 24}`, **5.8 → 745 points/TPC**. Every arm's kernel was
numerically gated before timing. **MEASURED.**

Fitting `span_min = a + b × (points per TPC)` per arm, over the **65 of the 71 arms whose own fit
reaches R² > 0.995**:

```
a (µs) = 0.6706 + 8.8911 ns × bundles        R² = 0.9766   rms 0.498 µs
a (µs) = 0.0531 × bundles^0.736              R²(log-log) = 0.9434   ← SUBLINEAR
```

`a / bundles` **falls** from **15.7 ns at 64 bundles to 8.7 ns at 2,029** — the cost per bundle
*declines* with code size, which is why the exponent is 0.74 and not 1. **MEASURED.** *(Population
note: the regression is over **65 arm-level intercepts**, each itself fitted over its own `n = 30`
sweep. The 425 points are the campaign's size, not the regression's.)*

**The causal variable is the bundle count, not the instruction count.** Fifteen **matched pairs** —
*byte-identical source* built twice, once with `-mllvm --pre-RA-sched=vliw-td` (§4.2.5) — separate
the two, because **`WE` is identical within every pair** (verified 24/24, 22/22, 16/16), so the grid,
the point count and the work distribution are held fixed and only the code size moves:

| source (identical within the pair) | bundles default → `vliw-td` | ratio | real slots | `a` default → `vliw-td` (µs) | `a` ratio |
|---|---:|---:|---|---:|---:|
| `fs32_U1` (WE 24) | 157 → 156 | 1.01 | 66 → 66 | 2.2230 → 2.2245 | **0.999** |
| `fs256_U1` (WE 24) | 829 → 828 | 1.00 | 290 → 290 | 7.7599 → 7.7991 | **0.995** |
| `dep8_U8` (WE 16) | 270 → 145 | 1.86 | **133 → 142** | 3.1316 → 2.2742 | **1.377** |
| `dep16_U16` (WE 24) | 470 → 210 | 2.24 | **229 → 248** | 4.9427 → 2.6777 | 1.846 |
| `dep32_U32` (WE 24) | 915 → 352 | 2.60 | **437 → 491** | 8.6385 → 3.9245 | 2.201 |
| `dep128_U128` (WE 22) | 3,984 → 3,500 | 1.14 | **1,749 → 4,530** | 28.979 → 23.771 | 1.219 |

**Read the two `fs` rows against the four `dep` rows.** In all fifteen pairs the intercept moves with
the bundle count, and in none of them does it stay put while bundles move; the flagged arm carries
**more** real instruction slots than its twin in **every** pair, so the fall cannot be a fall in work.
Where the scheduler changes *nothing* — straight-line bodies it already packs — the intercept moves
by **0.1 % and 0.5 %**. The sharpest single case is `dep8_U8`: **270** bundles and 133 slots at
`a = 3.1316 µs` against `vtdep8_U8`'s **145** bundles and **142** slots at `a = 2.2742 µs` — **more**
slots, **fewer** bundles, **1.377× lower**. **MEASURED.** *(Artifact:
`Archive/by-date/2026-10-13/tpc-launch-cost-vs-codesize/` — `REPORT.md` §1.4, `logs/extra.log` §A,
`json/verdict_span.json`.)*

> **The cause of the code-size dependence is UNKNOWN.** I-fetch, I-cache, loop setup, prologue cost
> and a per-bundle decode cost are all consistent with these data, and **none is tested here**. The
> *relationship* and the *causal variable* are measured; the **mechanism is not**, and it is not
> guessed. **UNKNOWN.**

**The trap this measurement had to get past: `dur_min` is not the node's cost.** Fitting the
per-TPC duration instead of the node's span gives **negative** intercepts on the large bodies,
because the TPC split is uneven at some grids and `min` over TPCs reports the *least*-loaded TPC
rather than the node. Re-derived here from the archived sweep rows (`json/sweep_all.json`, the same
6-point ladder per arm):

| arm | `dur_min` intercept | `span_min` intercept | `span_min` R² |
|---|---:|---:|---:|
| `fs64_U1` | **−0.974 µs** | +2.796 µs | 0.99998 |
| `fs128_U1` | **−2.494 µs** | +4.411 µs | 0.99999 |
| `fs256_U1` | **−5.474 µs** | +7.760 µs | 0.99999 |

The uneven split is visible in one row directly: **`fs16_U1` at 170.67 points/TPC** has a per-TPC
`dur_min` of **3.280 µs** against a node `span_min` of **8.633 µs** — **2.63×** — because the busiest
TPC carries much more than the mean. Switching the statistic to `span_min` moves **every** intercept
positive and takes those arms' R² from ≈0.968 to ≥0.9999. **`span_min` is the right quantity anyway:
the span is the node's device occupancy, which is what a launch term costs.** **MEASURED**
*(re-derived from `json/sweep_all.json` for this text).*

**And it is a per-launch cost, not a fresh-launch cost.** Over all 71 arms, `spans[0] / min(spans)` —
the first launch after warm-up against the best of the 30 — has **median 1.061, min 1.000, max
1.348**, i.e. inside the launch-to-launch spread. There is **no distinct first-launch penalty**.
**MEASURED.** *(Caveat: every recipe in this family holds **one** node, so what is measured is the
cost of a 1-node recipe. Whether the intercept is charged per node or per recipe is **UNKNOWN** here;
it is cheap to test on the archived multi-node path and it gates nothing.)*

**What the law is for.** It converts a **compile output** into a floor. A fused kernel of **six**
mixed-op stages at `U = 2` compiles to **147** bundles and fits `a = 1.7715 µs` (R² 0.99946); at its
own geometry (5.8 points/TPC) its measured span is **1.9131 µs**, i.e. a **0.2127 ms/step** floor over
43 layer-launches — **46.2×** against the port's **9.8171 ms/step** attention span (§3.6.5). The same
source under `vliw-td` is **110** bundles at `a = 1.6972 µs`. So a fused body's floor is set by a
number you can **read off a listing** before you run anything — and it is the number to read, because
the plan's three candidate floors (0.2155 / 0.3076 / 0.8356 ms/step) are three **bundle counts**:
**≤147**, **≈388** and **≈1,769**. *(Artifact: the same report's §0 and §2.)*

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
16-bit ops — `v_i8_*` is always a 256-lane type. **The register is also the machine's SIMD width, and
there is no narrower lockstep unit below it**: a 256-lane `i8` op and a 64-lane `f32` op with the same
instruction count cost the same, because each consumes one whole register (§4.1.4).

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
> this build" (the whole key list was enumerated), **not** "we could not look". **UNKNOWN** whether a
> newer or privileged build exposes them.
>
> ⚠ **And a hand-built C-API graph cannot be device-traced at all.** `synProfilerQueryRequiredMemory`
> and `synProfilerStart` return **`synUnsupported(18)`** for `synTraceDevice`, `synTraceAll` *and*
> `synTraceHost`, with `HABANA_PLUGINS_LIB_PATH` set — so the device trace is reachable only through
> the torch plugin path, and any C-API-only question needing execution windows is currently
> unanswerable by that route. **MEASURED.** *(Recorded so the next person does not re-try it.)*

---

## 4.6 What this chapter leaves UNKNOWN

Stated rather than omitted, per the manual's convention. Each entry says what would settle it.

| # | UNKNOWN | status | what would settle it |
|---|---|---|---|
| 1 | **Whether 4 slots is architectural or a compiler maximum** | every one of 8,483 censused bundles has ≤4 real instructions and the compiler names exactly four slots — so 4 is *at least* the compiler's maximum; whether the silicon has a fifth resource (e.g. a second scalar pipe) is unchecked | a TPC disassembly, or a vendor statement |
| 2 | **Why `{vliw-td, fast, linearize}` pack and the others do not** | open — `fast` ≡ `linearize` byte-for-byte is unexplained by their descriptions | compiler instrumentation (`-print-after`, a debug build) |
| 3 | **The TPC clock in Hz** | **unobservable** on this build — no counter of any name. Every figure above is in ns; any cycle figure is **INFERRED**. A cross-check that inverts a measured duration at the assumed rate gives **0.577–0.603 ns/bundle** against §4.2.1's 0.5556 — but **inverting a duration by an assumed rate cannot validate the rate**, so this is **INDETERMINATE** and neither corroborates nor contradicts §4.2.1. **The clock stays UNKNOWN** | a newer/privileged `hl-smi`, or a firmware route |
| 4 | **Whether `vliw-td` is safe for the whole model** | **NOT RUN** — bit-exact on 14 kernel bodies over the exhaustive domain plus four real shapes, but **no model run** | a model run with output comparison |
| 5 | **`shuffle.i8`: ISA or lowering?** | **UNRESOLVED** — the bare-kernel control was **INCONCLUSIVE** | a probe with a purpose-built 1-D access pattern |
| 6 | **Whether `--pre-RA-sched=vliw-td` survives a compiler/Synapse bump** | **UNKNOWN** — it is an LLVM hidden option with no vendor statement. The failure mode is loud (an unknown `-mllvm` option fails the build), so a stale build cannot pass unnoticed | a vendor statement, or the next compiler release |
| 7 | **The exact fixed per-launch term** | it is **not a constant**: §4.2.10 measures it as a **sublinear function of the emitted bundle count** — `a = 0.6706 + 8.8911 ns × bundles` over 65 arms, R² 0.9766; `a = 0.0531 × bundles^0.736`. The **1.26–16.4 µs** range this row used to carry, "at one point per TPC", was a **bundle-model complement, not a measurement** (§4.2.8). The cross-arm fit and the per-kernel size fits still disagree on the marginal slot rate (0.580 vs 0.544 ns) at the ~6 % level. **What stays open is the *mechanism*** — item 12 | a fit that identifies the second-order term (outer-loop / first-touch), a clock that can be pinned, or a disassembly that names the per-TPC overhead |
| 8 | **Whether the bundle win generalises in wall time to the other kernel families** | **NOT MEASURED** — bundle ratios exist for 12 more kernels (up to 1.84×); only the fp4 kernel was timed | time the `lutptr` / `lut` arms |
| 9 | **The TPC ELF encoding** | no `tpc-objdump`/`llvm-objdump` in the image. `.text` bytes per bundle are 21.3–23.1 across seven kernels (~22), consistent with a fixed-width bundle word plus a variable-length immediate extension — but that is **INFERRED** from byte counts | a disassembler |
| 10 | **Whether the *silicon* can host two TPC contexts** | **INFERRED: no — but only the scheduler was tested directly.** What is MEASURED is that the runtime never co-schedules two independent TPC nodes, both wanting all 24 TPCs (34,702 kernel events, 0 overlapping intervals on any `tid`); whether the hardware could interleave two SPU contexts if a runtime asked it to is not observable from Python, and we found no way to force a second task onto a TPC | a Synapse C++ recipe with two TPC nodes whose `tpc_working_engines` overlap — the archived multi-node recipe harness (`Archive/by-date/2026-10-11/synapse_cpp/`) is the starting point. If it still serialises, "single-context" is a statement about the hardware; if it overlaps, about the GC's scheduler |
| 11 | **The split pattern — is a TPC's index-space slice contiguous or strided?** | **UNKNOWN** — not observable from the trace, which gives per-TPC durations and not addresses | a kernel that writes its slice's first index-space offset into a per-TPC output element, then reads the 24 values back |
| 12 | **Why the per-launch term scales with the emitted bundle count at all** | **UNKNOWN** — §4.2.10 measures the *relationship* (sublinear, exponent 0.736) and the *causal variable* (bundles, by 15 matched scheduler pairs with `WE` fixed), but **not** the mechanism. I-fetch, I-cache, loop setup, prologue cost and a per-bundle decode cost are all consistent with the data; none is tested | a counter or a disassembly that separates fetch from issue, or an arm that moves the bundle count at constant `.text` bytes |

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
9. **There is no wave, no warp, and no occupancy.** The TPC is single-context: a 256-lane and a
   64-lane kernel with the same instruction count cost the same, and two independent TPC nodes
   serialise on every TPC. Nothing inside a TPC hides anything else's latency — which is why the
   intra-TPC relation is a `MAX` and not a sum.
10. **Count index-space points, not bytes.** `P = ∏_d ceil(size_d / a_d)` spread over `WE` TPCs at the
    per-point bundle cost; the access pattern's `a` moved a fixed 16 MB tensor **5.235×** with an
    unchanged binary.
11. **The per-launch term is a bundle law, and it is sublinear.** `a(µs) = 0.6706 + 8.8911 ns ×
    bundles` over the 65 of 71 arms whose fit reaches R² > 0.995; `a = 0.0531 × bundles^0.736`. The
    causal variable is the **emitted bundle count**, not the instruction count — 15 matched scheduler
    pairs with `WE` held fixed inside every pair, and the flagged arm carrying **more** slots every
    time. **The mechanism is UNKNOWN; do not invent one.** Fit **`span_min`**, never `dur_min`.

---

**Artifacts this chapter restates** (all under `Archive/by-date/<date>/<task>/` unless noted):

| what | artifact |
|---|---|
| the 4-slot bundle, the listing identity, the 1-bundle/cycle model, the scheduler lever, the decisive pair | `Archive/by-date/2026-10-12/tpc-coissue/` (`REPORT.md`, `tools/vliw.py`, `tools/bundles.py`) |
| the emitted-slot rate, the NOP-inclusive count, the *copy* arm's `5.954 µs` floor, the TPC-clock query absence | `Archive/by-date/2026-10-12/tpc-issue-rate-ceiling/` (`tools/slots.py`) |
| the `MAX(t_ALU, t_mem)` pipe-overlap model (the `t_ALU` fit, `t_mem = 15.334 µs`) | `Archive/by-date/2026-10-09/fp4_arith/` (`README.md` §1.6), re-fit in `Archive/by-date/2026-10-12/arith_review/` (`FINDINGS.md` §3); coefficients restated in `Archive/by-date/2026-10-12/tpc-coissue/` (`REPORT.md` §5) |
| the nine-dtype sweep, widths, folding, ILP, pack/unpack, the operand-aware gate | `Archive/by-date/2026-10-12/vector-op-throughput/` (`tools/isa_gate.py`, `listings2/`) |
| the `shuffle.i8` refutation, the constant-index rule, the `switches` sweep, the W1 22-instruction case | `Archive/by-date/2026-10-12/lut-build/` (`REPORT.md` §5) and `Archive/by-date/2026-10-12/lut-verify/` |
| the `+6.08 µs` shuffle/`sel` price over the copy floor | `Archive/by-date/2026-10-12/vector-op-throughput/` (`REPORT.md` §Q4) |
| no wave / no warp / no occupancy, the index-space point law and the `a` lever, the `WE`-is-not-24 set, the `tpc_working_engines` / `tpc_tpc_cycles` dump census | `Archive/by-date/2026-10-13/tpc-wave/` (`REPORT.md` §1–§5, `tools/overlapcheck.py`, `tools/dumpfields.py`, `json/`) |
| the per-launch bundle law, the 15 matched scheduler pairs, the `dur_min` trap, the model-free re-derivation of the `1.26/1.80/4.12/16.4` residues | `Archive/by-date/2026-10-13/tpc-launch-cost-vs-codesize/` (`REPORT.md` §0–§3, `logs/extra.log` §A, `json/verdict_span.json`, `json/sweep_all.json`, `tools/archived_fit.py`) |
| the build path, the zero-edit route, the 14-kernel bit-exactness, the scheduler set, the `.text`-not-`.o` rule | `Archive/by-date/2026-10-13/vliw-buildpath/` (`REPORT.md`) — archived 2026-10-13, **text only**: the `.o`/`.so`/`.pt` build outputs are excluded as regenerable (12.0 MB with them, 3.8 MB without) |
