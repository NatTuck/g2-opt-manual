# Gaudi 2 HPU Optimization Manual

**Status: DRAFT.** Six chapters are written and filled with measured material. Gaps are
marked explicitly rather than left blank or guessed — each chapter carries its own
**UNKNOWN register**, and chapter 3 also carries a **dead-end register**.

This manual explains how to write code that executes on Gaudi 2 HPUs and runs fast. It is
intended to be sufficient for a human **or an LLM** to get up to speed on programming
Gaudi 2 without access to the source project it was distilled from.

## How to read this manual

Every non-obvious claim is tagged with its **basis**. This is the manual's most valuable
property — a number whose provenance is unknown is worse than no number, because it gets
built on.

| tag | meaning |
|---|---|
| **MEASURED** | observed on real Gaudi 2 hardware by an instrument we can point at |
| **source read** | read directly out of a named source file or binary — vendor or our own — with the path and line given. **Not executed, so not a measurement** — and not an assumption either: the artifact is quotable |
| **ASSUMED** | taken from documentation, a vendor constant, or a model's own assumption — **not** independently verified |
| **INFERRED** | derived by reasoning from measurements; the derivation is stated |
| **UNKNOWN** | we looked and did not find out. **Stated, not omitted** — see the note below |

**Numbers are quoted with the conditions they are valid under.** Bandwidth in particular is
size-dependent on this hardware; a figure without its size is not a figure. Where a number
has a known regime boundary, the boundary is given.

**Every figure names its population and its denominator, inline.** "938 of 1,556" is not a
statement; "938 of 1,556 rows of the six custom sweeps, with the 23 stock observations scored
separately" is. The chapters already carry the pattern — [§4.2.8](04-execution-model.md)'s two
controls, [§4.2.5](04-execution-model.md)'s "22 arm-comparisons", [§2.2](02-regions.md)'s "13 arms"
(of the harness's 33 arm-runs).

**A figure carried in from another document is re-derived from that document's data — its table,
JSON, log or listing — not from its prose.** Where only the prose carries it, the figure is
**marked as inherited** (e.g. *(inherited from the source report's prose)*). A description and the
table beneath it can disagree, and the sentence is usually the one that is wrong.

**Every citation names its source** — a URL, an upstream file path, or a named artifact from our
measurement campaign, archived in the source project. **It is not a promise that every source is one
you can open:** the archived artifacts live in a project that is not published, so they are
provenance pointers rather than checkable citations. A claim whose *only* support is a collection you
cannot open is an assertion, not a citation: mark it as such, or drop it.

**A note on UNKNOWN.** Three separate errors during the work behind this manual came from
claims inherited from a summary and never checked — a clock, an engine label, and a
throughput label, two of them wrong by a clean factor of 2 or a sign flip. Where we did not
measure something, this manual says **UNKNOWN** and says what would settle it. **Do not fill
those in from general knowledge.**

## Chapters

| # | chapter | ~lines | status |
|---|---|---|---|
| **0** | **[TL;DR](00-tldr.md)** | ~90 | **read this first** — the five facts that change how you write code, plus the one measurement rule that changes what you optimise |
| 1 | [Gaudi/HPU Basics](01-basics.md) | ~1,000 | hardware model · engines · queues · compilation model · memory · bandwidth · cost models · markers · profiling · pinning |
| 2 | [Optimizing small programs and regions](02-regions.md) | ~1,000 | overlap law · composition · recipe boundaries · region strategy · the lever · measurement discipline · dump instrument |
| 3 | [Porting a large language model to HPU](03-porting.md) | ~3,700 | prefill vs decode · MoE · quantization · specialization · pinning · dead ends · defect case studies |
| 4 | [The TPC execution model](04-execution-model.md) | ~600 | the three concurrency levels · the VLIW bundle as the cost currency · vector registers and dtype rules · the VLIW scheduler · reading a listing |
| **5** | **[Profiling and the identity chain](05-profiling.md)** | ~430 | **read before taking a profile you intend to act on** — the five legs from a trace event to a source line, which are keys and which are inferences, the run-time knobs that cannot be set afterwards, how to collect from a vLLM run (where the engine, not you, owns the device), and the traps that silently break the join |
| **6** | **[The fast iteration workflow](06-fast-iteration.md)** | ~330 | **how optimisation actually gets done** — one attributable profile, then spans cut at recipe boundaries, harnessed against captured bytes in seconds instead of minutes, gated on correctness, and confirmed in situ once per span |

**`~lines` is a deliberately coarse band, not an exact count.** This column used to carry exact
figures maintained by hand, and **two of the five had gone stale** — chapter 1 read 944 against an
actual 936, and chapter 3 read 1504 against an actual 1765 when this was checked. An exact number here
is a number that is usually wrong, so it is now an order of magnitude for judging reading effort. **Do
not cite it.**

Example code lives in [`examples/`](examples/), organized by chapter — **19 files**, of which
three run CPU-only and have been verified end to end on archived traces
(`examples/02-regions/03_inspect_graph_dumps.py --demo`, `examples/01-basics/gap_histogram.py`
including its `--selftest` control, and `examples/05-profiling/attribute_chain.py`), and one
(`examples/03-porting/fp8_scale_adjustment.py`) is host-only and has been run. **The rest are
annotated sketches** that need a device; each says so in its header.

Chapter 5's three programs are the exception to "annotated sketch": they were **run on a real
Gaudi 2 on 2026-10-21** and their outputs are archived, so the worked example in §5.8 is a
measurement rather than a description.

## What this draft does not yet cover

The outline in `AGENTS.md` is broader than what is written. Known gaps, so they are not
mistaken for omissions:

- **No prefill code example.** Chapter 3 gives the prefill *structure* (co-schedule MME with
  TPC, budget DMA for MME staging) but the evidence base is decode-heavy, so there is no
  worked prefill arm.
- **No firm prefill saturation target.** The MME rate is measured at 430 TFLOP/s but carries
  a 2× unit ambiguity in its source artifact, so the target is stated as pending.
- **All workload-specific numbers are from one model** — DeepSeek V4 Flash, tagged `[DSv4F]`
  where they appear. No per-chapter figures exist for any other model.
- **No engine-scale worked overlap verdict.** Chapter 2's numbers are all from synthetic
  harness arms; the project's one engine-scale figure is recorded as UNRESOLVED.
- **The shared-layer op-support table** (what decides `hpu_cluster` vs `eager`) is measured as
  a *mechanism* but its contents are not enumerated.

Each chapter's closing register lists its own gaps with **what would settle each one**. Those
registers are the intended starting point for continuing this work.

## Where this material came from

Distilled from a **DeepSeek V4 Flash** port on a **4-AIP Gaudi 2 container** (TP4), over a
sustained measurement campaign. Most hardware facts here are general to Gaudi 2; the
workload-specific numbers are from that model and are labelled as such.

The measured evidence behind each claim is archived in the source project under
`Archive/by-date/<date>/<task>/`. This manual cites it by name so a reader can go deeper,
but **nothing here requires that archive to be readable.**
