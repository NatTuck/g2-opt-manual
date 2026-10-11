# Gaudi 2 HPU Optimization Manual

**Status: DRAFT.** Three chapters are written and filled with measured material. Gaps are
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
| **ASSUMED** | taken from documentation, a vendor constant, or a model's own assumption — **not** independently verified |
| **INFERRED** | derived by reasoning from measurements; the derivation is stated |
| **UNKNOWN** | we looked and did not find out. **Stated, not omitted** — see the note below |

**Numbers are quoted with the conditions they are valid under.** Bandwidth in particular is
size-dependent on this hardware; a figure without its size is not a figure. Where a number
has a known regime boundary, the boundary is given.

**A note on UNKNOWN.** Three separate errors during the work behind this manual came from
claims inherited from a summary and never checked — a clock, an engine label, and a
throughput label, two of them wrong by a clean factor of 2 or a sign flip. Where we did not
measure something, this manual says **UNKNOWN** and says what would settle it. **Do not fill
those in from general knowledge.**

## Chapters

| # | chapter | lines | status |
|---|---|---|---|
| **0** | **[TL;DR](00-tldr.md)** | 93 | **read this first** — the five facts that change how you write code, plus the one measurement rule that changes what you optimise |
| 1 | [Gaudi/HPU Basics](01-basics.md) | 920 | hardware model · engines · queues · compilation model · memory · bandwidth · cost models · markers · profiling · pinning |
| 2 | [Optimizing small programs and regions](02-regions.md) | 1010 | overlap law · composition · recipe boundaries · region strategy · the lever · measurement discipline · dump instrument |
| 3 | [Porting a large language model to HPU](03-porting.md) | 1369 | prefill vs decode · MoE · quantization · specialization · pinning · dead ends · defect case studies |

Example code lives in [`examples/`](examples/), organized by chapter — **13 files**, of which
one (`examples/02-regions/03_inspect_graph_dumps.py --demo`) runs CPU-only and has been
verified end to end, and one (`examples/03-porting/fp8_scale_adjustment.py`) is host-only and
has been run. **The rest are annotated sketches** that need a device; each says so in its
header.

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
