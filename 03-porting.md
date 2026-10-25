# 3. Porting a large language model to HPU

**Status: draft.** Tags follow the manual's convention — **MEASURED** / **ASSUMED** / **INFERRED** /
**UNKNOWN**. Where we did not measure something, this chapter says **UNKNOWN** and says what would
settle it; nothing here is filled in from general knowledge.

**What this chapter is.** A porting guide: what to do, in what order, and what will bite you. It is
written for someone who has a model that runs on another accelerator and wants it to run *fast* on
Gaudi 2.

**What it is not.** It is **not general Gaudi 2 expertise.** Almost every number below was measured on
**one workload on one box**: DeepSeek V4 Flash, TP4, a single 4-AIP Gaudi 2 container. Numbers that
are workload-specific are labelled **[DSv4F]** every time. The *shapes* of the arguments are general;
the *values* are not.

**And "the port" is one specific thing.** Throughout this chapter, **"the port"** means this project's
DeepSeek V4 Flash port — `src/vllm-gaudi` with the `src/vllm` engine — driven **offline** by the harness
in `src/load_and_decode/` and served by the `vllm serve` CLI. It is not any other DeepSeek V4
implementation: a `vllm_gaudi` tree from a **different repository** is a **different port**, and its runs
are not evidence about this one.

**And the engine-run figures measure a configuration, not the defaults.** Every offline run behind them
**on the current harness** went through a harness that exports `VLLM_T_COMPILE_DYNAMIC_SHAPES=0` and
`VLLM_SKIP_WARMUP=1` — dynamic shapes **off**, warm-up **skipped** — while the product default is
`dynamic_shapes_compilation = True` with warm-up **on**. The applier is `_apply_env` in
`src/load_and_decode/engine.py`, and its two lines do **not** share a date — `VLLM_SKIP_WARMUP` has been
there since 2026-09-03, `VLLM_T_COMPILE_DYNAMIC_SHAPES` only since the **2026-09-21** harness commit — so
the qualifier is a **date**: engine figures older than **2026-09-21** are **not** covered by the
dynamic-shapes half (§3.10 item 26). **Quote these figures with that configuration**: the default path is a
latent defect, not a measured alternative.

**The one-line thesis.** **Prefill and decode bind on different resources, and the doctrine inverts
between them.** A port that is tuned for one will be mistuned for the other. Get the binding resource
right for each phase before optimising anything inside it.

**Example code** for this chapter is in [`examples/03-porting/`](examples/03-porting/). Those files
are **annotated sketches, not tested code** — they are written to show the shape of the thing and
where the hazards are.

---

## 3.0 The porting order

Do these in this order. Each step's output is the next step's precondition, and doing them out of
order is how you end up optimising a term that is not the largest one.

| # | step | why it is here | section |
|---|---|---|---|
| 1 | **Pin devices and get a reproducible single-rank baseline.** | Everything downstream is a comparison. A mis-pinned run invalidates it, and a pin set must be a **partition** — a module is acquired exclusively, so two jobs cannot share one (§3.7). | §3.7 |
| 2 | **Establish the *floor* for your model at your shapes.** Compute it from the config and the checkpoint headers, not from a trace. | It tells you which term is largest *before* you write any code. | §3.1 |
| 3 | **Classify each phase: compute-bound or weight-read-bound.** Prefill is M ≫ 1; decode at bs=1 is weight-read. | The doctrine inverts; the wrong choice optimises nothing. | §3.1 |
| 4 | **Get the quantization formats right at load time.** | Format bugs are silent, and one class of them is unrecoverable. | §3.4 |
| 5 | **Get the graph structure right: few regions, no dataflow-edge surprises, watch the specialization count.** | Graph breaks and specialization both cost, and they are hard to retrofit. | §3.5 |
| 6 | **Only then price optimisations, in `merged` (device occupancy).** | Summed kernel time is not the quantity that moves the wall. | §3.6 |
| 7 | **Keep a dead-end register** so nobody re-walks a closed path. | §3.8 — ours is 13 entries long. | §3.8 |

### 3.0.1 The method: replace SPANS, not kernels

**This is the most reusable thing in this chapter, and it is the method the order above assumes at
steps 5 and 6.** It is a *strategy for choosing what to rewrite* — stated separately from any fact
about one model, because it is the part that transfers.

> **Replace spans, not kernels.**

1. **Decompose the task into a sequence of steps.** Number them. A step is a named stage with a
   checkable output — the granularity our own gates already use (`src/gates/step*.py`), and that is
   not a coincidence: **a step you can validate in isolation is a step you can replace in isolation.**
2. **Rank sub-sequences by marginal.** Find the slow **contiguous span** — steps 3, 4, 5 of a 1..10
   task — and rank candidate spans by what replacing each would buy. **Price the marginal the way §3.6
   prices everything: in `merged` (device occupancy), not in summed step cost.** The premise of a span
   is that its cost is not the sum of its parts, so a ranking built from per-step costs will mis-rank —
   and a span's own marginal is **not** simply its summed time either, because removing a span changes
   the union non-additively. **Measuring a span's marginal properly needs an arm, not arithmetic.**
3. **Pull the span out and write entirely different code that computes the same result.** **Not a
   kernel swap — a rewrite.** If the replacement is the same algorithm with a faster inner loop, this
   method has not been applied.
4. **Validate by ulp histogram over the span's output** (`src/gates/ulpval.py`). See the precision
   rule and the span-combination rule below.
5. **If the signature does not fit, WIDEN THE SPAN.** Replace steps 2, 3, 4 and 5 rather than 3, 4 and
   5 — so that step 2 emits exactly what the new code wants to consume.

#### Step 5 is the load-bearing one: a signature belongs to the *boundary*, and the boundary is a design variable

**This is the part that took us longest to see, and the part most likely to cost you a candidate for
the wrong reason.**

**A signature — an op's input/output contract: arity, dtypes, shapes, layout (§3.3.4 is our worked
instance of one) — is not a property of the kernel being replaced. It is a property of the boundary you
chose to replace at** — and that boundary is *yours to move*. A hard-coded input count in a glue layer
reads like a property of the hardware. It is not. **It is a property of replacing exactly one node**,
which is a decision you made, not a constraint you were given.

**The shape of it.** Suppose the code you want to write needs **three** inputs and the step you are
replacing takes **two**. The naive reading is *"the op does not fit — drop it"*. The correct reading
is: that third input is produced by an earlier step, so **absorb that step into the span**. Now the
boundary takes two inputs and emits the new code's output, and the glue layer you were fighting is
*inside* the span — where you are allowed to delete it.

**Why we are writing this down.** We spent a long time treating a signature mismatch as a wall, and it
**closed a candidate for the wrong reason**. The mismatch was evidence about **where we had drawn the
line**; we read it as evidence about the hardware. **When a candidate dies of a signature mismatch,
check whether the span is the reason before you write it off.**

#### The precision rule: establish what the precision **was**

**Find out what precision the span actually had — which is not necessarily what its dtype was — then
either match it or justify a looser bound.**

- **What dtype is the tensor *stored* as?** Readable off the checkpoint and the graph.
- **What precision was it *computed* in?** A tensor stored bf16 but computed in fp32 has a different
  effective precision from one computed in bf16. **The span's precision is the second thing, and it is
  the one that decides whether a looser bound is available.**

**For these DeepSeek models a great deal was presumptively designed for native NVIDIA fp4**, so an
argument that *fp32 can drop to bf16*, or *bf16 to fp8*, is **usually available**. It still has to be
**made**: the argument is that the reference's own effective precision is already at or below the
format you propose to use, and that has to be shown for the span at hand. *(The design intent behind
the checkpoint is our reading of it — **ASSUMED**, not measured. The effective precision is
measurable, and that is what the argument must rest on.)*

#### The span-combination rule — the subtle part, and the rejection is *measured*

**The ulp histogram is admissible PER STEP. It cannot simply be expanded to a whole sequence**, and that
is not a caution — it is a **decision we took and quantified**: `notes/numeric-acceptance.md`
(2026-10-07) admits criterion N for a **single op**, and for an **op pair** *only where nothing
accumulates between them* (the pair's budget then comes from **its own measured amplification**,
declared before the run), and **explicitly rejects it for anything spanning a residual add or any
accumulation** — whole layer, layer stack, whole process. **The rejection is measured, not argued.**
**Do not re-derive it in the optimistic direction.**

**The precisions do not compose by any fast rule.** There is **no `± 2 ulp` shortcut for a two-step
span** — the pair's budget has to come from the pair's own amplification. And the reason a general rule
fails is concrete rather than formal: **an fp4 ulp and a bf16 ulp are wildly different quantities.** Our
own metric makes that structural — `src/gates/ulpval.py` carries a **separate ordered-int key per
format** (`ulp_bf16`, `ulp_fp32`, `ulp_fp8`) precisely because a ulp count is only meaningful against
the format it was measured in. Summing or bounding error bars **across a format change** is therefore
not a conservative approximation; it is a **category error**. *(**source read** — `src/gates/ulpval.py`.)*

**And the composition that does work is not a sum.** The path error is the per-step residuals
**weighted by a product of downstream Jacobians** (`E ≈ Σ_i J_{i+1:n} · δ_i`), and that weighting is what
makes a composed bound useless as an acceptance test: on our own stack the amplification is
**~1.42× per layer and ~2500× by layer 22**, so a coherently propagated per-op bound lands at
**≈ 0.15 relative — `2^-2.7`, about 2.7 bits of mantissa where bf16 offers 7**, i.e. a tolerance worth
roughly a 4-bit float. That bound is *satisfiable*. It is rejected because it is **not falsifiable**:
no width is simultaneously passable by a correct implementation **and** sensitive to a defect that
matters. **A criterion a correct implementation cannot fail, and that a wrong one cannot fail either,
is not a measurement.** *(**MEASURED**; `notes/numeric-acceptance.md` §2–§3.)*

**So the method is: run the histogram, then reason about the result in light of how many steps are in
the span.**

1. **Run the histogram** over the span's output, in the format the span actually emits — and only where
   the span is admissible at all (a single op, or an op pair with nothing accumulating between them).
2. **Read the number against the span's step count.** The more steps you absorbed, the more room a
   large number has to be legitimate — and the more you owe an account of **which step** the error came
   from.
3. **Do not accept a bound no tighter than the naive sum of the constituents' bars.** A span whose bound
   is no better than the sum of its parts **has told you nothing**: that is the null result of a span
   replacement, and it should be read as *"this span bought no precision back"*, not as *"this span is
   correct"*.

> ⚠ **What is missing here, stated rather than smoothed over: for a span that crosses an accumulation we
> have no ulp-based acceptance criterion at all — by decision, not by omission.** The instrument adopted
> for that case is a **paired-differential** fidelity measure plus a **decision-level** one, rather than a
> tighter local histogram (`notes/numeric-acceptance.md` §3.1–§4). **For an admissible span the method is
> the histogram plus the step count plus a per-step attribution; the acceptance bound is a judgement made
> against those three, not a number that falls out of them.** Treat any span bound presented without that
> reasoning as unverified.

---

## 3.1 Prefill and decode bind on different resources

### 3.1.1 Decode at bs=1 is weight-read bound — the floor

At batch size 1, every GEMM in the step has **M = 1**. A GEMV reuses each weight element exactly
once, so its arithmetic intensity is `2 / bytes_per_element` — 2.0 FLOP/B for fp8, 4.0 for fp4, 1.0
for bf16. **The operation is weight-bandwidth-bound by construction.** **INFERRED** (from M = 1
arithmetic; the measured confirmation is §3.2).

So the useful pre-code exercise is: *how many bytes must cross HBM to advance one decode step, and
how long is that at the achievable bandwidth?*

**Worked example — one `bs=1` decode step, DeepSeek V4 Flash, TP=4, L=230, bf16 KV [DSv4F].**
`bytes/step` is the whole **4-card device**; `ms/card` = `bytes / 4 / 2147 GB/s`. Computed from
`config.json` plus the 46 safetensors headers — **not** from a trace. **MEASURED [DSv4F]**

| stage | bytes/step (4 cards) | ms/card | share |
|---|---:|---:|---:|
| **attention QKV/o projections** | **4.599 GB** | 0.536 | **40.9 %** |
| **routed experts (6 of 256 / layer)** | **3.449 GB** | 0.402 | **30.7 %** |
| shared expert (1, fp8) | 1.082 GB | 0.126 | 9.6 % |
| lm_head (+ final norm / head mix) | 1.060 GB | 0.123 | 9.4 % |
| compressor | 0.526 GB | 0.061 | 4.7 % |
| indexer | 0.276 GB | 0.032 | 2.5 % |
| norms + hyper-connection mix | 0.150 GB | 0.017 | 1.3 % |
| router | 0.090 GB | 0.011 | 0.8 % |
| attention core + KV cache | 0.0069 GB | 0.001 | 0.1 % |
| collectives | 0.0021 GB | 0.000 | 0.0 % |
| **TOTAL** | **11.248 GB (99.7 % weight reads)** | **1.310** | 100 % |
| **per card** | **2.812 GB** | **1.310** | |

**⇒ 99.7 % of the floor is weight reads.** KV is **7.19 MB/step = 0.06 %**; activations
**17.57 MB (0.16 %)**; writes **5.59 MB (0.05 %)**. **MEASURED [DSv4F]**

> ### ⚠ THE HEADLINE CORRECTION — and it is the one most likely to be got wrong
>
> **Attention projections (40.9 %) exceed the routed experts (30.7 %).**
>
> **A MoE-only optimisation targets the *second*-largest term.**
>
> The mistake this corrects is seductive and we made it: the phrase *"at M = 1 the step reduces to
> reading the weights"* is true, but if you silently read "the weights" as *"the expert weights"* you
> build a floor that **cannot see** attention, the shared expert or the lm_head at all — together
> **59 %** of the floor. **MEASURED [DSv4F]**, and independently corroborated by a separate
> implementation of the same model on the same hardware, whose own accounting gives attention 4.598
> vs MoE 2.998 ms/token.

**Context dependence.** At **L = 1M** the KV term grows to **1.59 GB (12.4 %)** and the floor to
**1.492 ms/card**; the ordering of the two largest terms does not change. **MEASURED [DSv4F]**

**Two caveats on the floor, both real:**

- **It costs every byte at the TPC *stream* envelope (2,147 GB/s).** Any byte actually carried by the
  **DMA engine in the 16–64 MB band** should be costed at **~1,776 GB/s — ~17 % slower** (§3.2.2).
  **MEASURED.**
- **The per-card figure assumes a balanced partition.** With plain expert-parallel dispatch the routed
  experts land on the cards binomially and the **busiest** card sets the wall:
  `E[max of 4 Binomial(6, ¼)] = 2.603` experts ⇒ **1.605 ms/card** instead of 1.310; worst case 2.515.
  **INFERRED [DSv4F]**. *This is a general porting point: **the floor is per card, and an unbalanced
  dispatch can be worth +22 % on it before you optimise anything.***

### 3.1.2 The FLOPs term — check whether it binds before you optimise it

For each stage compute `AI = flops / bytes`. The FLOPs term binds only if the MME rate
`R < BW × AI`. **INFERRED.**

**For the DSv4F decode step as a whole, the FLOPs term binds only below ~5.2 TFLOP/s** — about 1 % of
any plausible Gaudi 2 rate — so **decode is not FLOPs-bound**. The only stage where it is even close
is the attention core, which binds below ~273 TFLOP/s and is 0.001 ms of the floor. **INFERRED
[DSv4F]**

**Carry the FLOPs column anyway, so the claim is auditable rather than asserted.** That is the whole
value of doing this arithmetic: it converts "we think it's bandwidth-bound" into a number you can be
wrong about.

### 3.1.3 Prefill is the opposite

Prefill runs at **M ≫ 1**, so GEMMs have real arithmetic intensity and the FLOPs term is live.
**MEASURED** (`0.889 ms/prefill-token` for DSv4F **MEASURED [DSv4F]**).

**The MME throughput rate.** A rate exists: **430 TFLOP/s**, from a C++ bench slope, cross-checked at
**407 TFLOP/s** from an independent dump/trace pair (6 % agreement), and at **430.0 vs 430.3**
(0.07 % agreement) from an unrelated third route. It is **MEASURED — but pending a 2× unit
ambiguity**, because the source artifact's own label reads "215 TFLOPS". **Do not use it as a firm
saturation target until that ambiguity is settled.** The clock behind it is sourced from the binary
(`Gaudi2::MmeHalReader::getClkFreqMHz() = 1650 MHz`, matching the SoC clock `hl-smi` reports), and
`262144 FLOP/cycle × 1.65 GHz = 432.5 TFLOP/s` against a measured 430.0–432.7 ⇒ **the MME cost model
is accurate to 0.5 %**. **MEASURED**; the `262144` constant itself is **INFERRED** (it is absent from
the binary).

**The structural prefill moves follow from the `MAX` rule — **Chapter 2 §2.6** — whose precondition is
the four-condition overlap law (**Chapter 2 §2.2**):**

1. **Co-schedule MME with TPC in one recipe** — they share only HBM, which the MME barely touches
   (13 %), and absorb **82 %** of each other. This is the highest-value prefill scheduling move.
2. **Budget DMA for the MME** — MME operands stage via `memcpy_dma`, so **the MME is DMA-dependent by
   construction**, and those staging nodes ride the same queues as your explicit copies. This is why
   MME×DMA absorbs only **48 %**.
3. **Keep ops clusterable** or you fragment recipes and lose the `MAX`. **MEASURED** (synthetic C-API
   recipes; the percentages do **not** transfer to a real model).

### 3.1.4 The traffic-amplification ratio — rank spans by *removable bytes*, not milliseconds

> **CORRECTED 2026-10-14 — the first version of this section (commit `6c94851`) gave the KV-slab term
> as `13.65 GB` and the port's traffic as `35.083 GB/rank/step = 12.5×`, costed at `19.76 ms → 50.6
> tok/s`. Those figures omitted a whole *population* of the same staging pass — the `ATTN_CORE`
> launches, §3.6.5 — and are superseded here; the original values are retained in this marker as the
> record. The corrected figures are `20.3172 GB`, `44.996 GB/rank/step = 16.0×`, `25.35 ms → 39.4
> tok/s`. The original was **wrong when written**, and *the way it was wrong* is this section's most
> transferable lesson — see the warrant warning below.**

§3.1.1 gives the model's **irreducible** byte floor. The porting lesson is the ratio between that floor
and what the port **actually moves**:

1. **Build the model's irreducible per-step requirement** — the weights actually read per token (every
   weight element once at M = 1), the KV read per token, the activations. That is §3.1.1.
2. **Measure what the port moves** — a byte account of the port's own recipes and materialisation
   passes, per rank per step.
3. **Divide.** `traffic amplification = port bytes ÷ model bytes`.

**The ratio is the finding.** A port that moves **16× the bytes the model requires cannot be rescued by
a faster kernel.** Making every pass infinitely fast gets you to the port's own byte count costed at the
band and no further: kernel efficiency and byte count are *different levers*, and on a real port the
second is usually the larger. Do this arithmetic *before* optimising a kernel, because a kernel win is
capped by the traffic the kernel is asked to move.

**And it changes the order you attack spans in.** A span already running at ~99 % of the measured band
has **no time headroom** — all of its headroom is in the traffic it moves. Rank a span table by
**milliseconds** and that span sits mid-list; rank it by **removable bytes** and it is the largest item
on it. This is the byte-side companion of §3.6.4's rule (*price a removal by its leave-one-out, not its
union*): **a milliseconds-ranked list hides exactly the spans whose cost is traffic rather than time.**

**The worked example — one decode step of the DSv4F port [DSv4F].**

| | GB/step | source |
|---|---:|---|
| **model floor** (weights + KV + activations, one pass, batch 1) | **2.8061 /card** — 11.2244 device | §3.1.1's floor, re-derived independently |
| **what the port moves** | **44.996 /rank** | the port's own recipes and materialisations |
| **amplification** | **16.0×** | |

The floor re-derives from the HF `config.json` and the 46 safetensors headers and **reproduces §3.1.1's
own numbers — 11.248 GB device, 2.812 GB per card — to 0.2 %, and nine of its ten shares to ≤0.1 pp.**
At the TPC stream envelope (2,147 GB/s) the floor is **1.307 ms/token**.

> ### ⚠ That cross-check is aimed at the **wrong half of the model** — and that is how the error above survived it
>
> §3.1.1 is the **floor**: the model's irreducible requirement. The KV-slab materialisation is **port
> overhead** and has **no row in §3.1.1 at all**. So the 0.2 % / ≤0.1 pp agreement validates the
> *floor* side of the arithmetic and **cannot, even in principle, see a defect on the *port* side**.
> The aggregate check that gave this section its confidence was **structurally incapable of catching
> the mistake it then made** — the thing that was wrong was not in the thing that was checked.
>
> **Two corollaries a port should carry.**
>
> - **A number that reproduces against its own generator is not verified.** The port's byte model and
>   the figures above come from the same script, so comparing them is *self-consistency*. The §3.1.1
>   cross-check was the one genuinely external test — and it was aimed at the other half of the model.
> - **When you build a warrant, say which half it covers.** "The floor agrees" is not "the port's byte
>   account agrees", and only the second would have caught this.

**The port's own bytes, costed at the measured band, are 25.4 ms/token — the current code is ~21 % *short*
of the port's 50 tok/s goal.** 44.996 GB/rank/step at the measured band gives **20.96 ms/token (47.7 tok/s) at
2,147 GB/s, 22.19 ms (45.1) at 2,028 GB/s and 25.35 ms (39.4) at 1,775 GB/s** — the last being the
slowest measured point. The port *currently* runs at a measured **56.6355 ms/step device union**
(`Archive/by-date/2026-10-13/dsv4-profile-run/span_table_P1_rank0.txt`), i.e. **2.2× above its own byte
floor**. **That is the number to take away: the port's traffic, not its kernels, is the first thing
between it and a 50 tok/s step — and measuring your own byte floor is how you find out how far you
are.** *(The first version of this section read "the code as it stands has no algorithmic change left
before it hits the target". The correction moves the code from level with a 50 tok/s step to **21 %
short of it** — the more useful reading, because it tells you the gap is real and where it is.)*

**How 44.996 is built, term by term, so it is reproducible rather than asserted:**

| term | GB/rank/step | note |
|---|---:|---|
| KV-slab staging, `ATTN_PREAMBLE` | 13.6506 | 43 launches × 158.7 MB × 2 (round trip) |
| KV-slab staging, `ATTN_CORE` | **+6.6666** | 21 launches × 158.7 MB × 2 — **the population the first version omitted** |
| kexp read by the dequant | **+3.2464** | the dequant reads the 3.2464 GB int8 `kexp` tensor it was handed (§3.4.6) |
| the rest (routed fp4, fp8 write, fp8 GEMM read, attention projections, shared / lm_head / compressor / indexer / HC / router / KV) | 21.4321 | the floor-side terms, unchanged |
| **TOTAL** | **44.996** | **16.0× the 2.8061 GB/card floor** |

The **`+6.6666` and the `+3.2464` are different terms** and both are real: the `ATTN_CORE` population
costed at §3.6.5's own per-descriptor size is `25,515 × 130,640 B = 3.333 GB`, which is **close to but
not the same as** the 3.2464 GB the dequant spends re-reading the `kexp` tensor — a coincidence of
magnitude, not an identity.

**Two population facts, one of which names *which* `43` the byte model means.** §3.6.5's own
per-descriptor arithmetic gives `158,728,192 ÷ 1,215 = 130,640 B`, so `43 × 1,215 = 52,245` and
`21 × 1,215 = 25,515` descriptors/step, summing to **`77,760 = 64 × 1,215`** — **the `64` is real**. But
the **`43` is *staging launches*, not `ATTN_PREAMBLE`'s own run count**: the span table reports
`ATTN_PREAMBLE` at **63.0 runs/step (42 handles)** and `ATTN_CORE` at **43.0 runs/step**, so a reader
comparing 64 against the span table's 63 is comparing two different things. What the trace settles is
the **descriptor population**, and on populations it is **exact** — the byte size per descriptor is not
in the trace at all.

**The byte/time pairing, because without it the total looks impossible.** `20.3172 GB ÷ 7.7751 ms` would
be **2.61 TB/s** — above every measured bandwidth in §1.6, and this manual's own rule (*a floor cannot
exceed its measurement*) says that cannot be right. The resolution is that the two populations have
**two separate unions**, and they are additive because they are disjoint in time:

| population | bytes (round trip) | its own measured union | implied rate | % of the 1,775.1 GB/s point |
|---|---:|---:|---:|---:|
| `ATTN_PREAMBLE` staging | 13.6506 GB | 7.7125 ms | **1.770 TB/s** | **99.7 %** |
| `ATTN_CORE` staging | 6.6666 GB | 3.8588 ms | **1.728 TB/s** | **97.3 %** |
| **both** (overlap **0.0000 ms**) | **20.3172 GB** | **11.5713 ms** | **1.756 TB/s** | **98.9 %** |

*(The `7.7125` and `3.8588` unions are the **DMA-staging subsets** of the two span rows; those rows'
own **whole-span** unions are `ATTN_PREAMBLE` **9.8063** and `ATTN_CORE` **8.1241** ms in the span
table `Archive/by-date/2026-10-13/dsv4-profile-run/span_table_P1_rank0.txt` — all measured on the same
`merged = 56.6355 ms/step` trace. The staging-subset derivation is
`Archive/by-date/2026-10-14/traffic-ledger/`.)* **Every rate is at or just under the band, which is
what makes the correction credible**: the pass is not inefficient, it is doing **1.49× the work the
model credited it with**.

**Two spans are 74 % of that traffic, and one of them has no time headroom at all:**

| span | traffic | measured union | implied rate | share of the port | headroom |
|---|---:|---:|---:|---:|---|
| KV-slab staging, `ATTN_PREAMBLE` + `ATTN_CORE` (strided → dense, **round trip**) | **20.3172 GB** | **11.5713 ms/step** | **1.756 TB/s** = **98.9 %** of the 1,775.1 GB/s DMA point | **45.2 %** | **traffic only** |
| fp4 dequant (fp8 write + GEMM read) | **12.986 GB** | **6.8411 ms/step** | **1.90 TB/s** = **88.4 %** of the 2,147 GB/s TPC envelope | **28.9 %** | **not established** |

- **The KV slab is the largest single removable item in the step, and a milliseconds-ranked list puts it
  second.** It is a 20.3172 GB round trip serving **~2.8 MB of useful fp8 KV** — **~7,200× the payload**,
  one row per page out of a dense `[32768, 2422]` bf16 tensor built per layer-launch (§3.6.5) — and it
  already runs at 98.9 % of the DMA engine's measured envelope, so **no kernel work can buy anything
  here; only moving less can.**
- **The fp4 dequant's headroom is *not* established, and the earlier "the kernel is done; only its
  traffic is left" is withdrawn.** It is a **TPC** kernel (`fp4_dequant_lutptr_rc_gaudi2`, engine
  `kernel`), so its envelope is the **TPC stream** figure (2,147 GB/s) and not the DMA peak — and its
  implied 1.90 TB/s is **88.4 %** of that, against the dequant's own best single-shape kernel figure of
  **1,885.7 GB/s** (§3.4.8), which the model's implied rate *exceeds*. **So this section has not shown
  that the kernel is at its bandwidth limit rather than its issue limit.** An archived decomposition of
  the **pre-LUT** kernel put **72.7 % of its time in ALU** at 0.95× its own memory bound, and the LUT
  promotion has since changed that kernel — so treat the fp4 row as **traffic worth removing, with a
  kernel cost this section does not measure.**

**Check compute too, so you know traffic is the binding resource.** The step needs **29.14 GFLOP/token**
— an arithmetic intensity of **2.60 FLOP/B** against the floor — so the FLOPs term binds only below
**~5.6 TFLOP/s**; against the **430 TFLOP/s** MME rate (§3.1.3) that is a **~77× margin**, and the 2×
unit ambiguity §3.1.3 flags does not change it (halving the rate still leaves ~38×). **A 16.0× traffic
amplification against a 77× compute margin is the shape of this port's problem: it is bytes, not FLOPs.**

> **A surprise in the *other* direction: a novel architecture is not automatically a porting
> liability.** DSv4's compressed-KV plus learned sparse indexer makes the mandatory per-token KV read
> **14.1 MB/step and independent of context** — 32× less than a dense MLA cache at 20 k positions —
> because both the sliding window (128) and the indexer's `topk` (512) are fixed. **That is measured,
> not argued: the decode step is flat in context, 78.468 → 78.154 ms from 35 to ~20,000 tokens**
> (`Archive/by-date/2026-10-14/dsv4-perf-ab/REPORT.md` §3.3). **So build the floor for the architecture
> you have** — a structure that looks expensive (a second compressor, an indexer, a `topk`) can be the
> reason the floor does not grow with context.
>
> **But the chapter now carries two different KV numbers, and they are different quantities — read
> this before quoting either.** §3.1.1's KV term is **7.19 MB/step**, and **6.88 MB of it is exactly
> `640 × 512 × 21`** — the window + top-k read counted on the **21 layers that have an indexer** — with
> the remaining 0.31 MB the indexer's own state *(**INFERRED** — the split is arithmetic; §3.1.1 does
> not carry the composition)*. The **14.1 MB** here is that same per-layer read
> (`640 × 512 B`) applied to **all 43 layers** (`640 × 512 × 43`). **The two differ by the layer base,
> not by a measurement**, and **neither is the full mandatory read**: the 22 layers without an indexer
> read the window (`128 × 512 B`) rather than a top-k of 512, which §3.1.1 does not carry and the
> 43-layer count over-applies the top-k to. A layer-aware count therefore sits **between** them. **Both
> tabulate to 0.1 % of the floor — which is exactly why the ≤0.1 pp share test that validates the floor
> cannot see a 1.96× difference between them**, and why the 32× above should be read as a floor on the
> advantage rather than a precise figure.

**What removing the traffic buys, so the target is legible.** Take the three materialisations out of the
44.996 — the KV slab (**−20.3172**), the fp4→fp8 round trip (**−12.986**) and the `kexp`
re-materialisation (**−6.4928**) — and the port lands at **5.20 GB/rank/step → 2.422 ms/token (413
tok/s) at 2,147 GB/s to 2.930 ms (341) at 1,775**, against the model's own irreducible 1.307–1.581 ms.
**And if the routed-expert over-read is fixed too** (the port gathers a **static 6 experts/rank** where
~1.5 are routed there, worth ~2.4 GB/step), it lands at **2.765 GB/rank/step — the model's own 2.8061
GB/card to within 1.5 %** — **1.288 ms → 776 tok/s.** That is what "rank by removable bytes" is worth on
this port: 16.0× down to 1.0×.

> ### The instrument note that would have caught this: **count populations, not nodes**
>
> A compiled-graph dump can hold **more than one node of the same kind**, so a census of "*the*"
> materialisation must count **populations across recipes**, not one node in one graph. The `ATTN_CORE`
> node is **bit-identical** to the `ATTN_PREAMBLE` node — same `guid` (`memcpy_dma`), same `num_of_ROIs`
> (**243**), same `dma_expected_duration` (**3719.180392235825**), same 158,728,192 B payload — and it
> lives in a **different recipe's** dump
> (`Archive/by-date/2026-10-13/staging-origin-nodes-dump/evidence/post_decode_ATTN_CORE_210node.json`,
> node `46_memcpy_124917_internal`, against `…post_decode_ATTN_PREAMBLE_111node.json`'s
> `24_memcpy_116119_internal`). A reader who found one node and stopped had **2/3 of the population**.
> The trace settles it in one pass, because the two populations carry **different event names**
> (`24_memcpy_*_internal` vs `46_memcpy_*_internal`).

**Bounds on all of this, because the section should not read as more precise than it is.**

- **44.996 GB is a *lower* bound.** It uses only the roofline's own per-event byte sources plus the two
  terms the trace shows exist and the model omitted. **If §3.4.6's six-stage MoE model is right about
  the code path**, the port total is **53.92 GB/rank/step = 19.2×** — **22.27** (the non-MoE side; the
  port's routed-expert terms total 22.72) **+ 31.65** (§3.4.6's MoE path: its 44.64 less the
  uncorroborated 12.99 GB `cat`). **⇒ the port's traffic is not determined
  to better than a factor of ~1.2 (44.996 vs 53.92) by the accounting that exists** — quote the range,
  not a point.
- **The port's 44.996 GB is a byte *model*, not a byte counter.** The profiler trace carries **no byte
  field at all**: a device event's arguments are `Device / EventName / HW event name / dataType /
  eventHandle / hidden / recipeHandle / recipeId / recipeName / streamHandle`, and the only place bytes
  appear is on the HCCL collectives (`count (elements) × dtype`). So the port's traffic is
  **populations (exact, from the trace) × per-event sizes (from a compiled-graph dump or the
  checkpoint's shapes)**. The unions above are measurements; the bytes are not.
- **Both sides are per rank against per card**, so the 16.0× is like-for-like — but it is `[DSv4F]`, one
  model on one box. The *shapes* of the argument are the general part: build the floor, measure the
  port, divide, and rank by removable bytes.

### 3.1.5 Batch size > 1 on a *sparse* model does not amortise the way a dense one does

Everything above is **bs = 1**. The obvious next move — raise the batch, read the same weights once
and serve N tokens — is right in form and much weaker in degree on a MoE, and the reason is worth
stating before someone plans around it.

**The conclusion holds; the mechanism most people reach for does not.** Both the conclusion and its
replacement were measured on a parallel port of this model, and the second refutes a mechanism this
section previously carried.

**What the cost is a function of: the compiled bucket, not the routed-expert set.** A decode step is
compiled per batch bucket, and its cost is a **step function of the bucket** — flat in the number of
real rows inside it:

| bucket | 1 | 2 | 4 |
|---|---|---|---|
| step time, ms | **10.4** | **25.9** | **32.5** |

Fitted per bucket, the residual rms is **0.11 ms** (max 0.15); fitted as an affine function of rows
it is **3.95 ms**. And **n = 3 and n = 4 are both bucket 4 and cost the same** — +33 % of the rows for
−0.9 % of the time. **Rows inside a bucket are free.** **MEASURED** (a parallel port's serving arm,
TP4×PP1, buckets 1/2/4, 96–128 generated tokens per cohort, median of two cohorts).

**The routed-expert union is NOT the pacing term — it was tested directly, and it failed.** Two
requests carrying the **same** prompt route to nearly the same six experts per layer, holding the
distinct-expert count at its n = 1 minimum; two different-domain prompts reach up to 12. At every n
the two arms are **0.1–0.4 ms apart** — noise. A single-card arm with real prepared N256 weights and
the ids passed as data (so the distinct count varies without recompiling) puts **D = 6 vs D = 36
identical within 1 %** at every row count. And the routed MoE's own row scaling (1.0 / 1.05 / 1.16 /
1.75 at 1/2/4/6 rows) explains **at most 1.16× of the step's 3.13×** growth. **MEASURED**

> **CORRECTED 2026-10-24.** This section previously stated the mechanism as **INFERRED**: that on a
> sparse model the weight bytes are "the **union of the routed experts over the batch**, and that
> union **grows with N**", so the amortisation is N ÷ union growth. The measurement above was
> designed to test exactly that, and it refutes it. The **conclusion** — batching is not ~N× on a
> sparse model — survives and is now **MEASURED** rather than inferred; the **mechanism** does not.

**⇒ The lever this changes — and it is a different lever than the one this section used to name.** If
the term is fixed per **bucket**, the thing to attack is whatever is fixed per bucket: the compiled
graph's own cost at that bucket, not the expert union. **More rows inside a bucket is free; crossing
into the next bucket is what costs.** Before choosing a batching change, read `merged` **per token**
at the batch sizes you care about (§3.6.1) *and* which bucket each maps to — a change that keeps you
inside a bucket has bought nothing however much summed work it moved, and a change that moves you
across one has bought a step.

**Two bounds on the curve.** On that port **bs > 4 does not compile** — a wiring limit of that build,
not a kernel limit — so the measured curve ends at bucket 4. And the earlier statement that this
manual carries **no** batch-scaling curve at all is now superseded: it carries one at **1 / 2 / 4**,
on a parallel port's code state, and it is not ours. *(That port separately reports **10.6 / 28.4 /
35.2 ms/step** at bs 1 / 2 / 4 on a **different arm** of the same port — the same bucket-shaped curve
from a different code state, and not directly comparable to the 10.4 / 25.9 / 32.5 above. Quote one
or the other, not a mix. See also §3.1.1's independent-corroboration note for that source's
accounting.)*

---

## 3.2 The M = 1 GEMV ceiling is HBM — and it has a cliff

### 3.2.1 The measured curve, and the cliff

`[1,4096] @ [4096,N]` bf16, span = per-launch span (a launch is bounded by its **slowest** MME view,
not the median — the median understates a memory-bound launch). **MEASURED**

| N | bytes | achieved | % of 2.45 TB/s nominal | % of a pure `.clone()` (2.07 TB/s) |
|---|---|---|---|---|
| 512 | 4.20 MB | **1.14 TB/s** | 47 % | 55 % |
| 2048 | 16.79 MB | **1.94 TB/s** | 79 % | 94 % |
| 8192 | 67.13 MB | **2.19 TB/s** | 89 % | 106 % |

**⇒ At N ≥ 2048 the M=1 GEMV is within 6 % of a pure device memcpy and at 79–89 % of hardware peak.
Below ~16 MB of traffic it collapses.** **MEASURED**

**⇒ Small-tile guidance: keep a decode weight tile at or above ~16 MB.** Below that you are paying
launch/latency, not bandwidth, and the loss is large (1.14 vs 1.94 TB/s at the two nearest points).
**MEASURED**

### 3.2.2 The cliff is a TPC/GEMV phenomenon — the DMA engine does NOT have it

This is a scope correction we had to make, and it matters because the naive reading is *"small
transfers are slow on this hardware"*, which is **false**.

**The DMA engine, measured** (unions over the DMA tids, never sums; all four points **pass** their
precondition — busy ≥ 99.8 %, ops available ≥ 18.9): **MEASURED**

| size | DMA engine | vs the TPC *stream* envelope (2,147 GB/s) |
|---|---|---|
| **16 MB** | **1,775.1 GB/s** | −17.3 % |
| 64 MB | 1,776.6 GB/s | −17.2 % |
| 256 MB | 2,021.3 GB/s | −5.9 % |
| 1024 MB | 2,028.4 GB/s | −5.5 % |

**⇒ The DMA engine passes its precondition at 16 MB and delivers 1,775 GB/s. There is NO small-size
cliff on the DMA engine.** The 1.14 TB/s collapse is a **TPC/GEMV** phenomenon. **MEASURED**

**⇒ The DMA penalty is TWO-REGIME, and it is a penalty against the TPC *stream* envelope only.**
And **against the TPC `memcpy` path the DMA engine is FASTER at every size measured** (1,776 vs 1,014
@64 MB; 2,021 vs 1,704 @256 MB; 2,028 vs 1,888 @1024 MB). **The former blanket claim — *"routing
bytes through the DMA engine instead of the TPCs costs 14–24 %"* — is WITHDRAWN.** **MEASURED**

> **Two traps inside this one number, both of which produced wrong figures before:**
>
> 1. **A mislabelled row.** A row reading *"D2D copy (**DMA engine**) | 1,014 / 1,704 / 1,888 GB/s"*
>    is **not** the DMA engine: eager `y.copy_(x)` lowers to a **TPC `memcpy_u8`** kernel (768 TPC
>    kernel events, **zero** DMA-engine events). Under `torch.compile` the same op **does** go to the
>    engine. **Whether an op reaches the DMA engine depends on whether it is inside a compiled
>    region** — check, do not assume. **MEASURED**
> 2. ~~**A 512 MiB C++ bench point reading 1,646 GB/s sits BELOW the measured 256 MB (2,021) and 1 GB
>    (2,028) points.**~~ **CLOSED — RESOLVED 2026-10-11. The point is RETRACTED and the curve is
>    monotone.** Two independent reasons, either of them sufficient: **(a)** the bench times
>    `launch → sync` **inside its repetition loop**, so its **622.1 µs** median is a *host* figure
>    carrying the per-rep launch+sync cost — the device union is **530.5 µs** and is *identical*
>    whether you sync once or per rep, and the ~90 µs difference is exactly the overhead it needs
>    (530.5 + ~90 ≈ 620 ≈ 622.1); **(b)** an **arithmetic slip**: `1,073,741,824 B / 0.6221 ms` is
>    **1,726 GB/s, not 1,646**, and 1,646 corresponds to **652.3 µs**, which is none of the bench's
>    min (608.1) / median (622.1) / mean (635.2) — **it was never a bench figure at all.** The
>    **corrected point is 2,023 GB/s** (flat 1-D **2,022.9** vs shaped 2-D **2,023.5** — **0.03 %
>    apart**, so the 2-D hypothesis is **dead**: the GC flattens a contiguous 2-D memcpy).
>    **MEASURED.** *Artifact: `Archive/by-date/2026-10-11/dma_open_items/OPEN-ITEMS-c2-and-512MiB.md`
>    §2 (the four-cell arm `d2b.py`, the overhead harness `d2b_ovh.py`, the pre-committed falsifier
>    `D2b_PRE_COMMITTED_FALSIFIER.md`).* **⇒ Do not carve an outlier out of this curve.** Full
>    treatment, including how the wrong point survived four passes: **Chapter 1**.

### 3.2.3 The general lesson

**Bandwidth on this hardware is size-dependent, and different engines have different size behaviour.
A bandwidth figure without its size is not a figure.** This is why the manual's convention is to quote
the size with every rate. **MEASURED**

---

## 3.3 The MoE path is SIX materialising stages, not four

This is a porting-relevant counting error. A routed-expert path "obviously" has four stages —
gather, dequantize, concatenate, GEMM. On this stack it has **six**, and the two that get forgotten
are the ones that multiply the op count.

### 3.3.1 The chain

Read from the source. **source read** (the DSv4F implementation)

```
compute_routed(flat, topk_ids, topk_weights)
  → g = min(n_local, T*K); gather_ids = topk(hit, g)          # routing
  → deq_gathered(packed, scale, idx):
       p = packed[idx]                                        # (1) gather packed fp4
       s = scale[idx]                                         # (2) gather scales
       dequant_v4_fp4(blocks, scales):
            kexp = ((scales.view(int32) >> 23) & 0xFF) - 127   # (3) bitshift — e8m0 exponent
                   .to(int8).repeat_interleave(16)             # (4) expand  — broadcast
            o0, o1 = FP4_CUSTOM_OP(blocks, kexp)               # (5) dequant kernel
            return torch.cat([o0, o1], dim=-1)                 # (6) cat
       .view(n, R, C*2)
  → fp8_expert_gemm(x, wq):                                    # consumer
       for g in range(G):                                      # 2-D only ⇒ a loop
            torch.ops.hpu.fp8_gemm_v2(A=xq, trans_A=False, B=wq[g],
                trans_B=True, out_dtype=bfloat16,
                A_scale_inv=scale, B_scale_inv=None, ...)
```

**Two of the six are easy to miss:** the **scale gather** (stage 2) and the **`repeat_interleave`
expand** (stage 4). The expand in particular is a *broadcast of an exponent across 16 bytes* — it
looks like a detail and it is one of the largest op-count items in the whole step.

> **TRAP — the e8m0 guard is a `Tensor.item()`, so it is a dynamo GRAPH BREAK inside a compiled region.**
> `decode_e8m0_exponent` / `decode_e8m0_scale` run `_e8m0_guard`, which for **`uint8`** input evaluates
> `int(w.max()) != 0xFF` — a host sync and a branch, which `torch.compile(fullgraph=True)` cannot trace.
> **This port is safe today only because it stores the scales as `float32`** (decoded from the e8m0 byte
> at load time), for which the guard returns before that line. **Any future change that keeps scales as
> `uint8` — or as the checkpoint's own `F8_E8M0`, whose branch tests `torch.isnan(...)` +
> `bool(bad_mask.any())`, also a `.item()` — into the dequant path introduces a graph break in the
> hottest region of the step**, and it will present as a `UserError: Could not guard on data-dependent
> expression` from dynamo, not as a correctness failure. **source read** — the guard's own comment names
> `isnan` + `bool(bad.any())` as the graph break and cites the port's own rule against it
> (`ops.py:1118-1125`). *(A standalone compiled probe did hit it: the port's pre-flight feeds fp32 scales
> for exactly this reason — `Archive/by-date/2026-10-14/fp4-layout-probe-cm/test_cm_equiv.py`. **The
> failure itself is not logged anywhere a reader can open** — treat that half as inherited, not as a
> measurement.)*

### 3.3.2 The measured cost, per stage [DSv4F]

Two frames: **F2** = whole-trace ÷ 4 steps; **F1** = the decode window. **MEASURED [DSv4F]**

| stage | ops/step (F2 / F1) | merged ms/step (F2) |
|---|---|---|
| (1) gather packed fp4 | 3,096 / 2,826 | 3.85 |
| (2) gather scales (fp32) | 3,096 / 2,826 | 1.89 |
| (3) `bitshift` (e8m0 exponent) | 9,288 / 8,478 | 0.74 |
| (4) `expand` (broadcast) | 45,365 / 41,409 | **10.78** |
| (5) **dequant kernel** | 9,288 / 8,478 | **22.63 / 20.66** |
| (6) `cat` | 4,365 / 3,974 | 0.22 |
| consumer: expert GEMM | 3,096 / 2,826 | 4.07 |
| **producer path (1)–(6)** | **74,498 / 67,990** | **30.84 / 28.15** |
| *the three "obvious" kernels only* | *21,672 / 19,782* | *27.21* |

**⇒ The path's op count is 3.4× the three named kernels.** If your port's accounting counts only the
kernels you wrote by hand, you will undercount the path by that factor.

### 3.3.3 `fp8_gemm_v2` is 2-D only ⇒ a loop over experts

`torch.ops.hpu.fp8_gemm_v2(A, trans_A, B, trans_B, D, out_dtype, A_scale_inv, B_scale_inv, bias,
accumulate) -> Tensor` takes **2-D tensors**. It does **not** batch. So a batched expert GEMM becomes
**G separate 2-D calls**. **source read** (schema read from the installed library)

**At K = 6 routed experts per token and 43 layers × 3 weights, that is 774 GEMM calls per decode
step.** **MEASURED [DSv4F]**

**⇒ Consequences for a port:**
- **A TPC kernel cannot replace this GEMM** — an MME GEMM is what you want for the arithmetic.
- **The dispatch count of the consumer scales with the expert count**, so a fused producer that emits
  one operand for all G experts still needs G GEMM calls.
- **The op requires a *materialised* operand** (see §3.3.4) — there is no producer hook.

### 3.3.4 The operand contract (this is what a fused producer must satisfy)

Per expert, per weight, `fp8_gemm_v2` needs:

1. **dtype** `float8_e4m3fn` — consumed with HPU's fp8 **code** semantics, ceiling **240** (§3.4);
2. **shape** `[N, K]`, 2-D, **row-major contiguous** — consumed with `trans_B=True`, so **the last
   dim is the contraction dim**;
3. **`B_scale_inv=None`** — i.e. the operand must be **scale-free**: the per-32-group e8m0 scale is
   already folded into the fp8 exponent. The op has **no 2-D block-scale input**;
4. **one tensor per expert** (2-D only).

> **The four clauses above are NECESSARY, NOT SUFFICIENT — the runtime's acceptance criteria are
> narrower.** A 2-D call that satisfies every clause — `A` `float8_e4m3fn` `[1, 4096]`, `B` `[2048,
> 4096]` with `trans_B=True`, `out_dtype=bfloat16`, `A_scale_inv` a **0-dim** float32,
> `B_scale_inv=None` — raised **`fp8_gemm_v2 is not yet supported on HPU`** inside a
> `torch.compile(fullgraph=True)` region, while the model's own path calls the same op successfully.
> **Which argument it objects to was NOT isolated** (candidates: the `M = 1` extent, the 0-dim
> `A_scale_inv`, the operand alignment), so **do not read the four clauses as a guarantee** — if you
> restructure a call site, budget for bisecting the arguments. **MEASURED-as-inconclusive** *Artifact:
> `Archive/by-date/2026-10-14/fp4-layout-probe-cm/test_cm_equiv.py`.*

**The K-axis order is FREE**, and this is the single most useful layout fact in the MoE path: because
the contraction is a sum over `K`, **any permutation of `K` applied to *both* operands leaves the
result unchanged.** The reference implementation exploits exactly this — it lets the dequant kernel
emit its **native** 64-byte-chunk interleave and permutes the much smaller **activation** to match,
rather than un-permuting the large weight. **MEASURED**

**⇒ A producer can emit the operand in whatever chunk order its kernel naturally writes, provided the
consumer permutes the activation identically.** Verify the identity holds for your shapes before
relying on it.

> **BUT the K-permutation freedom has a precondition that is easy to lose: the layout change must not
> move the axis the split runs along.** The reference dequant's `[o0‖o1]` interleave is emitted by an
> intra-vector unpack, so **it runs along the kernel's VECTOR axis.** With the vector on the packed-byte
> axis the interleave is entirely inside `K` (a K permutation, absorbable as above). **Move the vector
> axis to `N` and the interleave folds the K-parity into `N`: the output stops being an `[N,K]` tensor
> at all, and no activation permutation can undo it.** The fix is an **emission** change — emit the lo
> and hi decodes as *separate outputs at the same `(n,c)`* rather than as one interleaved stream. Done
> that way the operand's K order becomes the packed activation's **natural (byte, nibble)** order, so
> the activation needs no scatter at all, and the operand is row-major `[K,N]` — which the MME consumes
> with **0** layout DMA nodes. **Any layout change on such an op must re-derive the emission, not just
> the strides.** *Check: a ~30-line probe (identity LUT, a known nibble pattern, `kexp ≡ 0`) reads the
> emission straight off the output bytes — `Archive/by-date/2026-10-14/fp4-layout-probe-cm/`. Verified:
> both kernels match their analytic models at **100.0000 %** over 25,165,824 positions, and the two
> operands are **byte-identical** under the K permutation `s_cm = par(c)`,
> `c_cm = blk + ell(c) + 32·s_sh`.* **MEASURED**
>
> **And the route this describes was then built and priced end-to-end — it is MEASURED, and it is
> BANKED, NOT WIRED: `notes/fp4-cm-major-route.md`; the numbers behind it:
> `Archive/by-date/2026-10-14/fp4-layout-probe-cm/` — `REPORT-CM.md` §4 and
> `REPORT-CM-MEASURED.md`.** Store the packed experts **C-major with N
> innermost** and the four expand nodes go to **zero** (`DmaBroadcast` 0, `DmaMemcpy` 0, one
> 3,146-byte activation permutation left), with the operand's K order becoming the packed
> activation's natural (byte, nibble) order. Traffic **−16.840 GB/step**, confirmed on two independent
> instruments; **3.53–3.58 ms/step** measured at the dequant call — **not** a step time. **Read it
> before re-deriving any of this**, and note the two numbers it says must never be quoted: the expand
> frees only **27.4 of the ~76 µs/call it removes** (the rest was already overlapped with the TPC
> kernel), and the **−9.584 ms/step** figure is a superseded `dma_expected_duration` **prediction**.
> Its residual ~1.2× per-byte cost is **structural** (three candidate mechanisms tested and refuted).
>
> **A zero-copy `expand` spelling does NOT avoid the broadcast.** Writing the exponent as a stride-0
> view of size > 1 instead of `repeat_interleave` still makes the GC **materialise** the stride-0 dim
> into a `DmaBroadcast` node; only a **size-1** exponent dim compiles to **0** DMA. **MEASURED**
> *(same artifact.)*

### 3.3.5 DEAD END: "remove the `cat`"

This one is worth recording in full because it looks like an easy win and it is not.

- **The `cat` is real and on the path.** Read off the compiled IR: in **277 of 277** graphs that
  contain the fp4 dequant op, `aten::cat` consumes its output. The pattern
  `ListConstruct(fp4_dequant, fp4_dequant) → aten::cat → aten::view → aten::select →
  hpu::fp8_gemm_v2(..., select, ...)` is read directly off the IR text. **MEASURED**
- **But the GC inserts its own layout ops for GEMM staging regardless.** In a controlled A/B (one
  synthetic recipe, `cat` vs no `cat`), `concat_logical` is present in **BOTH arms, 2 nodes each** —
  the graph compiler inserts `concat_logical` + `memcpy_dma` itself to stage a GEMM's operands. **A
  recipe containing a GEMM contains DMA layout ops whether or not the user's `cat` is there.**
  **MEASURED**
- **~~A recipe containing a GEMM contains DMA layout ops whether or not the user's `cat` is there.~~**
  **CORRECTED 2026-10-14 — that generalisation is FALSE as stated.** The GC **absorbs** a GEMM
  operand's layout into the GEMM *descriptor* whenever the operand's memory order is
  descriptor-expressible, so a GEMM-containing recipe can compile to **ZERO** layout DMA nodes.
  Measured on the DSv4F dequant path: the *same* recipe, the *same* GEMM, the *same* `cat`, gives
  **4** DMA layout nodes when the `repeat_interleave` expand is inside and **0** when the exponent is
  passed in as an input — and those 4 nodes' own names say `expand/broadcast/33/…`, i.e. they are the
  **broadcast's** staging, **not the GEMM's operand staging**. The `concat_logical` node is present in
  both arms (engine `None`, in place), which is the part of the bullet above that *does* hold.
  Separately measured: an operand whose memory is K-contiguous and one whose memory is N-contiguous
  both give **0** layout DMA nodes, with `trans_B=True` and `False` alike, and a pure GEMM on the same
  logical matrix in either memory gives a **0.0** diff — so the MME genuinely consumes either
  descriptor. **⇒ Whether layout DMA appears is a property of the operand's memory order, not of the
  presence of a GEMM.** *Artifact: `Archive/by-date/2026-10-14/fp4-layout-probe/` (arms B/G/H/J) and
  `Archive/by-date/2026-10-14/fp4-layout-probe-cm/` — `A_CTL` 45 nodes / 2 `DmaTranspose` +
  1 `DmaBroadcast` + 1 `DmaMemcpy` at 1.5101481958762886 / 24.462371134020618 / 24.162371134020617 /
  24.162371134020617 µs (the model's own four expand nodes, to 16 s.f., so the census can fail) vs
  `B_CTL_PRE` 28 nodes / **0** DMA / 2 `Concatenate`.*
- **And the `cat` does not lower to what you would guess.** It lowers to **`concat_logical` with
  engine `None`** plus **2× `memcpy_dma`** — the concat itself carries no engine; the DMA nodes do.
  **MEASURED**
- **The IR cannot decide critical-path membership at all**: the sidecars carry **no unique node ids**
  (a producer reference is an *op name*), so instance-level path questions are undecidable from the
  IR. **MEASURED** (a property of the format)
- **And the one synthetic measurement of the delta is INCONCLUSIVE, not negative**: it did not assert
  its own precondition (no `pending ≥ 2`, no `device_busy ≥ 80 %`), and it **changed the engine mix**
  (DMA 6 → 4, TPC 2 → 4) while removing the cat. Its span delta (42 µs) is consistent with the cat
  costing roughly its own work, i.e. no material extra serialisation — but that is a hint, not a
  result. **MEASURED-as-inconclusive**

> **⇒ Verdict: "remove the cat" is NOT an actionable lever — but NOT for the reason given here until
> 2026-10-14.** That reason ("the GC re-inserts the equivalent layout work for the GEMM's operand
> staging, so removing the user-level `cat` cannot remove the DMA layout ops") is the generalisation
> the bullet above **corrects as FALSE**. The ground is narrower, and it is enough: the `cat` lowers
> to `concat_logical` with engine `None` plus `memcpy_dma`; on this path the layout DMA is the
> **expand's** staging, not the cat's — the no-expand control, with the cat present, compiles to
> **0** layout DMA nodes — and the one synthetic measurement of the delta is **inconclusive** (it did
> not assert its precondition and it changed the engine mix). **Do not spend a run on it.** Recorded
> here so nobody re-walks it. **MEASURED**

### 3.3.6 What the MoE path is *already* doing right (so you don't "fix" it)

- **The MoE path is already ONE multi-engine recipe per layer**, carrying **66–67 % of device time**,
  with the dequant and the GEMM **already sharing ONE submission** (172 launches carry both under one
  recipe handle). **MEASURED / NOTE-CLAIM** — corroborated by two independent instruments at 66.42 %
  and 67.04 % on different ranks. *(The two numbers are **not** identical by construction: a recipe
  handle is a **per-process pointer**, so cross-rank aggregation is invalid.)*
- **99.11 % of decode engine-labelled device work is already in multi-engine recipes**, and **no
  MME-only recipe exists anywhere in the corpus**. **MEASURED**
- **Concentration**: the top-1 recipe is **67.0 %** of engine-labelled time; top-5 77.2 %; top-20
  86.4 %. The step is **one big multi-engine recipe replicated ~43×**. **MEASURED [DSv4F]**

**⇒ "The dequant is in a separate submission / the overlap is missing" is NOT the finding.** The terms
are the **ALU** and the **materialised traffic**. Check your own port's structure before assuming
either.

**And before re-implementing any of this by hand, read §3.4.7:** the vendor ships a fused MXFP4 MoE op
(`torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights`) that collapses the six stages above into one,
with a measured conformance table for whether it is a drop-in for *your* model. (For DSv4F it is not —
the activation is the wrong shape.)

---

## 3.4 Quantization formats and their hazards

**This is the section most likely to cost you a week.** Format bugs here are silent, and one class is
**unrecoverable**. *(§3.4.1 — what the device's fp8 actually *is* — was carried as an UNKNOWN in the
first draft and is now **SETTLED** by measurement. Read it before anything else in this section: the
wrong format *name* gives the wrong poison set, inverted.)*

> **On sources.** A few claims in this section are **vendor-sourced** — the vendor's C++, the
> vendor's published documentation, and public format specifications. Those are cited inline to a
> file path or a URL a reader can fetch, so the vendor-sourced claims can be checked. **Everything
> else in this section is our own measurement**, and the text says which is which at each claim.
> **Nothing here depends on a source a reader cannot reach**: every claim is also stated in full
> inline.

### 3.4.1 What the device's fp8 actually is — **SETTLED**

> **HPU's fp8 is `e4m3fn` CODES (bias 7) truncated at 240. It is NOT `e4m3fnuz`.**
>
> **`float8_e4m3fnuz` contributes exactly one useful number to this stack: the ceiling `240`.**

**MEASURED** — and note the provenance: this was settled by a **decode + encode sweep on one HPU**
(one 11-second device probe, disclosed below), because the question was **not answerable from the
archive alone**. Earlier drafts of this chapter carried it as an **UNKNOWN** rather than pick a side;
it is now closed with measurement.

| | device fp8 | checkpoint fp8 (OCP) |
|---|---|---|
| code assignment | **`e4m3fn`, bias 7** | `e4m3fn`, bias 7 |
| max finite | **240** (`0x77`) | **448** (`0x7E`) |
| `E=15` bin | **reserved**: `M==0` → ±inf, `M!=0` → NaN | finite 256…448, plus NaN at `0x7F`/`0xFF` |
| symmetric | yes | signed zero |
| accepted dtype on HPU | `float8_e4m3fn` **only** | — |

**The three decisive facts:**

1. **`torch.float8_e4m3fnuz` is not a supported dtype on HPU.** A fnuz-declared decode is rejected by
   the runtime (`Expected habana_helpers::is_supported_type(dtype) to be true, but got false`).
   **There is no fnuz code path on the device to appeal to** — the only fp8 dtype HPU accepts is
   `e4m3fn`.
2. **The device's own ENCODER emits bias-7 codes.** `cast_to_fp8_v2` writes 1.0 → `0x38`, 0.5 →
   `0x30`, 2.0 → `0x40`, 2⁻⁶ → `0x08`, 240 → `0x77` — all `e4m3fn` codes. (Bias-8 would give
   1.0 → `0x40`, 240 → `0x7F`.) **This is what kills any "storage is fnuz, decode is fn" rescue**:
   the device does not *write* fnuz codes either. Storage and decode agree — on `e4m3fn`.
3. **The consumer agrees.** `cast_from_fp8` matches the CPU `e4m3fn` table **242/256** (the same 14
   codes) and the `e4m3fnuz` table **1/256**, byte-for-byte over all 256 codes. Nothing in the stack
   reads our stored bytes with bias-8 semantics.
4. **The rule is ENGINE-INDEPENDENT — tested, not assumed.** See below.

#### The rule is a property of the *device*, not of an engine

**This was a legitimate worry and it needed measuring.** Facts 1–3 above came from **cast ops, which
run on the TPC** — while the op that actually consumes every fp8 weight in a decode step is
**`fp8_gemm_v2`, which runs on the MME**. A vendor is free to implement different decode behaviour per
engine, so "TPC decodes it this way" does not by itself license a statement about the MME.

It has now been tested in one process, on one code set, in one run:

> **`MME == TPC byte-for-byte over all 256 codes: True.`** Same 14-code mismatch set
> (`0x78–0x7E`, `0xF8–0xFE`). **MEASURED**

- **Calibration passed first**, so the readout is a pure decode: the 242 non-disputed codes are exact
  **through the MME** (242/242), the spot values check (`0x00`→0, `0x38`→1.0, `0x3C`→1.5, `0x40`→2.0,
  `0x76`→224, `0x77`→240), and a B-doubling control holds.
- **`e4m3fnuz` is excluded twice over, on the MME**: it reads `0x76` as **224** (fnuz would be 112) and
  `0x80` as **0** (fnuz would be NaN). The only representation-level difference from the TPC is the
  *sign of zero* at `0x00`/`0x80` (TPC `−0`, MME `0`) — numerically equal.
- **⇒ The format statement needs no engine qualifier.** "`e4m3fn` codes (bias 7), 240 ceiling, `E=15`
  reserved as an inf/NaN bin" holds for **TPC and MME alike**, and **the guard's byte set
  `0x78–0x7E` / `0xF8–0xFE` is confirmed for both engines.** This is the sentence a port needs.

> **Why this had to be measured rather than read off a dump — and the trap it hides.** The fp8 decode
> happens **inside the GEMM node**: a post-graph dump of the same run shows the
> `MME GEMM ['hfloat8','hfloat8'] → ['bf16']` node consuming fp8 operands **directly, with no
> intervening cast node** — i.e. **not a TPC pre-pass**. So **no engine-attribution pass can ever see
> this decode**; engine attribution would have reported *"MME, no fp8 cast"* and been **silently
> wrong**. *(A microarchitectural pre-stage inside the MME node cannot be excluded from a numeric
> readout — but the operational answer is the same either way: the rule seen through the GEMM is
> identical to the TPC's.)* **MEASURED**

**The structural relation — this is the useful part.** Over all 256 codes,
**`e4m3fnuz(c) == e4m3fn(c) / 2` for all 253 codes finite in both.** `e4m3fnuz` is exactly
*`e4m3fn` shifted down one exponent bin* (bias 7 → 8), with the top bin extended to 240 and the NaN
moved to `0x80`. So the two names describe the **same code assignment at two different scales**, and
the only thing the fnuz name buys you here is the number **240**. *(It is also worth recording that
the old "`e4m3fnuz`: max 240, no NaN/inf" description was not even true of the dtype it named:
torch's `float8_e4m3fnuz` has **no inf** but **does** have a NaN code, `0x80` — only 255 of its 256
codes are finite.)*

#### The exponent bias is SELECTABLE — our bias-7 semantics describe the **configured default**

**This is new content, not a correction: our measurement was right, and the *scope* of the claim was
too wide.** It also dissolves an apparent conflict with Intel's `neural_compressor`, which describes
Gaudi 2's fp8 as `float8_e4m3fnuz` (**bias 8**). **Both statements are true about different things.**

**The vendor C++ says the bias is a runtime-selected parameter.** The vendor's PyTorch bridge,
`backend/h2d_scales_cache.cpp:24-88` — public source, `HabanaAI/gaudi-pytorch-bridge` (our copy of
the tree is archived at `data/Archive-2026-10-12/reference-diff/bridge/`):

```cpp
 * Default exp_bias is 7 (only torch.float8_e4m3fn is supported).
 * Possible exp_bias for gaudi2 is [3, 7, 11, 15], so -1 is added as an inversion of 15.
static constexpr int default_bias = 7;
  if (is_gaudi2) { biases = {-1, 3, 7, 11, 15}; }
```

…and the comment says these are *"picked in runtime by fp8 ops supporting H2D scales"* — i.e. the bias
is carried **in the scale tensor**, per op.

**Corroborated three ways:**

- **Vendor documentation** —
  `https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html`
  (verified present in the 1.24.2 docs set): *"In Gaudi 2 there are **4 possible exponent-bias values
  (3, 7, 11, 15), where 7 is the default exponent bias**"*, against Gaudi 3's expanded range (0, 63).
- **Intel's open-source `neural_compressor`** —
  `neural_compressor/torch/algorithms/fp8_quant/_core/fp_utils.py:73-146`:
  `EXP_BIAS_SETS[(GAUDI2, torch.float8_e4m3fn)] = [3, 7, 11, 15]`.
- **The vendor docs describe our measured *default* byte-for-byte** — the vendor's
  model-troubleshooting material: *"S.1110.111 = 240.0"*, *"Encoding for INF (S.1111.000)"*,
  *"more encoding for NaN (S.1111.{001..111})"*. That is exactly the table in §3.4.1. *(⚠ We could
  not resolve a fetchable URL for that page in the current public docs set — recorded rather than
  papered over. The table it agrees with is measured on the device, so nothing here rests on it.)*

**⇒ The statement to carry:**

> **On Gaudi 2 the fp8 exponent bias is SELECTABLE — `{3, 7, 11, 15}`, default `7`, chosen at RUNTIME
> per op via the H2D scale tensor.** Everything in this section describes the **configured default**.

**The operative consequences:**

- **Scope the ×0.5 mitigation to "bias-7 configurations."** The poison set published below is the
  poison set **for the default bias**. A port that **selects a different bias has a different
  reserved bin**, and the ×0.5 compensation is not automatically the right one for it.
- **The exponent bias is part of the operand's contract, exactly like the dtype.** A model or kernel
  that selects a different bias will not match a bias-7 reference.
- **Nothing needs re-measuring.** The measurement was right; it was the *scope* of the claim that was
  too wide. **Do not read this as casting doubt on the decode/encode tables above** — they are the
  configured default, measured directly.

#### The operational rule — the part a port needs

- **Poison on any HPU decode: `0x78–0x7E` and `0xF8–0xFE` — 14 codes.** The whole `E=15` bin is
  reserved, with `M == 0` → ±inf and `M != 0` → NaN. Measured decode: `0x78` → `+inf`, `0x79–0x7E`
  → NaN, `0xF8` → `−inf`, `0xF9–0xFE` → NaN. **MEASURED — and this set is for the DEFAULT bias (7);
  a port that selects another bias has a different reserved bin (see the bias subsection above).**
- **NOT poison:**
  - **`0x7F` / `0xFF`** — NaN in `e4m3fn` *and* NaN on the device. They **agree**; they are **not**
    mismatches.
  - **`0x80`** — the device reads it as `±0.0`, exactly as `e4m3fn` does.
  - everything else: the two agree on every `|v| ≤ 240`.

> ### ⚠ CPU torch and the HPU DISAGREE about the `E=15` bin — and the HPU is the one that matters
>
> **This is measured, not inferred** (a device probe that casts the same bytes both ways):
>
> | byte | **CPU torch `float8_e4m3fn` reads** | **HPU reads** |
> |---|---|---|
> | `0x78` | **256** | **+inf** |
> | `0x7C` | **384** | **NaN** |
> | `0x7E` | **448** | **NaN** |
> | `0x7F` | NaN | NaN |
>
> **CPU `e4m3fn` treats the whole `E=15` bin as *finite* (up to 448) and reserves only `0x7F` for
> NaN. The HPU reserves the entire bin**, with `M == 0 → ±inf` and `M != 0 → NaN`. **MEASURED**
>
> **⇒ Any fp8 reasoning done from CPU byte behaviour is reasoning about the wrong decoder.** A port
> that validates its fp8 handling on CPU — or that reasons from `torch.finfo(torch.float8_e4m3fn)` —
> will compute the **wrong poison set**, and will be **confident** about it, because the wrong set is
> internally consistent and the CPU never objects. This is the plausible wrong route, and it is the
> reason the poison set is stated here from device measurements rather than from the format
> definition.
>
> **The largest *finite* code on the HPU is `0x77` = 240** — and that is also what the platform's own
> activation encoder (`cast_to_fp8_v2`) **saturates** to. So **240 is the right saturation target,
> not the format's nominal 448.** A saturation bound taken from the CPU dtype's `max` would be 448,
> i.e. ~1.9× too high, and would let the very bytes this section warns about into the operand.

- **The condition is "HPU DECODES the byte", not "the byte is stored".** The upload is **bit-exact**,
  so the bytes are **inert until something decodes them**; the poison is created **at the decode**.
- **The checkpoint legitimately contains those bytes** — anything quantised at `amax/448` fills the
  bin. So *"no poison byte exists"* can never hold for the **input**; it can only hold for **what we
  hand the device**.

> **⚠ WARNING — a naming trap that inverts the guard.** The existing code comment describes the ×0.5
> as fitting the weights *"inside `e4m3fnuz`'s 240"*, i.e. as an `e4m3fn → e4m3fnuz` **format
> conversion**. **It is not a format conversion.** It is **range-fitting the checkpoint's top bin into
> the device's representable range.**
>
> This is not pedantry, because the wrong belief gives the **opposite** poison set: a reader who thinks
> the device is `e4m3fnuz` will treat `0x78–0x7E` as ordinary values (128…224 under bias 8) and guard
> against `0x80` (the bias-8 NaN) — **exactly inverted** for the real hazard. **Recorded so a port does
> not make it.**

> **One honest caveat, carried rather than buried.** The source did **not** inspect Habana's vendor
> documentation. **If the vendor docs call Gaudi 2's fp8 "`e4m3fnuz`", then the documentation and the
> silicon disagree about the code assignment — and a port must code against the silicon**, which is
> what the measurements above describe. Worth a one-line check in the vendor docs before quoting a
> format *name*.

**Provenance — which is which.**
- The **`e4m3fn`-declared decode table** (§3.4.1's measured decode values, and the 14-code set) is
  **archive-sourced** — it was measured earlier and archived.
- The **`e4m3fnuz`-specific facts** (facts 1 and 2, and the encoder codes) came from **one new minimal
  device probe** (~11 s, one process, no vLLM), run because the question was **not answerable from the
  archive**.
- The **engine-independence comparison** (fact 4) came from a **second new device probe** (~11 s, one
  process, no vLLM), with its own calibration gates and a post-graph dump of the same run.
- The **e8m0 scale histogram** (§3.4.5) was **CPU-only** over the checkpoint — no device.

The **three new measurements** are disclosed rather than presented as archive-sourced. The
**vendor-sourced claims** (the selectable exponent bias and the e8m0 byte-255 definition) are cited
inline to the vendor's own source, the vendor's documentation, and a public format specification —
**not** from our own measurements; the chapter says which is which at each claim.
Device hygiene for both device probes: one HPU, no vLLM, `GC_KERNEL_PATH` untouched, all four AIPs at
the 768 MiB / 0 % idle baseline before and after.

### 3.4.2 The conversion, and the wrapper that implements it

The standard conversion that makes checkpoint fp8 valid on Gaudi 2 multiplies **fp8 weights by 0.5**
and **fp32 scales by 2.0**. The real (dequantized) value is preserved while the fp8 bit pattern is
**range-fitted into the device's representable range** — i.e. the checkpoint's top bin (`0x78–0x7E` /
`0xF8–0xFE`, §3.4.1) is moved out of the device's reserved `E=15` bin. **It is NOT an
`e4m3fn → e4m3fnuz` format conversion** — see the warning in §3.4.1. **This is a compensation for the
DEFAULT exponent bias (7); a port that selects another bias needs its own analysis (§3.4.1).**
**MEASURED**

See [`examples/03-porting/fp8_scale_adjustment.py`](examples/03-porting/fp8_scale_adjustment.py) for
the annotated wrapper logic.

### 3.4.3 The order-criticality — READ THIS TWICE

**The halving is safe by LAYOUT, not by design.** It works **only** because it happens on a
**host/checkpoint tensor**, before the value ever meets the device decoder.

> **On an HPU tensor, the same expression is a NO-OP that PRESERVES THE POISON.** Measured: the
> HPU-side variant of the `× 0.5` line leaves **16 inf/NaN bytes** in the stored weights.
>
> **And the poison is UNRECOVERABLE.** Three independent reasons:
> 1. `inf * 0.5 == inf` and `nan * 0.5 == nan`;
> 2. the post-decode map is **many-to-one** with nothing downstream inverting it — `0x79–0x7E`
>    (288…448) all decode to NaN, so the re-encode cannot tell them apart;
> 3. **`0x78` is a fixed point** of the poison — `0x78 → inf → ×0.5 → inf → 0x78` — so iterating the
>    scale does not help.
>
> The re-encode is not even needed: `f_dev.float() * 0.5` left in fp32 is **already** inf/NaN, so a
> dequant built as `decode × scale` is poisoned **whatever the output dtype**. **MEASURED**

**⇒ The guard must be STRUCTURAL: assert that the `× 0.5` is applied to a *host/checkpoint* tensor.**
That is the property that actually prevents the hazard. **A code-value scan is NOT a substitute** — it
is (a) **data-dependent** (a tensor whose `amax` < 256 passes even on the broken path, so its verdict
moves with the data while the structural property does not), (b) **ambiguous** (`0x7F` is a legitimate
`e4m3fn` NaN and `0x78` a legitimate checkpoint code, so it cannot distinguish a genuine NaN from a
collapse), and (c) **detect-only and too late**. Keep it as a cheap **backstop** if you like — but the
checkpoint **legitimately contains** the poison bytes, so a scan can never be the thing that *prevents*
the hazard (§3.4.1). **MEASURED / INFERRED**

### 3.4.4 The e8m0 expert-scale raw-uint8 bug — and its fix

**A concrete instance of the class above, found the hard way.** **MEASURED**

- Some MoE checkpoints store the **e8m0 expert scales as raw `uint8` bytes** (to preserve them for a
  vendor-specific path), rather than as a real float.
- Those bytes then hit the conversion's `else` branch — the `* 2.0` meant for fp32 scales — and are
  **corrupted** (byte 115 → 230).
- **Fix: before the conversion, if `loaded_weight.dtype == torch.uint8`, decode `2 ** (byte - 127)`
  into a real scale.** Then the `× 2.0` applies correctly.
- **⚠ The naive decode just given is itself wrong at one byte, and this is the site where it lives.**
  `2 ** (byte − 127)` returns a finite value at `b = 255`, where the format says **NaN**. That is the
  defect in §3.4.5 — and this load path is one of the **two live** sites it was fixed at. **A port
  should route this decode through a helper that RAISES on 255** rather than paste the one-liner.
- **Do NOT** re-add CPU dequant/requant hacks, and **do NOT** force the MoE onto a different
  execution path. The native path works once the expert scale bytes are decoded. **MEASURED**

**The generalisation for a port: a weight-loading wrapper must be *dtype-aware*, and "scale" tensors
in a checkpoint can arrive as raw exponent bytes.** Inspect every scale tensor's dtype at load and
handle the byte case explicitly. **INFERRED**

### 3.4.5 fp4 — fold the scale, and the layout becomes free

For MXFP4 expert weights:

- **The per-32-group e8m0 scale is an exact power of two**, so it can be **folded into the fp8
  exponent** inside the dequant kernel (it adds `k` to the exponent field). The emitted fp8 operand is
  then **scale-free**, which is exactly what `fp8_gemm_v2` requires (§3.3.4). **MEASURED**
- **Folding it is not optional if you want to use `fp8_gemm_v2`** — the op has no 2-D block-scale
  input.
- **The output layout is modellable, and it is a chunk interleave**: the kernel emits `o0` = the
  **even** 64-byte chunks of the dequantized row and `o1` = the **odd** chunks; the `cat` places them
  at columns `[0, C)` and `[C, 2C)`. Equivalently, `cat([o0, o1], -1)` is a flat element interleave
  re-chunked by `[even, odd]`. **MEASURED** (derived from the device, verified at six shapes)
- **And here is the INVERSE, because a port writing an oracle needs it and it is the part that is
  easy to get wrong.** With `interleave` the flat element order:

  ```
  o0  == interleave.view(R, -1, 64)[:, 0::2, :]     # forward: o0 = the EVEN 64-byte chunks
  perm = [0, 4, 1, 5, 2, 6, 3, 7]                   # inverse (nW = 8): interleave from cat([o0, o1])
  #      [0, 2, 4, 6, 1, 3, 5, 7]  is WRONG
  ```

  **The kernel's own comment was right all along**; it was **our inverse permutation that was
  mis-derived** — and the diagnosis built on top of it (*"the oracle's model of the TPC vector-load
  layout over an `[R, C]` tensor is wrong"*) was itself the **mis-diagnosis**. With the correct
  permutation a layout-aware oracle matches **exactly** (0 of 1,024 bytes differ). **MEASURED**
  *(`Archive/by-date/2026-10-11/fp4_clamp/analysis/perm_probe.py`; also `T7` of the replacement
  `Archive/by-date/2026-10-12/fp4-op-contract/test_fp4.py`.)*

  **⇒ This is the manual's own reading lesson in miniature: a wrong inverse made a *correct*
  implementation look wrong, and the wrong conclusion is the one that got written down. When a layout
  disagrees with your oracle, check your inverse before you indict the kernel.**
- **`fp4_dequant_fwd_gaudi2` is bit-exact against its own contract oracle** on the full domain (all
  256 byte values × all 256 `k` values, 8 388 608 bytes) and at six production shapes, including the
  consumer operand. **MEASURED**
  > **⚠ The oracle is a transcription of the kernel, so this says less than it looks.** `test_fp4.py`'s
  > `fold()` contains the kernel's subnormal arithmetic **verbatim**
  > (`((1 << max(t+2,0)) & m) + …`), so it cannot disagree with the kernel on any `t <= 0` cell —
  > `T1` and `T5` therefore pass *by construction* on the whole lower end. Against the
  > **correctly-rounded** `e4m3fn(fp4_value · 2^k)` the kernel differs in **1,764 of 4,096 cells**:
  > the entire `t <= -3` subnormal **floor** and the `k >= 119` wrap. The kernel is bit-exact only for
  > `k >= -8` — see the next subsection and
  > `Archive/by-date/2026-10-12/fp4-op-contract/REPORT.md`.

#### The exponent fold was UNCLAMPED — **FIXED**, APPLIED to the tree, and reviewed

> **Status: FIXED, APPLIED to the shipping tree, and independently REVIEWED.** The fix is
> **a load-time assertion as the primary defence (zero runtime cost)** plus **a magnitude saturation
> to `0x77` as belt-and-braces**, and both are now in the tracked kernel. The review found **three
> real problems in the fix** — one of them a behaviour change the fix's own tests did not cover — and
> **all three are addressed**. The findings are kept in the status block at the end of this
> subsection, because they are the most transferable part of the story.

The kernel folds the scale by **adding `k` to the exponent field** and does **not** clamp the upper
end. Two ways that could bite: the folded exponent `t = exp + k` reaching **15** (the reserved bin,
§3.4.1) would emit a **poison byte**, and `t ≥ 16` would overflow the 4-bit exponent field into the
sign bit. **We checked whether it can actually happen, and it cannot — on this checkpoint.**

**Exhaustive scan of every e8m0 scale byte in the checkpoint** (not a sample). **MEASURED, CPU-only**

| group | tensors | bytes | byte range | implied `k` | bytes ≥ 133 (`k ≥ 6`) | byte 255 |
|---|---:|---:|---|---|---:|---:|
| **routed experts** | 33,792 | **8,858,370,048** | **[118, 127]** | **[−9, 0]** | **0** | **0** |
| non-expert scales (shared experts etc.) | 375 | 367,616 | [114, 121] | [−13, −6] | **0** | **0** |

- **`k ≥ 6` (needed for the folded exponent to reach the reserved bin): does not occur.** Not one byte
  in 8.86 GB of routed-expert scales, nor in the other 375 scale tensors.
- **`k ≥ 7` (exponent field overflows into the sign bit): does not occur.**
- **byte 255** — where a hand decode `2 ** (b − 127)` disagrees with torch's `float8_e8m0fnu`
  (inf vs NaN) — **does not occur**.

> **⇒ State it as a LATENT DEFECT, not an active bug.** And record *why it survived*: the distribution
> is extraordinarily tight — `k = −6` is 60.1 %, `k = −7` is 39.8 %, everything else 0.1 %; only **10
> distinct byte values exist in the whole 8.86 GB**. `k > 0` never occurs. An unbounded fold never had
> a chance to bite. **That is exactly the shape of defect that breaks on a different checkpoint** — one
> with `amax ≥ ~768` in any block.
>
> **Recommendation (now adopted): add the `t ≥ 15` upper clamp anyway.** It is cheap, and the failure
> mode is **silent** (a poison byte in the operand, §3.4.1). Defensive work, not a bug fix.

**And the `t ≥ 16` failure mode is worse than "a poison byte": it is a silent WRONG NUMBER.**
Measured on the consumer: `t = 16, mant = 0` emits `0x80`, which the HPU decodes as **`−0.0` — i.e.
zero**; `t = 17` emits `0x84` → **`−0.0078`**, a small *negative* value where the true magnitude is
≥ 256. That is a **silently sign-corrupted number**, not a NaN — and a silent zero is much harder to
notice than an inf or a NaN, which is exactly why this needed a gate rather than an inspection.

**The fix, and its measured cost.** Two variants were built and timed at the production tile
(`[4096, 2048]`, 40 launches = 960 device events per arm), **three times each**:

| arm | instr/iteration | `dur` median | per-event span (3 runs) | Δ span | ms/step @ 387 launches |
|---|---|---:|---|---:|---:|
| **unfixed** (the pre-fix kernel) | 68 | 61.281 µs | 62.186 / 62.173 / 62.173 µs | — | 24.06 |
| exponent clamp (`t ≤ 14`) | 70 | 63.920 µs | 64.594 / 64.603 / 64.603 µs | **+2.41 / +2.43 / +2.43 µs (+3.9 %)** | 25.01 (+0.95) |
| **magnitude saturation → `0x77`** (applied) | 72 | 67.760 µs | 68.636 / 68.543 / 68.543 µs | **+6.45 / +6.37 / +6.37 µs (+10.3 %)** | 26.53 (+2.47) |

**The three runs agree to ±0.1 µs, and the delta is visible in `dur` AND in the launch span** — so it
is not a one-run artefact. The same deltas hold at `8192×1024` and `12288×2048`: relative deltas
3.93 / 4.03 / 4.51 % and 10.27 / 10.41 / 10.90 % — a spread of **~0.6 pp**, *not* the "0.1 %" an
earlier draft claimed (only the absolute span between the two *small* shapes holds to 0.1 %).

> **The cost is NOT predicted by the instruction-count slope, and that is the engineering lesson.**
> The project's measured slope of ~0.81 µs per instruction/iteration predicts **+1.6 µs** for the
> clamp and **+3.2 µs** for the saturation; the measurements are **+2.4** and **+6.4 µs**. The added
> instructions sit on the decode's **dependency chain** (`t → over → mag → or → store`), not in
> parallel issue slots, so each costs ~1.2–1.6 µs rather than 0.81. **A safety fix for a *latent*
> defect that sums to 2.5 ms/step of device time deserves an explicit decision, not a reflex "add the
> clamp".** **MEASURED**
>
> **Honest caveat on that explanation** (from the review): reproducibility *supports* the
> dependency-chain story but does not test it — the marginal cost is superlinear (2 instructions →
> +2.4 µs, 4 → +6.4 µs), which the chain story explains but which register pressure or a VLIW
> bundling change would explain equally. **A fourth control arm with 2–4 *independent* dummy
> instructions would separate "on the chain" from "any instruction".**

**How much of that reaches the wall — and the honest bound on it.** An added `D ≈ 6.4 µs` per launch
is only exposed if the device has a gap to absorb it. Measured from the existing decode trace
(1,548 launches; `Archive/by-date/2026-10-11/fp4_clamp/analysis/exposure.py`; **no engine run**): the fp4 launches are a **packed critical
chain** — **union/sum = 0.965** (90.501 of 93.735 ms), the **median gap to the next launch is
−2.14 µs**, and **1,376 of 1,547** gaps are **≤ 0**. So **88.9 %** of `D` is exposed on the *device*
timeline: **2.20 ms/step** (against a summed-op bound of 2.47 ms/step).

> **The WALL cost is an ESTIMATE, not a measurement** — pinning it down needs an engine run. If the
> dequant's device busy is 22.6 ms/step and its contribution to `merged` is 12.086 ms/step, then ~53 %
> of it is inside the critical region, giving **~1.2 ms/step ≈ 1.4 % of an 88 ms step.** Treat the
> **2.20 ms/step device bound as measured** and the **~1.2 ms/step wall figure as an estimate**.
> **The decision that follows is explicit, not a reflex:** the clamp costs ~2.2 ms/step of device
> time to guard a condition that **cannot fire on this checkpoint** (64× margin — see the correction
> below), while the load-time
> assertion alone costs **0 ms/step**. Both are in place; reverting the clamp is one patch reversal
> plus a rebuild. **MEASURED / ESTIMATED**

**Why the assertion is primary.** The overflow needs `k ≥ 6`, i.e. a scale byte ≥ 133. A **load-time
assertion** makes the defect **unreachable at zero runtime cost** — so the saturation is a
**deliberate correctness-over-speed choice**, not the thing that buys the safety. **The assertion is
written on the scale VALUE, which is the form the loader actually has in hand:**
`max(scale) ≤ 2⁵ = 32.0` ⟺ exponent byte `≤ 132` ⟺ `k ≤ 5` ⟺ `t ≤ 14`. On this checkpoint the global
max over the scales the fp4 kernel is actually **fed** is **0.5 = 2⁻¹** (byte 126,
`layers.35.ffn.experts.12.w2.scale`), i.e. a **64× margin** — and the guard was verified to **fail**
when it should (`2⁵` passes, `2⁶` raises). Keep both if you want safety by construction *and* safety in
fact.

> **Correction (2026-10-12) — this margin was wrong by 4×, and it contradicted this section's own
> table.** An earlier revision said *"max scale = 0.125 = 2⁻³ ⇒ 256× margin"*, measured over **3,840**
> scale tensors — that is **5 of the 44 MoE layers** — and stated as a whole-checkpoint property. An
> exhaustive scan of all **33,792** kernel-fed tensors (`layers.*.ffn.experts.*.w{1,2,3}.scale`,
> 8,857,043,456 elements) gives **max byte 126 → 0.5 = 2⁻¹ ⇒ 64×**. The table two paragraphs above
> already said the byte range is `[118, 127]`, i.e. it *contains* a byte whose scale is `2⁰ = 1.0`
> (⇒ 32×) — that one lives under the **uninstantiated** `mtp.` head, which is why the two figures
> differ. **The assertion itself is correct and passes; only the stated margin was wrong. MEASURED**
> *(`Archive/by-date/2026-10-12/fp4-op-contract/` — `scale_hist.py`, `scale_extremes.py`, REPORT §5.1.)*
> Note it is also a **per-layer** quantity at runtime (`_v4_load_routed_experts` runs once per MoE
> layer), so a single checkpoint-wide number is the wrong *shape* for it as well.

> **Status block — what "FIXED" does and does not mean here.**
> - **APPLIED to the shipping tree, deliberately.** Both protected artifacts are **superseded by
>   design and recorded**: kernel source `9a16d2f7… → b60ff0b5…`, `.so` `0fffb589… → e8622c21…`. The
>   pre-fix source is preserved **byte-identically** in the archive *and* as the `base` arm, so **RED
>   stays reproducible after the apply**; the applied diff is `applied.patch`. **⇒ Do not check a
>   future tree against the old md5s** — they describe a state that no longer exists, and an md5
>   guard-rail that is not updated becomes a false negative.
> - **The shipped binary is the verified binary.** A cross-library byte comparison of the **shipped
>   `.so`** against the verified saturation arm is **IDENTICAL at all six shapes** (`2×256`,
>   `12288×2048`, `24576×1024`, `4096×4096`, `4096×2048`, `8192×1024`; 0 differing bytes of
>   25,165,824 per output at the large shapes). Post-apply: `shipped` **PASS**, saturation arm
>   **PASS**, unfixed arm **FAIL** (544 failing `(byte, k)` pairs, first `(6, 6)`: expected `0x77`,
>   got `0x78`).
> - **The load-time assertion is `max(scale) <= 32.0`** (threshold derivation above), and its margin
>   is measured: on the real checkpoint the global max is **0.5 = 2⁻¹** ⇒ **64× margin** (corrected
>   2026-10-12; an earlier revision said 0.125 ⇒ 256×, from a 5-of-44-layer subset). It **can
>   fail** (verified: `2⁵` passes, `2⁶` raises), costs **zero runtime**, and converts a silent
>   corruption into a loud load-time failure. **Limitation, accepted:** it lives at **one caller**, so
>   a future quantiser or a different checkpoint path bypasses it — which is exactly why the kernel
>   clamp is there as belt-and-braces rather than instead of it.
> - **Independently reviewed — the review found three real problems in the fix, all since addressed.**
>   1. **A behaviour change the fix's own tests did not cover.** The saturation as first written
>      (`mag = over ? 0x77 : mag`, applied *after* the magnitude is zeroed for `idx == 0`) turned an
>      **exact zero into ±240** for a zero nibble at `k ≥ 15` — a value the *unfixed* kernel got
>      right, and **no test covered it** (the identity test stops at `k = 5`; the zero-nibble tests
>      use `k ≤ 8`). Fixed by gating the flag on the nonzero predicate
>      (`over = sel_geq(t_raw, 15, nz, 0)`). **⇒ A fix that replaces a whole expression can change an
>      input class nobody tested — and "it changes nothing that was already correct" is a claim that
>      needs a test per input class, not one boundary probe.**
>   2. **The guard is INERT at the top of the range.** `t_raw = exp + k` is an **8-bit** add, so for
>      `k ≥ 119` (scale byte ≥ 246) it wraps negative, the flag never fires, and **all** arms emit the
>      subnormal floor. Not a regression — but it **falsifies any absolute "`E=15` is now
>      unreachable"**: that holds for `t_raw ∈ [15, 127]` only, and `k` legitimately reaches 127 for
>      the largest e8m0 scale (byte 254). **The load-time assertion closes this hole; the kernel clamp
>      alone does not.** *(This is why the two defences are not redundant.)*
>   3. **The headline defect was not gated.** The test credited with catching the sign-clobber defect
>      **passed on the unfixed kernel**, because its oracle is **arm-dependent** — `want` encodes the
>      arm's own rule, so `got == want` is a self-consistency check, not a test. Only the reserved-bin
>      defect was genuinely RED. Fixed by asserting the **contract** directly (`sign == 0 ⟹ bit 7
>      clear`; `t ≥ 15 ⟹ byte ∈ {0x77, 0xF7}`).
>   **Also found and fixed:** the replacement `test_fp4.py` had **not been installed** in the shipping
>   directory — the old, broken test was still in place, so the shipped kernel was
>   **fixed-but-unvalidated** until that was corrected.
> - **The correctness check that matters:** on a 12,288 × 2,048 tensor with `k ∈ [−9, 0]` (the
>   checkpoint's range) and with `k ∈ [−8, 5]` (the clamp's own boundary), **all three arms are
>   byte-identical** — 0 differing bytes in 25,165,824 per output. **The clamp fires only at
>   `t ≥ 15`; below that the emitted byte is bit-identical to the unfixed kernel.**

**The lower bound: `k < −8` DOES occur — and it is benign, not a poison.** The kernel's own comment
only reasons about the lower bound ("exact for `k ≥ −8`"). It is exceeded: **1,070 routed-expert
bytes**, and **365,620 shared-expert bytes** down to `k = −13`. What happens there is **not
corruption**: the subnormal branch's `sh2 = max(t + 2, 0)` makes the emitted code **saturate to the
e4m3fn subnormal floor `2⁻⁹`** (byte `0x01`), the smallest value the format can represent — an
**overestimate** of weights whose true scaled value is below the floor, not a poison code.
Magnitude-tiny. **Confirmed on device at `k = −9, −10, −13, −20`** — monotone, the emitted byte is
`0x01`/`0x02`, **never `0` and never an `E=15` code**. **Deliberately NOT changed:** it is a
**modelling choice, not a defect**, and changing it would alter currently-correct outputs, so the
error is left in the safe direction and bounded. Recorded so nobody re-derives it. **MEASURED**

> ### ⚠ The lesson worth keeping: **a green test is not evidence about the thing you think it tests**
>
> Both defects in this subsection — **the fold above and the e8m0 decode below** — were **live in
> shipped code behind a passing test.** The story has **two halves**, and both belong in a manual,
> because together they are the most concrete instance we have of the rule that **an instrument which
> cannot fail is not evidence**.
>
> **First half — the old test could not validate anything.** The shipping `test_fp4.py`:
> - called `fp4_dequant(b)` with **one** argument against the **two**-argument schema;
> - its reference implementation **ignored the scale term (`kexp`) entirely** — so it never tested
>   scale folding at all;
> - used the **wrong exponent bias** (8 where the kernel encodes 7), i.e. off by one exponent field
>   even at `k = 0`; and
> - compared the outputs against the low/high nibbles **in input order**, when the kernel emits the
>   `[o0 ‖ o1]` chunk interleave — so every byte would mismatch.
>
> **It passed while the kernel emitted poison.** "Stale" understates it: it could not have caught
> anything.
>
> **Second half — the replacement gate was itself decoration on its first attempt.** Its byte-255
> check used `torch.isfinite`, and **`inf` is not finite** — so it **passed on the defect it existed
> to catch.** It was caught only by running the gate in **break mode** against the unfixed code, which
> is the entire reason to build a break mode.
>
> **The replacement, measured — and this is the shape a real gate has:**
>
> | arm | result |
> |---|---|
> | **unfixed** kernel (the `base` arm — byte-identical to the pre-fix source) | **FAIL, exit 1** — 5 failing checks, incl. **1,396,623 poisoned bytes** on a tensor whose `k` drives `t` to 17 |
> | fixed arm A (exponent clamp) | **PASS, exit 0** |
> | fixed arm B (magnitude saturation) | **PASS, exit 0** |
>
> **All three arms share one library**, so **RED and GREEN come from the same binary** and the only
> difference is which kernel the op resolves to. That is what makes the RED result evidence rather
> than a claim.
>
> **⇒ What to carry into a port.** For any gate you write on a format, a layout or a decode:
> **(a)** run it against the *unfixed* code and require it to fail — if it cannot fail, it is not a
> gate; **(b)** compare against a reference that consumes **all** the inputs (the scale term here was
> silently dropped for months); **(c)** check the *layout* assumption explicitly, because a wrong
> layout assumption makes every byte mismatch and hides a real pass; and **(d)** prefer a
> **structural** check (the pattern must not exist outside the helper) over a value check, because a
> value check moves with the data. **MEASURED**
>
> **And a third instance, from the review of the fix itself.** The gate's *headline* check — the one
> credited with catching the sign clobber — **passed on the unfixed kernel**, because its oracle was
> **arm-dependent**: `want` was produced by the code under test, so `got == want` was a
> self-consistency check. **A gate whose expected value is produced by the thing being tested is not a
> gate** — the same failure as the two halves above, one level up, and it survived until the check was
> made to assert the **contract** directly (`sign == 0 ⟹ bit 7 clear`).

#### The OTHER end of the fold — the subnormal FLOOR, and it is not guarded

> **Status: REAL GAP, NOT FIXED in the shipped kernel, and LIVE on this checkpoint.** Found
> 2026-10-12 while chasing a side finding in the vendor-conformance probe. Instruments + report:
> `Archive/by-date/2026-10-12/fp4-op-contract/`. **No code was changed** — this section is the
> specification of the fix, not the fix.
>
> **Update 2026-10-13 — the kernel half of the fix now exists in a variant, and it is measured.** A
> promoted fp4 LUT library carries a variant whose 240-cell table changes exactly the **2** `k = −9`
> tie cells this section names (codes `0x1`/`0x9`), i.e. it emits ±0 where the shipped decode emits
> ±2⁻⁹ — the *intended* floor fix, and the variant this checkpoint's scale bounds select. Its
> divergence from the shipped op is therefore **bounded and confined**: **954 of 65,536** block
> outputs, `max|d| = 2.441e−04` against `max|A| = 1.572e−01`, and **1084 of 218,103,808** weight bytes
> (= the 1084 engaging tie weights). **The load-time guard this section also specifies is a separate
> change and is not what that is** — do not read the two as one. **At whole-model TP=4 the same
> correction is bit-exact against the shipped path** — 0 of 3072 logprob cells, `max|d| = 0` — because
> the `2⁻⁹ → 0` change is **below the model's bf16 output resolution**, even though the tie cells are
> demonstrably reached; the 954-of-65,536 block divergence above is the same correction seen through
> **synthetic activations and forced routing**. Full statement, including why the two relative-error
> columns must be quoted together: **§3.4.8**.

The upper clamp above guards `t = exp + k` **above**. The same expression is unguarded **below**, and
the subnormal branch is where it shows:

```c
VECTOR sh2 = v_i8_max_b(v_i8_add_b(t, BCAST(2)), BCAST(0));      // clamp EACH term to >= 1
VECTOR sh1 = v_i8_max_b(v_i8_add_b(t, BCAST(1)), BCAST(0));
VECTOR sub = v_i8_add_b(v_i8_shl_b(BCAST(1), sh2),
                        mant ? v_i8_shl_b(BCAST(1), sh1) : 0);
```

**The bug is that the two terms are clamped separately where their SUM should be rounded.** For
`t <= -3` the true value is less than one unit of `2⁻⁹`, so the correctly-rounded answer is `0` — but
`2^max(t+2,0)` cannot go below 1, and the `mant` term cannot either, so the sum floors at 1 (or 2).

**What it is, and what it is not — measured on all 4,096 `(code, k)` cells, one op call.** The kernel's
byte is compared against the correctly-rounded (RNE, 240-ceiling) `e4m3fn(fp4_value · 2^k)`:

| region | condition | cells | differ | worst \|err\| |
|---|---|---:|---:|---:|
| `zero` / `normal` (`t∈[1,14]`) / `sub_ok` (`t∈[−2,0]`) / `sat` | — | 512 / 196 / 42 / 1582 | **0 / 0 / 0 / 0** | 0 |
| **floor** | **`t <= -3`** | **1656** | **1656** | **`2⁻⁸` = 0.00390625** |
| **wrap** | `t` unwrapped `> 127` | **108** | **108** | 1.02e39 |

* The fold is **bit-exact everywhere except those two regions** — so the lower end has **no third
  failure mode**: **0 poison bytes and 0 sign flips in all 4,096 cells** (the int8 add cannot wrap at
  the bottom, `exp + k >= 6 - 128`). It is a **bounded magnitude overshoot**: magnitudes are only ever
  `±2⁻⁹` (mantissa-0 codes 1,2,4,6,9,10,12,14) or `±2⁻⁸` (mantissa-1 codes 3,5,7,11,13,15).
* **Absolute error bounded, relative error unbounded.** `|kernel - true| <= 2⁻⁸` always; the ratio
  reaches 1.3e36 at `k = -128`. **8 of the 1656 cells are exact ties** (true value exactly `2⁻¹⁰`,
  half a unit of `2⁻⁹`) and 1648 are unambiguous.
* **Strictly per-cell.** One group at `k = -128` with 255 benign groups leaves **0 of 4080** benign
  cells changed. Unlike the vendor MXFP4 fused op (§4.2 of the conformance probe), one bad scale
  **cannot** poison a whole tensor here: `0 × inf` needs a non-finite operand and the floor emits a
  finite byte. **Silent**, though — the output is a plausible number, and the shipped gate passes.
* **The device disagrees with the kernel too.** `cast_to_fp8_v2` on the HPU maps `2⁻¹⁰ → 0x00`,
  `1.5·2⁻¹⁰ → 0x02`, `3.5·2⁻⁹ → 0x04` (all RNE), `248`/`256` → `0x77` (the 240 ceiling) and
  `−2⁻¹⁰ → 0x80`. So the kernel's tie-break is a deviation from the hardware's own cast, not just from
  the spec's `T.Cast(FP8, .)`.

**Exactness boundary: `k >= -8`** (exponent byte `>= 119`, scale `>= 2⁻⁸`) — the kernel header states
this correctly, and it is confirmed by measurement (0 mismatches in every `t >= -2` cell). What the
header gets wrong is the attribution: *"exact for the checkpoint's scale range"* — **this checkpoint
reaches `k = -9`** (byte 118, 1,070 groups). **As fixed, the correction landed at seven sites, not
five.** This section named the five *copies* — the four sibling kernel arms (`fp4_dequant_arith`,
`fp4_dequant_lut`, `fp4_dequant_clamp`, `kchain`) and the spec-arm gate's `engine_weight_replica`
docstring (`src/gates/step12_routed_spec_arm_gate.py`). Two more were fixed as well: the **shipped
kernel header** itself, and a **seventh** the reviewer had not listed — `deepseek_v4.py`'s own
`_dequant_v4_fp4` docstring, found by grepping the phrase once the first six were fixed.
(`Archive/by-date/2026-10-12/fp4-op-contract/REPORT.md` §4.6.)

**Reachability, measured.** The scales the fp4 kernel is actually fed are
`layers.*.ffn.experts.*.w{1,2,3}.scale`: 33,792 tensors, **byte [118, 126] → k ∈ [−9, −1]** (the
`mtp.` head reaches [119,127] but `DeepSeekV4MTP = None`; the non-expert set reaches [114,121] →
`k ∈ [−13,−6]` but never enters this kernel). Byte 118 occurs in 51 tensors / 1,070 groups, and a
byte-level scan of the packed weights finds **1,399 weights** carrying nibble code 1 or 9 — the only
two codes that engage the floor at `k = -9`. Confirmed on the device on real weights (layer 35 `w1`:
experts 101/104/200/231 → 9/0/41/211 = **261**, exactly the scan's layer-35 total). **The margin is
ZERO**: the checkpoint's minimum byte is 118, which is precisely the byte at which the floor stops
being a tie and becomes an unambiguous over-shoot one step deeper (`<= 117`).

**The fix — and what it does NOT do.** A load-time **guard** (a `raise`, like the top one — *not* an
assertion that silently passes) at **`min scale >= 2⁻⁹`**, i.e. exponent byte `>= 118`, `k >= -9`,
costs **0 ms/step**. **It must NOT mirror the top-end assertion:** the *exactness* bound is `2⁻⁸`
(byte `>= 119`) and **that threshold would refuse to load this checkpoint** (its minimum byte is
exactly 118). The right threshold is the *tie* threshold, `2⁻⁹` — the line at which the kernel is
"never worse than a tie". **It has ZERO margin on this checkpoint.**

> **⚠ Correction (2026-10-12, same day, after review).** An earlier revision of this paragraph said
> `2⁻¹⁰` / "byte `>= 118`" — those two are **not the same threshold** (`2⁻¹⁰` is byte **117**) — and
> then claimed the guard makes the condition "unreachable at zero runtime cost". **Both were wrong.**
> `2⁻¹⁰` permits byte 117, whose 6 engaged cells are *unambiguous* over-shoots (up to 3× relative),
> not ties; and **no guard can fix the 1,399 live weights**, because a guard only *refuses* inputs —
> it cannot change how a permitted input is computed. The live case **is** byte 118, which the guard
> must permit to load this model at all. Measured per-`k` profile of the floor (device bytes, RNE
> reference), which is what makes the threshold choice decidable:
>
> | `k` | byte | cells deviating | max \|error\| | max relative | class |
> |---:|---:|---:|---:|---:|---|
> | **−9** | **118** | **2** | **2⁻¹⁰** | **1 (i.e. 2×)** | **exact ties** |
> | −10 | 117 | 6 | 2.4e−3 | 3 | unambiguous |
> | −11 | 116 | 10 | 3.2e−3 | 7 | unambiguous |
> | −12 | 115 | 14 | 3.5e−3 | 15 | unambiguous |
> | −13 | 114 | 14 | 3.7e−3 | 31 | unambiguous |
> | ≤ −20 | — | 14 | → 2⁻⁸ | → 4e3…1.3e36 | unambiguous |
>
> So deeper bytes are worse **in both senses** — more cells *and* a larger per-cell error (8× from the
> boundary to the `2⁻⁸` asymptote) with an unbounded relative error. **The guard buys:** the deviation
> set shrinks from 14 codes to 2, the per-cell bound from `2⁻⁸` to `2⁻¹⁰`, the relative error from
> unbounded to 2×, and any deeper checkpoint becomes a loud load failure instead of a silent one.
> **It does not buy correctness on this checkpoint** — the 1,399 tie weights need the kernel's
> subnormal rounding changed (`(t >= -2) ? sub : ((t == -3) ? mterm : 0)`, ≈ +3–4 instructions, not
> the +2 that only fixes `t <= -4`), and that must be coordinated with whoever holds the kernel. The
> two are **not alternatives**.

**⇒ The transferable lesson.** A **one-sided guard is a claim about the other side.** The top-end
assertion was honest in its own title (*"upper-bound guard"*) and still misled, because every
description around it — the kernel header, the test's contract scope, and this manual — spoke as if
the range were bounded. And *"unreachable on this checkpoint"* needs the checkpoint's histogram **of
the right population**: the old "256× margin" came from a 5-layer subset, and the reported "−13" came
from a scale population that never reaches this kernel. Both numbers were real; only their
populations were wrong.

#### The e8m0 hand decode was WRONG at byte 255 — **FIXED** and reviewed

> **Status: FIXED, and independently REVIEWED — and the review found the fix itself broken as first
> written.** The fix routes the decode sites through **one helper**, uses `torch.float8_e8m0fnu` where
> it can, and **raises** on byte 255 rather than substituting a value. The review then found **(a) a
> regression that broke the decode path outright** and **(b) that the class was not deleted** — both
> since fixed. See the status block below; the findings are kept because they are the transferable
> part.

**The same class as the unclamped fold above, found by the same sweep — and this one is a defect in
our own loader.**

The e8m0 format is defined by the **OCP Microscaling Formats (MX) specification §5.4.1** (public:
the OCP MX v1.0 specification PDF,
`https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`):

> "E8M0 is an unsigned exponent-only scale format with bias 127. **A finite encoded byte `b` in
> `[0, 254]` represents `2 ** (b - 127)`; `0xFF` is reserved for NaN.**"

**And we confirmed the definition ourselves, so the claim does not rest on the citation:**
`torch.float8_e8m0fnu` reads bytes `0..254` as finite (`0x7F` → `1.0`, `0xFE` → `1.70e38`) and
**byte `255` → `NaN`**. **MEASURED** (our check, torch 2.11).

**So our hand decode is wrong at exactly one byte.** The loader's decode is `2 ** (b − 127)`, which
returns a finite value for `b = 255`; the format says **NaN**. **Torch's own `float8_e8m0fnu` is
right and we are wrong.** **MEASURED**

- **It is LATENT on this checkpoint**: the exhaustive histogram above found **0 occurrences of byte
  255** in **8,858,370,048** routed-expert scale bytes (range `[118, 127]`), and **0** in the
  non-expert scales either.
- **But it is a wrong VALUE at a reachable input — not merely an unreachable branch, and that makes it
  the more dangerous of the two defects in this section.** Compare:
  - the **unclamped fold** is an **unreachable branch**: if it were reached it would emit a *poison
    fp8 byte*, which the format itself marks as inf/NaN and which a reader of the operand could in
    principle notice;
  - the **e8m0 decode** would silently produce a huge/`inf` **scale** and multiply a whole 32-value
    group by it. `inf * 0 == NaN`, so one byte-255 group **silently poisons that group's weights** —
    and nothing downstream flags an `inf` *scale* as a format error, because a scale is just a
    number. **The failure mode is silent in the scale's own path.**
- **The fix, and the sweep that found its real size.** A sweep for the *pattern* (an unconditional
  `2 ** (b − 127)`) found **four sites, not one**:

  | site | path | live? |
  |---|---|---|
  | the block-fp8 scale load | called from three quantisation methods | **LIVE** — load-time |
  | the routed-expert e8m0 load | `_register_v4_routed_packed > _scale_float` | **the `uint8` branch is DEAD for this checkpoint** — every scale tensor in it is stored as **`F8_E8M0`** (**33,792 expert + 375 non-expert**, read from the safetensors headers), so `_scale_float` takes its dtype branch and never reaches the hand decode. Reached only through the **upstream loader's uint8 view** — i.e. via the weight wrapper, which *is* the site that reads those 8.86 × 10⁹ bytes |
  | the pure-Python dequant fallback | the non-custom-op control path | not production |
  | the fp4→fp8 pre-dequant helper | not on the production path | not production |

  **All four now route through one helper**, so the *class* is deleted rather than the instance
  patched four times. **That is the argument for fixing a class rather than an instance: the sweep
  found four where inspection had found one.** **MEASURED**
- **And the sweep still undercounted — the review is what found out. A pattern sweep finds the
  *spelling* you searched for, not the pattern.** The review found **at least six** hand-rolled e8m0
  decodes, including one in a **different model** (`gptoss_mxfp4.py`: `scales − 127` then
  `torch.ldexp`, so byte 255 → `inf` — **the same defect**, invisible to the gate's structural check
  because it is not spelled `pow(2.0, …)`) and one in this file that **silently accepted** `0xFF`
  instead of refusing it. **Widen the search to the arithmetic the pattern implies** (`exp2`,
  `ldexp`, `2.0 **`, `<< 23`), not the call it happens to be written as. All now route through the
  helpers. **MEASURED**
- **The loader RAISES on byte 255** rather than propagating NaN or substituting a finite value.
  Rationale, in order: a NaN scale is unambiguously a corrupt checkpoint (no MX quantiser emits one)
  and the load path's job is to fail loudly; propagating defers the failure to inference, where NaN
  logits are ~43 layers from one bad byte; and **never substitute a finite value** — that converts a
  detectable data error into a wrong number.
- **The dtype can be used after all — for the *conversion*.** `is_supported_type` rejects `e8m0`
  *and* `e4m3fnuz` (§3.4.1), but that is a constraint on **allocating an HPU tensor**, not on
  converting one: `bytes.view(torch.float8_e8m0fnu).float()` works and gives NaN for 255. So the fix
  prefers the dtype (it cannot drift) with a guarded arithmetic fallback.
- **Cost: one `max()` reduction (0.32 ms per 15.7 M bytes), on the `uint8` path only.**

> **Status block — what "FIXED" does and does not mean here.**
> - **The change is to tracked code** in the port's own source tree (**a 106-line diff**).
> - **The fix as first written BROKE the decode path — and the review caught it by executing, not by
>   reading.** The helper was imported **function-locally**, so the three call sites that use it had
>   **no binding at all** and raised `NameError` for **every** uint8 input: strictly worse than the
>   bug it fixed, on a path the port's own comments call live. Fixed by a **module-level import**.
>   **⇒ Two lessons: import scope is not a style question when one helper is called from several
>   functions; and a gate that imports the helper directly and never imports the *calling module*
>   cannot see this class of break at all.**
> - **The gate had a tautology on its PASS path.** Its "all finite bytes match the dtype" check
>   compared the helper's own body against itself — it has content only in break mode. **Use an
>   independent reference** (a literal 256-entry table, or `2.0 ** (b − 127)` in float64).
> - **A structural AST check** accompanies the fix, so the pattern cannot silently reappear outside
>   the helper — a gate on the *class*, matching the fix. **Its limits, stated:** it matches **one
>   spelling** (`pow(2.0, x − 127)`), so `exp2`, `2.0 **`, `ldexp` and `<< 23` all evade it — and
>   **two of those existed in the tree**. Widen it when you copy the idea.
> - **The refusal's blast radius is the whole load, and that is a design choice worth making
>   consciously.** The guard is on the **whole tensor**, so **one** `0xFF` byte anywhere in an 8.9
>   G-byte scale population aborts the load of the entire model, with no way to skip or repair. If any
>   checkpoint or vendor tool uses `0xFF` as "unset"/padding, that converts a working load into a hard
>   failure. **Prefer a NAMED exception, report the offending indices (not just the count), and give
>   the operator a switch.** *(The policy itself is now shared by both the value and the exponent
>   representations, so at least it is applied uniformly.)*

> **Why we are hand-rolling this at all — and why that makes it worth getting right.** The hand decode
> is **forced, not a shortcut**: two fp8-family dtypes are unusable on this stack. `float8_e4m3fnuz`
> is rejected outright by `is_supported_type` (§3.4.1), and the `e8m0` dtype is likewise not usable
> through this path. When the platform gives you no supported dtype for a format you must read, **you
> own the decode's correctness** — and this is the second defect found in a hand-rolled decode in this
> section.

*Corroboration, and how not to read it.* A vendor **test helper** clamps rather than special-cases —
`return max(0, min(255, exp + 127))` — so it **allows** 255 (the vendor's PyTorch bridge,
`tests/pytest_working/any_mode/test_hpu_mixture_of_experts_mxfp4.py:83-88`). **Do not read the test
helper as contradicting the spec**: it is a test helper, not a format definition. The OCP citation is
the authority.

**Don't pre-dequantize at load.** Converting fp4 → fp8 at load time **doubles the weight bytes**
(0.5 → 1 B/value), taking the routed buffers from **36.8 → 71.4 GB/rank** (+34.6 GB), on a 96 GB card
with **12.3 GB free** ⇒ **HBM-blocked**. **NOTE-CLAIM [DSv4F]**

**Read only the routed experts.** Reading the 6 routed experts of 64 local is **10.7×** less traffic
than reading **all 64 local** experts (**36.8 GB/step ÷ 3.449 GB/step**; `64 / 6 = 10.7`), and against
**§3.4.6's 44.64 GB** materialised path the fused read of the 6 routed experts is **12.9×** less while
the fused read of **all 64 local** experts collapses to **1.21×** (`44.64 / 36.8 = 1.21`). **I** —
*this is the single check that decides whether a fused-expert optimisation is worth anything.*

> **CORRECTED 2026-10-14 — the sentence this replaces read *"Reading the 6 routed experts of 64 local
> gives a 12.9× traffic win"*, and that ratio does not reproduce for the comparison its wording states;
> the original is kept here as the record.** Reading 6 of 64 local against all 64 local is
> **36.8 ÷ 3.449 = 10.67×** (and `64 / 6 = 10.67`). The `12.9×` is real but belongs to a *different*
> pair: the routed-only read (**3.45 GB**) against §3.4.6's **44.64 GB** materialised BEFORE total
> (`44.64 / 3.45 = 12.94×`) — which is also the baseline of the `1.21×` beside it
> (`44.64 / 36.8 = 1.21×`). **Both comparisons are now stated, and the conclusion is unchanged: read
> only the routed experts.**

### 3.4.6 The materialised-traffic model — and why you must not trust its biggest number

A shape × call-count model of the routed-expert path. **MEASURED-as-model [DSv4F]**

| stage | read GB | write GB | total GB |
|---|---:|---:|---:|
| gather packed fp4 | 3.25 | 3.25 | 6.49 |
| gather fp32 scales | 0.81 | 0.81 | 1.62 |
| kexp build | 0.81 | 3.25 | 4.06 |
| dequant kernel | 6.49 | 6.49 | 12.99 |
| **`torch.cat([o0,o1])`** | 6.49 | 6.49 | **12.99** |
| `fp8_gemm_v2` weight read | 6.49 | 0 | 6.49 |
| **BEFORE total** | **24.35** | **20.29** | **44.64** |

Amplification **13.74×** against the 3.25 GB the path itself reads.

> **⚠ THE WEAKEST LINK IN THE TRAFFIC STORY. The profiler trace CARRIES NO BYTE COUNTS.**
> **44.64 GB is a shape × call-count MODEL, not a measurement.**
>
> Its cross-check: the MoE recipes' measured **DMA merged busy** is 14.65 ms/step ⇒ **29.3 GB at
> 2 TB/s**, against the model's DMA share (31.6 GB) — agreement within **8 %**, so the model is
> **consistent with the trace but not verified by it**. **MEASURED**
>
> **And the single largest component — `torch.cat`, 12.99 GB, 28 % of the total — is
> UNCORROBORATED.** The only cat-attributed device ops are **4,365 `DmaTranspose`/step at 0.219 ms
> merged** — **~12× BELOW** what 12.99 GB needs at 2 TB/s (which would be 6.5 ms). Either the cat is
> not lowered to those events, or it is folded. **MEASURED**
>
> **⇒ Do not build a doctrine on the 12.99 GB.** Do not quote it as a measurement.

---

### 3.4.7 The vendor MXFP4 fused op — is it a drop-in for a new model?

**The op.** `torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights` is a vendor fused MoE for packed
MXFP4 weights: it takes packed `uint8` weights plus `uint8` E8M0 scales and does the dequantisation,
the two expert GEMMs, the activation and the routing **inside one op** — i.e. it is the fused
alternative to the six materialising stages of §3.3. Our tree wraps it at
`src/vllm-gaudi/vllm_gaudi/extension/ops.py` (`VllmMixtureOfExpertsOpMXFP4`, which passes
`permuted_weights=True` hardcoded and selects the `bias_*` variant purely on `has_bias`). **source read** — the artifact is named and quotable, but nothing was executed.

**Why this subsection exists.** *"Can we adapt the stock fp4 builtin to our model?"* is a recurring
question, and the answer is not yes-or-no — it is a **table of axes, each pass / fail / UNKNOWN**.
What follows is that table, from an **op-level unit probe**: one HPU (module 3, pinned
`HLS_MODULE_ID=3`), **no model, no vLLM, no server** — **under 30 s of device time for the
conformance sweep, and ~6 s for the separate activation probe.** Where an axis was not tested it says
**UNKNOWN**, never FAIL.

*Artifacts. The conformance table and every number in it:
`Archive/by-date/2026-10-12/fp4-builtin-conformance/` — `REPORT.md`, the decisive script `xprobe.py`
(the dequant-as-readout table), `consprobe.py` (block size / dtype / shape constraints), `compare.py`
(offline head-to-head against our op), `bench_ours.py`, and `out/*.json`. The activation and layout
axes: `Archive/by-date/2026-10-12/a1-probe/` — `REPORT.md`, `mxfp4_act_probe.py`, `A1.json`. The
clamp census behind the DSv4F line below:
`data/Archive-2026-10-12/a0b-clamp-census/REPORT.md`.*

**Bridge line numbers in this subsection are pinned to a revision.** Every
`mixture_of_experts_gen.cpp:NNN` reference here is to **`gaudi-pytorch-bridge` v1.24.1** (commit
`5176b1b2d6865608514e83a2b1fd66459920676e`, `github.com/HabanaAI/gaudi-pytorch-bridge`) — the
revision our image ships. A newer upstream `master` carries the same code about 20 lines later
(`:1441` → `:1461`, `:1553` → `:1574`, `:1448` → `:1468`), so a check against `master` lands in the
wrong place. **The numbers are right for the revision we measured, not for `master`.**

#### The readout — why the table is a measurement and not a fit

The op's GEMMs are made **transparent**. With `H = F`, `x = ones(T, H)`, exactly one non-zero per row
of `w12` at column `k` (`1.0` in the gate half, the test value in the up half) and `w3 = I_F`, the
device output **is** `relu(gate) · up` at inputs we chose — i.e. the **vendor decode of
`(nibble_k, e8m0_byte_k)`, in bf16, with no weight-quantisation error and no activation error**. One
call reads `F = 128` independent `(nibble, scale-byte)` pairs; the full 16 × 256 isolated table is
4096 cells and takes **2.42 s**. **MEASURED**

> ⚠ **A harness bug that produced a *plausible wrong answer* — kept because it is an instance of the
> op's most important property.** The first table put all 128 channels of a call in one byte group
> (bytes 248..255 together) and reported *"bytes 248–255 decode to NaN for every nibble"*. That is
> **false**: byte 248 decodes correctly to `6 × 2^121 = 1.59507e+37`. It was wrong because of trap 1
> below — the byte-255 channel's weight overflows bf16 to `inf`, and the down-projection GEMM then
> computes `0 × inf = NaN` for **every other channel**. **A dequantisation probe must isolate each
> cell.** The superseded table is kept as evidence (`out/decode_table_device.pt`).
> **MEASURED**

#### The conformance table

| # | axis | result |
|---|---|---|
| 1 | **activation** | **plain SILU, NO clamp** — `h = silu(gate) · up`. `gate = 40, up = 40` → the device returns **1600.0** (a clamp at 10 would give 99.9955; at 7.0, 48.9554). Fitted over 8192 cells: no-clamp `rel = 6.30e-04` (96.24 % within 1 bf16 ulp), against **15.0** for clamp-±10 and **31.7** for clamp-±7.0; on the **1505** cells where the hypotheses differ by more than 2 ulp the device is no-clamp on **100.00 %** and clamp-±10 on **0.00 %**. ⇒ **not a drop-in for a model that clamps.** **The kernel's SILU is not bit-identical to a torch `silu`** — 96.24 % of cells within 1 bf16 ulp, 99.93 % within 2, max **2.13 ulp** — which is why the fit above has a residual at all rather than being exact. **MEASURED** |
| 2 | **`limit` parameter** | **INERT on all three activation modes.** All three activation strings are honoured and none clamps — `relu(40) = 40` on the device, not 7. The clamped path is not merely unselected, it is **unreachable through this op**: `gpt_swiglu` is hardcoded `false` (`hpu_ops/backend/mixture_of_experts_gen.cpp:1441`), so the `1.702 / 7.0` pair that the same file writes into the params is never consumed. **MEASURED** (the inertness); **source read** (the unreachability) |
| 3 | **gate/up layout, non-bias path** | **HALVES, not interleaved** — `w12 = cat(w1, w2, dim=0)`. `halves` fits at `rel = 6.3e-04`; `interleaved` at **`rel = 1.0000`**, i.e. *completely* wrong, so the two are not confusable. ⇒ **a silent gate/up-swap trap.** **Scope: this is the NON-BIAS op only.** The bridge's *bias* / GPT-SwiGLU path is a **different op and a different code path** (`gpt_swiglu = true`, `mixture_of_experts_gen.cpp:1553`, against `false` at `:1441`), and its layout is **UNKNOWN** (§3.9 item 17). Do not read the two as a disagreement — see the note under this table. **MEASURED** |
| 4 | **e8m0 decode, bytes 0..254** | **PASS** — **4052 / 4068** finite cells are bit-identical to the correctly-rounded IEEE bf16 of the exact OCP MX value `fp4 × 2^(b−127)`. The 16 that differ are all **flush-to-zero** (see below), and the 28 non-finite cells are *exactly* `{true product exceeds bf16 max} ∪ {byte 255}` — verified by **set equality**. **MEASURED** |
| 5 | **fp4 exponent fold at the top of the range** | **PASS — the op does NOT have our defect.** There is no reserved bin and **no sign clobber**: the overflow is IEEE-correct with the **sign preserved** (code 7, byte 253 → `+inf`; code 15, byte 253 → `−inf`). This is the axis the port actually cared about, and it is the one place the vendor op is **better** than ours. ⇒ **a port does not inherit the `t = exp + k` bug class from this op** — but it must not *assume* our saturating semantics either. **MEASURED** |
| 6 | **block / group size** | **`block_size` is INERT — every value behaves as 32.** `8 / 16 / 64 / 128 / 0` all produce the `block_size = 32` answer, and a scale tensor shaped for anything other than `ceil(cols/32)` columns is a hard **graph-compile failure** (`synStatus 26`). **MEASURED** (the failure); **INFERRED** (the shape attribution — see the `synStatus 26` caveat below) — and it **contradicts the kernel source's own comment** (trap 2). |
| 7 | **input dtype** | **bf16 only.** fp16 and fp32 are **rejected at graph compile** (`synStatus 26`) — **the reason we recorded with it, "the cguid has no moe kernel for those precisions", is an INFERENCE the error code does not carry** (see the `synStatus 26` caveat below). **MEASURED** (the rejection); **INFERRED** (the reason) |
| 8 | **shapes** | **PASS — no alignment requirement.** `T ∈ {1,2,3,5,8,16,17,32,33,64,65,128,129,256,512,1024}` all correct; `H × F` including the non-multiples **31, 33, 65** all correct, with the tail beyond `F` exactly `0`. Output bf16, shape == `hidden_states`; padding is internal. **MEASURED** |
| 9 | **bit-exactness vs a guarded custom dequant** | **NOT bit-exact — 742 / 748 cells identical** over the region both output formats can represent; the 6 differences are all a **subnormal-floor** effect. See below. **MEASURED** |

**A non-conflict worth stating, because the naive reading of row 3 is a false alarm.** The
"interleaved gate/up split" the bridge *documents* belongs to the **`bias_mxfp4_fused_weights` /
GPT-SwiGLU path** (`gpt_swiglu = true`, `hpu_ops/backend/mixture_of_experts_gen.cpp:1553`), which is a
**different op and a different code path** from the one measured here (`mxfp4_fused_weights`,
`gpt_swiglu = false`, `:1441`). So the `halves` result above is **not contradicted** by it, and the two
must not be presented as a disagreement. What the *bias* path's split actually is remains **UNKNOWN**
(§3.9 item 17) — it is the arm that wedges the card. *(The one genuine disagreement in the vendor
artifacts is doc-against-doc and about that path: a bridge test docstring
(`tests/pytest_working/any_mode/test_hpu_mixture_of_experts_mxfp4.py:745`, in the same archived bridge
tree as the `gen.cpp` comment above) says "interleaved", while **vllm-gaudi**'s own `swiglu_oai`
reference (`vllm_gaudi/models/minimax_m3_sparse.py`) says split/first-half. Neither is a device
measurement, we have not resolved it, and we do not claim to.)*

> **A harness trap inside axis 8, recorded because it faked a result.** The first shape sweep
> reported `FailOnRecompileLimitHit` for 30 of 40 configurations — it measured the *harness*, not the
> op, because dynamo's recompile limit is **per code object** and a fresh `@torch.compile` closure
> does **not** reset it. `torch._dynamo.config.recompile_limit = 512` fixed it. **MEASURED**

**`synStatus 26` / "Current graph is nullptr" is NOT diagnostic either** — it is the sibling of the
`synStatus=8` caveat in §3.7.3, and it needs the same warning. Three places in this subsection lean
on it as a *cause*: the `block_size`-shape failure and the dtype rejection in the table above (rows 6
and 7), and the three `block_size` / scale-column arms in trap 2. It records **that a graph compile
failed, and nothing more**. A device experiment on this box (one process per arm) shows what it
actually tracks: whether the requested **kernel name is present in the active library's GUID table**.
The GC never calls `InstantiateTpcKernel` for a missing name — the failing step is a lookup *before*
instantiation — the error surfaces at the next `synchronize()` rather than at the op call, and **no
log at the default level names the missing kernel** (a search of `/var/log/habana_logs` after the
failure found **zero** occurrences of the absent name). A concrete instance of this class is now
measured: a **late `GC_KERNEL_PATH` write** reports `applied` and then dies `synStatus 26` at the
first fp4 compile, because the library was never swapped at all — the requested name *is not* in the
active table, and the table is the wrong one (the ordering bug behind that instance is itself now
fixed — §3.4.8). The identical string is produced by a missing GUID with a **single** extension
loaded, and is **absent** when two extensions are loaded and every GUID is present — so it is not
evidence about extension count either, contrary to a rule we had recorded in three places in the
source project. Never read it as "the shape was wrong", "the dtype was wrong" or "the library was
wrong": **it says a lookup failed, not what failed.** What would settle the *cause* in
each case is the GC's own log at a raised level (`LOG_LEVEL_GC`) — **not attempted**, so the
*attributions* in rows 6/7 and trap 2 are **INFERRED**, and only the *rejections* are **MEASURED**.
*(Source: `Archive/by-date/2026-10-12/fp4-guid-probe/REPORT.md` §4–§5 — five device rows, one process
each, `HLS_MODULE_ID=3`, shipped `.so` md5 unchanged before and after.)*

#### The other face of `synStatus 26`: the message TEXT is the handle, and it names the node

The GUID-lookup mechanism above is one cause. **`synStatus 26` is `synFail`** — a generic failure in
the status enum (`synapse/include/synapse_common_types.h`, where `synInvalidArgument = 1`,
`synUnavailable = 24`, `synInvalidTensorDimensions = 25`, `synFail = 26`) — so the number alone can
never be diagnostic. What *is* diagnostic is the **message text**, and one form of it carries more
than the section above suggests.

**MEASURED, on a parallel port of this model:** `synGraphCompile: Invalid Node Params! Node name:
n_aa`, `synStatus 26`, raised at a `synchronize()` flush point. The cause was **not** a missing
kernel. It was an in-place copy into an **offset view of a larger buffer with a mixed dtype**:

```python
packet = torch.empty(b * 10244, dtype=torch.int32, device=dev)
pre    = packet[b * 10240:].view(torch.float32).view(b, 4)   # offset view, same itemsize
pre.copy_(values["pre_mix"])                                 # this was rejected
```

It compiled at `b == 1` and failed at `b == 2`; removing the copy fixed it. **That attribution is
theirs and rests on one A/B** — no log line isolated the node's params — so read it as a reported
cause, not a demonstrated mechanism.

**The message has two forms, and the suffix tells you which constructor fired** (source read,
`synapse/src/infra/types_exception.h`, `InvalidNodeParamsException`):

| text | constructor | what it adds |
|---|---|---|
| `Invalid Node Params! Node name: <n>` | one-arg | a node whose *validation or shape inference* failed |
| `Invalid Node Params! Node name: <n>, param name : <p>` | two-arg | **names the offending user parameter** — far more useful when you get it |

**And the companion line that names the cause does exist — it is just not emitted at the default
log level.** These four strings are all present in the shipped `libSynapse.so` (sha256
`6796f09dac786b95970f55541c8c27dd4005e87a5ecd1b95bb11daec6f3b3913`):

```
Node Validation Failed. Cannot create node {}.
Failure to update output shape for node: "{}"
projectDmaRoisToFullSizes No output roi for node {}
Graph optimization failed pass: {}
```

So the recipe for this form is **raise the level and re-run**, then grep for those four — not "there
is no line". **Verified empirically** on that port's failing run: `synapse_runtime.log` carries only
the top-level re-print (`[SYN_API][critical] exception thrown in function: synGraphCompile, what:
Invalid Node Params! Node name: n_aa`) while its `graph_compiler.log` is **0 bytes** — the
node-naming lines are `GC` / `HABANA_NODE` / `ROI_SPLITTER` / `PASS_MANAGER` category `LOG_ERR`s and
are silent until `LOG_LEVEL_ALL`/`LOG_LEVEL_GC` is set. **A zero count here is a claim about the
instrument** (§3.10 item 23).

**What `n_aa` is — and why it never resolves in a dump.** It is **not** an op name. It matches the
**eager** graph's own name generator (source read, `synapse/src/eager/lib/eager_graph.h`,
`NameBuilder`): prefix `n_`, then **two base-16 letters** (`a`–`p`), starting at `n_aa` for id 0. So
it is an **insertion index in the eager graph** — `n_aa` is the first auto-named eager node. Two
consequences: it will never appear in a `DUMP_POST_GRAPHS` post-graph, whose nodes are named
`<op>/<guid>/<cnt>`, so "the name does not resolve" is expected and not a dead end; and **seeing an
`n_XX` name at all tells you the failing graph is the eager one**, which is a real discriminator when
you are deciding whether to look at the compiled-recipe path or the eager path.

**Reconcile this with the paragraph above, which is not retracted.** "Never read `synStatus 26` as
'the shape was wrong'" is right about the **number** and wrong if applied to the **message**: given
`Invalid Node Params!` plus a node name, a shape, view or alias cause is exactly what the companion
lines report. The GUID-lookup story remains the mechanism for the *unnameable* form; this is the
mechanism for the *named* one. They are different failures sharing a status code.

**Provenance, because it matters here.** The four message strings and the two message forms are
verified in the **shipped 1.24.2** `libSynapse.so` above. The **throw-site enumeration** — the
one-arg constructor is thrown from `graph_compiler/habana_nodes/node_factory.cpp` (a failed
`validateNode()`, and a failed max-dims shape inference) and from
`graph_compiler/roi_splitter.cpp` (`projectDmaRoisToFullSizes`, a DMA ROI with no output ROI), with
the two-arg form thrown from many `habana_nodes/*.cpp` sites passing `"userParams"` — is read from a
vendor Synapse **1.16-era source snapshot**, not from 1.24.2. Treat it as a map of *where the message
comes from*, not as a line-exact citation for the shipping build; **the count of throw sites in
1.24.2 is UNKNOWN** here, and the one place the snapshot and the shipped binary disagree is the
pass-level message text (`Graph optimization failed! Got SynapseException: {}` in the snapshot,
`Graph optimization failed pass: {}` in the binary) — which is why the binary is quoted above and
the snapshot is not. **The `n_aa` decode is INFERRED, not source-read:** it matches that snapshot's
generator format exactly (and the name is identical on all four ranks and across repeats, as a
per-graph counter starting at zero would be), but `NameBuilder` is a header-only class, so there is
no symbol in the shipped binary to confirm it against. *(A vendor-source enumeration of the 1.24.2
throw sites would settle it; our attempt to consult one returned HTTP 503 during this work, so that
is "could not consult", not "no evidence".)*

**And do not over-generalise this caveat.** The dump-directory `synStatus 26` failures elsewhere in
this manual — Chapter 2 §2.10 trap 1 and Chapter 1 §1.11 — are **not** in this class: they rest on
their own controlled A/B evidence and carry **distinct corroborating symptoms** of their own
(`rename failed after 3 retries … errno: 2`, i.e. positive evidence about a missing path, alongside
`size mismatch after 3 retries … expected 29855 got 0`), so they **stay MEASURED**. This caveat is
about `synStatus 26` read as a *cause* on its own; it is not a general retraction of every
`synStatus 26` in the chapter.

#### ⚠ TRAP 1 — one out-of-range scale NaNs the ENTIRE layer, not one channel

**This is the single most important porting fact in this subsection.**

With exactly **one** channel carrying an overflowing weight — `(code 7, byte 254)` — and the other
127 channels exactly `0.0`, the probe channel returns `+inf` and **all 1016 benign outputs (127
channels × 8 tokens) return NaN**. The down-projection GEMM computes `0 × inf = NaN`, so there is
**no per-channel containment**. **MEASURED**

**The op has no guard on the scale.** Contrast our own path, where the analogous failure is
*bounded*: the stack refuses `0xFF` at load and asserts a bound on the max scale (§3.4.5), and the
kernel saturates to the fp8 ceiling. On the vendor op a bad checkpoint scale is **global** — the
whole layer output becomes NaN, not one weight. **MEASURED** (the whole-layer NaN) / **INFERRED**
(the absence of a guard — an absence claim, supported by the byte-255 row below)

**And byte 255 is not refused either.** Even `(code 0, byte 255)` — a weight of exactly `0.0` —
returns NaN, which requires the *scale* to be non-finite. **UNKNOWN whether that scale is `inf` or
`NaN`**: the op cannot distinguish them (`0 × either = NaN`). Either reading means the op lets a
non-finite scale into the tensor rather than rejecting it. **MEASURED** (the NaN); **UNKNOWN** (which
non-finite).

⇒ **Before choosing this op, histogram the real scale tensors.** On our own checkpoint the
routed-expert scales live at bytes `[118, 127]` and the non-expert scales at `[114, 121]`, so byte
255 is unreachable *here* — which is exactly why the guard is cheap to keep and easy to lose.
**MEASURED** — the histogram and the byte ranges come from our e8m0 byte-255 gate,
`src/gates/step_e8m0_byte255_gate.py`.

#### ⚠ TRAP 2 — `block_size` is a decoy

The op *takes* `block_size`, and the bridge's own source comment says why, verbatim
(`hpu_ops/backend/mixture_of_experts_gen.cpp:1448-1450` in **bridge v1.24.1** — upstream `master`
has this at `:1468`; see the revision note above. Our copy of that tree is archived at
`data/Archive-2026-10-12/reference-diff/bridge/`):

```
// block_size must always be set for MXFP4: the cguid-inserted
// cast_packed_mxfp4_to_bf16 kernel uses it to index E8M0 scales
// regardless of whether expert-parallelism chunking is active.
```

**It does not.** Measured with a discriminating design — scale column 0 = byte 127 (scale `1.0`),
every other column = byte 0, so a *honoured* `block_size` would give `6.0` for channels `k < bs` and
`~0` for `k ≥ bs`: **MEASURED**

| declared `block_size` | scale columns | `k = 0..24` | `k = 32` | verdict |
|---:|---:|---:|---:|---|
| 32 | 4 | 6.0 | 0.0 | baseline, correct |
| 64 | 2 | — | — | **compile FAIL `synStatus 26`** |
| 64 | 4 | 6.0 | **0.0** | honoured would give `6.0` at `k = 32` ⇒ **inert** |
| 128 | 1 | — | — | **compile FAIL** |
| 128 | 4 | 6.0 | 0.0 | **inert** |
| 16 | 8 | — | — | **compile FAIL** |
| 16 | 4 | 6.0 | 0.0 | honoured would give `0.0` at `k = 16`; it gives `6.0` ⇒ **inert** |
| 8 | 4 | 6.0 | 0.0 | **inert** |
| 0 | 4 | 6.0 | 0.0 | **inert** |

⇒ **Fixed at 32.** The declared value is a no-op for E8M0 indexing, and the scale *shape* is what is
enforced — by a compile-time failure, not by a runtime check. The vendor's own test asserts the same
(`tests/pytest_working/any_mode/test_hpu_mixture_of_experts_mxfp4.py:103`:
`assert block_size == MXFP4_GROUP_SIZE, "TPC kernel requires group_size=32"`), and our own
`fp4_dequant` uses the same 32-wide group. **A port that "sets `block_size` to match the checkpoint"
silently gets group-32 semantics.**

**This is the §3.4.5 lesson in the vendor's tree rather than ours: a source comment is a claim, not a
measurement.** It is stated plainly here because a porter will find the comment and not the
measurement.

#### Not bit-exact — even where the two formats agree

Over the region **both output formats can represent**, **742 / 748 cells are bit-identical and 6
differ, by 64 bf16 ulp**: **MEASURED**

| cell | exact | vendor | ours |
|---|---:|---:|---:|
| code 3, byte 118 (`k = −9`) | 0.0029296875 | 0.0029296875 | 0.00390625 |
| code 5, byte 117 (`k = −10`) | 0.0029296875 | 0.0029296875 | 0.00390625 |
| code 7, byte 116 (`k = −11`) | 0.0029296875 | 0.0029296875 | 0.00390625 |
| *(plus the three negative mirrors, codes 11 / 13 / 15)* | | | |

Our op's e4m3 output grid **floors** magnitudes below `2^-9` to `2^-9` and **rounds within the
binade**, so `1.5 × 2^-9` lands on `2^-8` — a factor of **4/3**, which is the 64 bf16 ulp quoted
above (measured against the exact value's binade).
**These cells are inside the checkpoint's own scale range**: 52 of 128 cells differ over bytes
`[114, 121]`, and 4 of 160 over the routed-expert range `[118, 127]`. There are **zero** arithmetic
disagreements outside our fp8 representability, so this is a **format limit, not an arithmetic
error** — but it is real, it is quantified, and it means the two ops are **not interchangeable at
small magnitudes**. **⇒ "Adapt the builtin" is not a drop-in even where the formats agree.**
**MEASURED**

**The vendor decode also flushes subnormals to zero, and our kernel does not.** Byte 0 → scale
`2^-127` is an **fp32 subnormal**, flushed to `0.0` for every nibble — including code 7, whose true
product `3.53e-38` is a perfectly *normal* bf16. A **bf16-subnormal product** flushes too (code 1,
byte 1 → `2^-127` → `0.0`; code 1, byte 2 → `1.18e-38` → kept). Our kernel instead floors to its own
fp8 subnormal minimum. **Both are wrong; they are wrong differently.** **MEASURED**

#### What to check before adopting

Adopt the vendor op **iff all six hold.** Anything else is a rewrite or a rejection, not a config
change.

1. **Its activation matches the model's.** This op is plain SILU with no clamp, and the clamped path
   is **unreachable through it**. A model that clamps the gate/up is **not** a candidate — and this
   check alone rejects **DSv4F**, whose spec is `silu(clamp(gate, max=10)) · clamp(up, ±10)`: the
   census prices dropping that clamp at worst-frame `rel = 1.184`, with 7 of 96 decode frames above
   `8.524e-2`. **MEASURED [DSv4F]** (CPU census over captured decode frames)
2. **Its gate/up layout matches the checkpoint's.** The non-bias path is **halves**; the checkpoint
   must be `cat(w1, w2, dim=0)`. An interleaved checkpoint does not fit — and the mismatch is
   *silent* in the sense that the op still returns a plausible number, wrong by `rel = 1.0`. Verify
   it; do not assume it.
3. **Every scale in the checkpoint is provably in range.** There is **no runtime guard** — an
   **INFERRED** absence claim, whose measurement is the whole-layer NaN below. The first byte that
   can overflow is **253**, not 248: byte 248 decodes to `6 × 2^121 = 1.59507e+37` and byte 252 to
   `6 × 2^125 = 2.552e38`, both below bf16 max `3.3895e38`; the derivation is
   `6 × 2^(b−127) > bf16 max ⟺ b ≥ 253`, which first bites at nibbles 6/7/14/15. One such byte turns
   the **entire layer output** into NaN (trap 1). Histogram the real scale tensors first, and check
   byte 0 for the FTZ.
4. **The input is bf16.** fp16 / fp32 do not compile.
5. **A 32-wide block matches the checkpoint's quantisation.** `block_size` is inert; the *shape* of
   the scale tensor is not, and any column count other than `ceil(cols/32)` fails to compile.
6. **The tolerance budget accepts an IEEE-±inf-vs-saturation difference at the top of the range.**
   Ours saturates at the 240 ceiling inside its contract; the vendor overflows to `±inf`. Both
   preserve the sign — that is the only thing they agree on up there.

**The one-line version:** *the vendor op is a drop-in only for a bf16-input, SILU, no-clamp, block-32
MXFP4 model whose scales keep every `|weight| ≤ bf16 max`.*

#### Performance: measured, but NOT like-for-like — do not quote a ratio

A microbenchmark exists and is deliberately **withheld as a ratio**. Vendor **whole fused op** at
`T = 512, H = 4096, F = 2048`: **0.591 ms** median (min 0.519). Our `fp4_dequant` **alone** at the
same packed size (16.8 M fp4 values): **0.409 ms** median (min 0.361). **The vendor number buys
routing + two GEMMs + the activation; ours buys a dequant only** — so a naive ratio of the two
numbers would be misleading in the vendor's favour by an unknown factor. The honest statement is: the
fused op is cheap (**0.6 ms** at a production-ish decode shape); our dequant alone is the same order
**as the vendor number, which buys more work**; and **the GEMM term is not measured here**. **A
like-for-like chain benchmark is not yet built.**
**MEASURED / NOT COMPARABLE**

#### What this subsection does not establish

| # | UNKNOWN | why it is open |
|---|---|---|
| 1 | the on-device semantics of the **GPT-SwiGLU clamped path** (`limit`, `alpha`, the gate/up split) — **a different op** (`bias_mxfp4_fused_weights`, `gpt_swiglu = true`) from the one measured above | the op that reaches it **wedged the compute queue** with a 30 s timeout on its first execution and was not retried. Its split is **UNKNOWN**: a bridge test docstring (`tests/pytest_working/any_mode/test_hpu_mixture_of_experts_mxfp4.py:745`) says *interleaved*, while vllm-gaudi's own `swiglu_oai` reference (`vllm_gaudi/models/minimax_m3_sparse.py`) says split/first-half — a doc-against-doc disagreement we have not resolved, and **not** evidence about the non-bias path's measured `halves`. The clamped formula we have is framework source, **not** a device measurement. (§3.9 item 17.) |
| 2 | whether the byte-255 scale is `inf` or `NaN` | the op cannot distinguish them (`0 × either = NaN`). (§3.9 item 18a.) |
| 3 | the mechanism of the flush-to-zero | measured behaviour, unmeasured cause. (§3.9 item 18b.) |
| 4 | **multi-expert correctness** | the probe used one expert; a `topk` routing table is a different contract, and a naive all-zeros table gave garbage. (§3.9 item 19.) |
| 5 | like-for-like cost of the vendor op vs a dequant + fp8-GEMM chain | not built (above). (§3.9 item 20.) |
| 6 | cost at the real production shape and real expert count | the microbenchmark shape is synthetic. (§3.9 item 20.) |

---

### 3.4.8 Promoting a custom kernel variant: the library is the interface

*(Added 2026-10-13. This is the fp4 decode's variant-selection question — the fix §3.4.5 specifies for
the subnormal floor lives in one of the variants — and it **corrects a reading of our own**: the
wiring's blocker had been recorded as a missing GUID. It is not — the blocker was the late write, and
that is fixed below.)*

**The problem, and its two halves.** A variant-selection wiring (`dsv4_fp4_variant.py`) picks a kernel
variant from the checkpoint's routed-expert scale bounds and then writes the environment, from the
**load path**:

```
prev = os.environ.get("GC_KERNEL_PATH")
os.environ["GC_KERNEL_PATH"] = target          <-- the write
```

**That write could not work.** The worker fixes its kernel database at its **first HPU op** — when the
compiler builds its kernel DB and reads the variable — and the load path reaches this write ~40 s
later, so it landed **too late**. And the failure was not loud: with `GC_KERNEL_PATH` at the **image
default** (`/usr/lib/habanalabs/libtpc_kernels.so`) the wiring's `too_late` detector was **blind** —
it filtered `/proc/self/maps` against `known_kernel_libs()`, a list that **deliberately excludes the
image default**, so the code could not see that a database was already fixed. It **reported
`applied`**, wrote a path nothing would read, and the run then died at the first fp4 compile with
**`synStatus 26`**. **MEASURED**, and *reproduced first* on the unmodified tree: `load-path verdict =
'applied'` → `RuntimeError: Graph compile failed … synStatus 26` → `promoted library LOADED = False`.

**Both halves are now FIXED — the blind detector, and the ordering, which is now structural rather
than "usually true".** The install moved off the load path onto `register_utils()`, the
**`vllm.general_plugins` hook**, which runs at `WorkerWrapperBase.init_worker` — **before `init_device` and
before any `torch.compile`** — so the variable is set before the first HPU op *by construction*. It is
env-gated on `DSV4_FP4_LUT=1` and non-fatal (imported by path, wrapped so it cannot break plugin
loading). With the promotion in effect the load path now **writes nothing** — a write there would be
unread *and* would undo the promotion. The detector is closed the same way: `_kernel_libs_in_maps()`
**includes** the image default, and `kernel_db_fixed()` is the honest "an HPU op has already run" test
the load path now consults instead of the old blind filter — and an install that *is* late still
happens (a later-spawned worker inherits it) but is now **loud**, recording `db_already_fixed=True`
and naming the library actually in use.

**The evidence is the right shape — a reproduction and then a fix, not a fix alone.** In the passing
run the loaded library path is **read from `/proc/self/maps` rather than inferred**
(`libfp4_dequant_lut_kernels.so` is mapped after the first HPU op) and **both fp4 ops execute**: the
arithmetic op, whose GUID is not in the image default, and the promoted LUT op, whose GUID exists only
in the promoted library — the latter returning `0x00` at the `k = −9` floor cell, i.e. the floor-fixed
decode. The install itself is **unchanged**: `libfp4_dequant_lut_kernels.so`, md5
`abf368253d1433c3a79134863850d901`, 87,704 B — only *when* it is installed changed. **MEASURED /
source read.**

> **The limit, stated rather than glossed: there was no end-to-end engine decode.** The ordering is
> argued from the call site — `load_general_plugins()` at `WorkerWrapperBase.init_worker`, before
> `init_device` — **not measured in an engine run**. What is measured is one process on one module:
> the plugin hook, the load path, and both ops. **SUPERSEDED 2026-10-14 — a retry of the same launch
> completed the end-to-end engine decode; this paragraph is retained as the record of what was true
> before it ran (the retry note is below).**
>
> **Narrowed — the ordering is now measured in a real serving process.** In the `vllm serve` launch
> item 26 records, the hook fired pre-device in **all six** processes, before any HPU op — but still
> **without an end-to-end engine decode**: that run died at boot (§3.10 item 26;
> `Archive/by-date/2026-10-13/vllm-serve-smoke/REPORT.md` §2). **SUPERSEDED 2026-10-14 — a retry of the
> same launch completed the end-to-end engine decode.** *The clause above is retained as the record of
> what was true before the retry ran.* On the retry the server **booted**, ran all **90 prompt + 9
> decode** warm-up buckets, and **answered HTTP coherently** — so an end-to-end engine decode is now
> **MEASURED** in the **`VLLM_T_COMPILE_DYNAMIC_SHAPES=0`** configuration: the measured offline
> configuration, and a **configuration choice, not a fix**, so what is established is that a decode
> completes **in that configuration**, not that the product defaults work. The retry's other serving
> facts, and the serving path's own non-reproducibility, are §3.10 items 26 and 27
> (`Archive/by-date/2026-10-13/vllm-serve-smoke-retry/REPORT.md` §1–§4).

> **The correction, stated plainly.** **The blocker was the late write, not the GUID gap — and the late
> write is now fixed (above).** The GUID gap is a **separate, one-line fix**, and it only matters at
> **`DSV4_COMPRESSOR_OP=1`**: the single-GUID arm libraries bind one GUID where the shipped library
> binds two, and the wiring's own warning is conditional on exactly that variable
> (`if os.environ.get("DSV4_COMPRESSOR_OP", "0") == "1":` → *"the arm libraries expose ONE GUID … The
> E3a node will fail to instantiate"*). At the default `DSV4_COMPRESSOR_OP=0` it is **inert**. The
> promoted library does not have the gap at all. **MEASURED.**

**The fix that dissolves the *variant* half: promotion.** Instead of choosing a library at run time,
put the kernel's *name* into the **active** library's `GetKernelGuids` table — the *variant* then
rides the 240-byte table (below) rather than which library is mapped, so the op is callable from the
first compile. The promoted library is a **superset** of the shipped one, so it can be
`GC_KERNEL_PATH` for a whole run:

| library | md5 | size | GUIDs for `DEVICE_ID_GAUDI2` |
|---|---|---:|---|
| `libfp4_dequant_kernels.so` (shipped; the promotion never writes it) | `25c54d818a4c60e222720adbc155e741` | 62,624 B | 2 — `fp4_dequant_fwd_gaudi2`, `dsv4_comp_ident_gaudi2` |
| `libfp4_dequant_lut_kernels.so` (promoted) | `abf368253d1433c3a79134863850d901` | 87,704 B | **3** — the two above + `fp4_dequant_lutptr_rc_gaudi2` |

plus a **new op registered in the EXISTING `custom_op` namespace** (`custom_op::fp4_dequant_lut`) —
which is also exactly what §2.4's engineering recipe requires, and for a measured reason: the backend
short-circuits on `node.target.namespace == "custom_op"` and clusters the node regardless of what the
shared-layer validator would say, so a defensive new namespace would **cost the co-scheduling**.

**`GetKernelGuids` is per-`DeviceId`, and that is what makes the table a gate.** It is read by calling
the library's own exported function in the same two-call shape the graph compiler uses — query the
count, then fill:

| `DeviceId` | label | GUIDs | ours |
|---|---:|---:|---|
| 1 | GAUDI | 2,021 | `[]` |
| **3** | **GAUDI2** | **3,253** | `['fp4_dequant_fwd_gaudi2', 'dsv4_comp_ident_gaudi2', 'fp4_dequant_lutptr_rc_gaudi2']` |
| 4 | GAUDI3 | 3,358 | `[]` |

All three of ours are present for GAUDI2 and **only** for GAUDI2 — the library's `is_ours()` is
`deviceId == DEVICE_ID_GAUDI2`. **MEASURED.** *(A docstring in the wiring says **3252**; the measured
value is **3253**. The comment is stale and was left alone.)*

**Selection is active three independent ways.** *(a)* The graph compiler's own instantiation
handshake fires for the promoted name — `InstantiateTpcKernel LUT name='fp4_dequant_lutptr_rc_gaudi2'`
→ `18` ("not found, build it") then → `0` (success). *(b)* The load path's own gate reads the
`GetKernelGuids` table rather than trusting the build: `_install_fp4_lut('noguard_floor') -> True`.
*(c)* **The activity falsifier**: an all-zero 240-byte table changes **65,536 of 65,536** block
outputs, so the table flows into the compiled graph as **runtime data** and is not baked in at trace
time. The timing corroborates it — the two *branch* compiles cost **30.17 s and 29.98 s**, while two
new table *values* cost **0.02 s each** because they reuse the compiled graph. **MEASURED.**

**Which variant this checkpoint selects — the mechanism, and the half that is operational.** The
selection runs on the load path: `_prep_compress_caches` → `_check_v4_routed_all` →
`_select_fp4_variant_all`, which calls `decide_and_apply_from_bounds(hi, lo)` and then
`_install_fp4_lut(out["decision"].variant)`. On this checkpoint the
routed-expert scale exponent bytes are `[118, 126]`, so `top_guard_needed=False` (the upper guard's
cheapest firing needs byte 133) and `floor_fix_needed=True` with **zero margin** (byte 118 *is* the
boundary) → **`noguard_floor`**. Both predicates are tight, and the decision logic is **correct, not
defective**. **source read.** *(The single byte-127 element is in the `mtp.` head, which is not
instantiated.)*

**`guarded` means "reproduces the shipped kernel" — and the shipped kernel is the arm with the
defect. "Bit-exact" does not mean "correct".** The shipped decode says so in its own header: below
`t = -3` the two terms of the subnormal branch are each clamped where their *sum* should be rounded,
so the magnitude **FLOORS** to `2⁻⁹`, and at the `t = -3` **exact tie** (true value `2⁻¹⁰`) the code
resolves *away from zero* where RNE — and the HPU's own `cast_to_fp8_v2`, measured — resolve it to `0`
(`src/custom_ops/fp4_dequant/kernels/gaudi2/fp4_dequant_fwd_gaudi2.c`). So the two arms are:

| variant | its 240-byte table | what it is |
|---|---|---|
| `guarded` | md5 `c713d706…` | **reproduces the shipped kernel's `k = −9`, `t = −3` behaviour** (the `±2⁻⁹` floor) |
| `noguard_floor` | md5 `5a1e3296…` | **correctly rounded (RNE)** — `±0` at those cells |

They differ on exactly **2 of 240** cells, both at `k = −9` (nibbles `0x1`/`0x9`), and the
`5a1e3296…` md5 is the one the engine actually installs. **`guard` and `noguard` carry the *same* 240
bytes** — both md5 `c713d706…`, because the guard branch fires only at `t ≥ 15`, outside the table's
`k ∈ [−9, 5]` domain — so the selector's guard half cannot hurt you. The
**`954 of 65,536` difference between the arms is measured against the shipped kernel, and it is a
correctness improvement, not a regression** — it is the floor fix §3.4.5 specifies, and it moves the
decode *toward* the correctly-rounded reference. **MEASURED** for the tables and the 2 cells;
**source read** for the header.

**No environment variable makes the *selector* install `guarded` — and on correctness grounds that is
the right outcome, not a workaround.** (The one way to install it is the env-gated test seam
`DSV4_FP4_LUT_TABLE_VARIANT=guard` — which is **default off**: with it
unset, production behaviour is byte-identical, and it is how the block-test arms above are produced.)
`DSV4_FP4_VARIANT=guard` is consumed by `apply_selection` for the **library**
choice, while the **table** comes from `out["decision"].variant`, which that variable does not
affect — and since the shipped library **is** the `guard` variant, naming it would additionally
repoint `GC_KERNEL_PATH` at the shipped library, which carries **no LUT kernel**, killing the
promotion. The variant the engine selects is the **correctly-rounded** one. **source read.**

**Whether the tie cells are reached is MEASURED — twice, by two independent instruments — and the
answer is yes.** *Established from the checkpoint's own packed bytes:* **1,070** byte-118 groups, of
which **316** carry at least one tie nibble — **1,399** tie nibbles in **23** (layer, expert, tag) pairs,
every one of them `w1`, in layers 34/35/37/39/40 — and at TP=4 (`n_local = 64`, `g = min(n_local, T·K)`) a chunk with **T ≥ 11**
gathers **every** local expert, with all four ranks holding tie-carrying experts (**387 / 562 / 131 /
319** nibbles). The earlier TP=1 test had to **force** the 26 byte-118 experts in; a TP=4 prefill does
not. **But the gather argument alone proves nothing**: `out = (y * gate_wg).sum(0)` and `gate_w` is
exactly `0` for a gathered-but-unrouted expert, so only *routing* reaches. *(a) Routing, from the
model's own dump.* A `HPU_DUMP=1` arm saves the model's own `topk_ids` per layer per rank; reducing it
gives **132,096 routed (token, layer, slot) events** over one prefill forward (T = 128 padded tokens,
43 layers, 6 slots per token per layer), of which **24 are on a tie-carrying expert** — rank1 **16**,
rank2 **4**, rank3 **4**, rank0 **0**. The experts actually hit are layer 35 expert 101 (×14) and layer
39 experts 112 (×1), 81 (×1), 166 (×4), 213 (×4). *(b) Influence.* The `tieprobe` table is `guarded`
except at **exactly the 2 `k = −9` tie cells**, with the magnitude raised from `2⁻⁹` to `240` (≈2¹⁷×);
it moves **all 24 positions** (2746/2746 shared logprob cells, 3/24 token ids,
`max|d| = 1.283638e+01`), starting at position 0. ⇒ The cells are **reached and load-bearing**.
**One new scope statement from the same scan:** it covered all three tags `{w1, w3, w2}`, and **every
one of the 23 tie pairs is in `w1`** — no tie nibble lives in `w3` or `w2`, so the tie enters the model
only through the `w1` (gate) GEMM. **MEASURED.**

**The `guard_floor` mis-mapping — FIXED, with zero effect on this checkpoint.** `VARIANT_DECODE`
(`lut_tables.py`) used to map `"guard_floor"` onto `guarded`, **silently dropping the floor fix** for
a variant whose own predicate says the fix is needed — and **no guard+floor decode existed at all**,
contradicting the design's own rule that the floor fix must be present in **both** `_floor` arms. It
now maps to a real guard+floor decode (a shared `_floor_fixed()` backs both floor-fixing decodes, so
the fix cannot be present in one and absent from the other), whose table md5 is `5a1e3296…` with
**both** `k = −9` cells fixed. **The selected table is unchanged**: this checkpoint still selects
`noguard_floor`, md5 `5a1e3296fa3a6ecf3379b688a18e1540`, so nothing the engine runs today can move.
**source read / MEASURED.**

**The structural answer, and it outranks the fix: the `guard`/`noguard` distinction is meaningless in
the LUT path.** The LUT kernel clamps `k` into `[−9, 5]` (`LUT_K_LO`/`LUT_K_HI` in
`fp4_dequant_lutptr_rc_gaudi2.c`), so `t = exp + k ≤ 14 < 15` and the guard — which fires at
`t ≥ 15` (i.e. `k ≥ 6` at the top code) — **can never fire**. `guard` and `noguard` therefore build
**byte-identical** tables (`c713d706…`). ⇒ For any checkpoint that *needs* the upper guard, the LUT
path is **wrong by construction**, not merely unselected — a `k ≥ 6` group is decoded with the
`k = 5` row. The distinction stays meaningful **only for the arithmetic kernel**, which has no such
clamp and whose decode still carries the guard (`dec_guarded(0x06, 6) = 0x77` vs
`dec_pre_guard(0x06, 6) = 0x78`). ⇒ `guard_floor` was **both mis-mapped *and* unreachable on
the sound path**: any `guard*` variant needs `max_byte > 132`, i.e. exactly `not admitted`, and the
fresh-load upper-bound assertion raises *before* the selection runs. So the defect had **no numerical
consequence in the LUT path in any reachable case today** — but it is a **live trap if the clamp is
widened or a `k ≥ 6` checkpoint is admitted**, which is why it was fixed now rather than when it
bites. **source read.**

**FINDING, not a fix: the range-guard admission check is unsound on the snapshot-restore path.** This
is a **latent** path, not a demonstrated failure — **no checkpoint with such scales has been run
here** — and the code already concedes the gap in prose. The upper-bound assertion lives inside
`_register_v4_routed_packed`, which **returns early when the routed buffers are already filled** —
exactly the snapshot-restore case — while the **selection still runs on restore** (`_check_v4_routed_all`
→ `_select_fp4_variant_all(where="snapshot restore")`) and **takes the bounds itself**
(`_v4_fp4_scale_bounds`, a plain min/max with **no guard**). The lower-bound guard is skipped the same
way: `_v4_fp4_scale_lower_bound` is called only inside `_register_v4_routed_packed`. ⇒ A **restored**
checkpoint with `max_byte > 132` or `min_byte < 118` **selects what the fresh-load path refuses** —
and with `DSV4_FP4_LUT=1` the `[−9, 5]` clamp then yields a **silently wrong finite value** (the
`k = 5` row) instead of the guard's saturation. The code's own comment on the restore path says a
snapshot restore never ran the load-time selection *"nor the two range guards — pre-existing,
deliberately unchanged here"*, so this is a **known-but-unfixed hole, not a discovery**; what is
recorded here is that it is a finding with a stated precondition — a **restored** checkpoint whose
routed-expert scale bytes leave `[118, 132]`. **source read.**

**Block-level safety — and it is a BLOCK, not a model.** `_compute_v4_routed` (the model's own MoE
function) compiled with `torch.compile(fullgraph=True)`, on **real layer-39 checkpoint weights** (all
256 experts), at `H=4096, inter=2048, T=16, topk=6, g=96`, with all **26** byte-118 experts forced
into the gathered set. The output block is `(16, 4096)` bf16 = **65,536** elements:

| arm | table | op taken | vs shipped | differing | `max\|d\|` | `max_rel_vs_maxA` | `max_rel_on_differing` |
|---|---|---|---|---:|---:|---:|---:|
| `A_shipped` | none | `fp4_dequant_fwd_gaudi2` | — | — | — | — | — |
| `B_lut_guarded` | `guarded` | `_FP4_LUT_OP` | **bit-exact** | **0 / 65,536** | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `C_lut_noguard_floor` | `noguard_floor` | `_FP4_LUT_OP` | divergent | **954 / 65,536** | **2.441e−04** | **1.553e−03** | **2.209e+01** |
| `D_lut_all_zeros` | 240 zero bytes | `_FP4_LUT_OP` | falsifier | 65,536 / 65,536 | 1.572e−01 | 1.000e+00 | 1.000e+00 |

**Quote both relative columns.** `max_rel_vs_maxA = 1.553e−03` is the flattering one — its denominator
is `max|A| = 1.572e−01`, a large output. `max_rel_on_differing = 2.209e+01` is the relative error
**on the differing elements only**, where the output is near zero. The divergence is bounded in
**magnitude** (`max|d| = 2.441e−04`, `rms|d| = 6.669e−06`) and **unbounded in relative terms**. **The
flattering column must not stand alone.** **MEASURED.** At the weight level, straight through the two
ops on the 26 experts' real `w1` (**218,103,808** bytes): `guarded` **0 differ**,
`noguard_floor` **1084 differ** — exactly the **1084** engaging tie weights in those experts' **842**
byte-118 groups. **MEASURED.**

**The divergence is confined to the `k = −9` floor fix — and it is the fix §3.4.5 asks for.** The two
tables differ on **2 of 240** cells, both at `k = −9`, nibble codes `0x1` and `0x9` (the `t = −3`
tie): the `guarded` table emits `0x01`/`0x81` (the shipped ±2⁻⁹) where `noguard_floor` emits
`0x00`/`0x80` (±0). That is precisely the **subnormal-floor correction** §3.4.5 specifies as *not
fixed*, and **the variant this checkpoint's scale bounds select is `noguard_floor`**. **MEASURED.**
*Consequence for anyone promoting a LUT: **a promoted LUT needs a table per variant.** The 240-cell
table **is** the variant's specification, so two variants cannot share one table — and the table is a
runtime op input, so nothing about it is checked at build time.*

**The cost, and its scope — read this next to the safety result.** The flagged `lutptr` arm with
`-mllvm --pre-RA-sched=vliw-td` at `[4096,2048]` measures **17.794 µs / 1885.7 GB/s** against the
shipped kernel's **68.553 µs / 489.5 GB/s** — a composed **3.853×** — landing at **77.6 %** of the
box's measured **2.431 TB/s** peak (`n = 60`, `warm = 5`, 3/3 reps identical to ≤0.06 µs, one process,
arms interleaved, and the three control arms reproducing the archived module-0 numbers to 0.13 %).
The flag moves the **bundle** count (**38.25 → 20.75** per iteration) and leaves the **vector-op**
count (**14.2 → 14.5**) and the instruction count (**24.75 → 25.75** non-NOP) where they were —
**scheduling headroom, not arithmetic headroom** (§4.2.7's lever, applied to the LUT arm). **MEASURED.**
**The 3.853× is KERNEL-LOCAL — one kernel body at one shape `[4096,2048]` — and it is NOT a
model-level speedup.** The whole-model counterpart is measured below and is **1.1337× on the decode
step** and **1.358× on the prefill step**; the dequant is a small share of a 43-layer step, so the
3.853× does not survive to the model. Both numbers are real; they are **different quantities**, and
the 3.853× must never be read as "the model got 3.853× faster".

> The **3.853×** was timed on the **`lutptr`** arm and the bit-exactness on the **`guarded`** arm, but
> a static census of both listings shows them to be **the same kernel** — identical in every column
> (**20.75 bundles/iter, 14.5 vector ops, hr 1.431** — the same census; the `3616 B` of `.text` is
> measured on the timed listing only, and identical counts imply identical length) — because the
> 240-byte table is a **runtime op input** and cannot change the schedule. Read what "bit-exact" means
> here: the `guarded` arm is bit-exact **to the shipped kernel — which is the arm that reproduces the
> `k = −9` tie defect** (§3.4.8); the correctly-rounded arm is **`noguard_floor`**, and that is the one
> the engine installs. **What remains untimed *at the kernel level* is the `noguard_floor` table
> itself** — a **0.02 s** swap on the same compiled graph, not a safety margin; end to end it **is**
> timed below (`noguard_floor` 0.0817 s vs `shipped` 0.0927 s). On the **timed** distribution every
> arm is bit-exact against the shipped kernel, and on the exhaustive domain `lutptr` differs from it
> by **384 of 8,192** cells — the pre-existing pre-guard difference, **not** a `vliw-td` effect (the
> diff counts are identical for both schedules).

**Whole-model decode at TP=4 — the run that closes §3.9 item 21, and it is bit-exact.** *Five arms,
one engine process each, **four modules held simultaneously** (`HABANA_VISIBLE_MODULES = 0,3,6,7` —
**not** one of the vendor's recommended 4-Gaudi combinations; §3.7.2), the
W0-anchor config (TP=4, EP, `max_model_len` 2048, fp8 quant, bf16 KV), a **cold per-arm recipe cache**,
and `hl-smi` confirmed at the **768 MiB baseline with no compute processes** immediately before every
launch.* One fixed prompt of 35 real tokens padded to the model runner's `('prompt', 1, 128, 0)`
bucket, greedy, 24 decode steps, twice per arm, `logprobs=128`; decode-step medians over the **46
pooled steady steps** (2 × 23). **MEASURED.**

| arm | table md5 | token ids vs `shipped` | logprob cells | `max\|d\|` | decode-step median | prefill step |
|---|---|---|---:|---:|---:|---:|
| `shipped` (baseline) | — | — | — | — | 0.0927 s | 0.6659 s |
| `guarded` | `c713d706…` | **24/24 identical** | **0/3072** | **0.000000e+00** | 0.0818 s | 0.4903 s |
| `noguard_floor` (**the default**) | `5a1e3296…` | **24/24 identical** | **0/3072** | **0.000000e+00** | 0.0817 s | 0.4925 s |
| `zeros` (activity falsifier) | `cea67ffa…` | 2/24 | 740/740 | 1.327644e+01 | 0.0822 s | 0.4904 s |
| `tieprobe` (sensitivity probe) | `e34cdee7…` | 21/24 | 2746/2746 | 1.283638e+01 | 0.0822 s | 0.4957 s |

`max|A| = 1.806262e+01`, and for **both** bit-exact arms `max_rel_vs_maxA = 0` and
`max_rel_on_differing = 0`. Same-process control in every arm: `meas1` vs `meas2` = 24/24 token ids and
0/3072 cells, `max|d| = 0`.

**The end-to-end numbers, with their scopes — and they are 1.134× / 1.358×, not 3.853×.** The
whole-model decode-step speedup is **1.1337×** (`guarded`) / **1.1347×** (`noguard_floor`); those
multiples come from the **unrounded pooled medians** (`0.092696` / `0.081765` / `0.081692 s` over the
46 pooled steps), not from the 4-dp medians printed above (`0.0927 / 0.0818 = 1.1333`). The prefill
step is **1.358×** / **1.352×** — the **measured-request-#1** ratios; the second measurement gives
**1.3538×** / **1.3634×**. The **3.853×** (above) is a **kernel-local** ratio on the dequant
kernel alone at `[4096,2048]`; the dequant is a small share of a 43-layer step, so the two figures are
**different quantities** and the 3.853× must not be read as a model speedup. Nor is this a benchmark:
one prompt, `max_num_seqs = 1`, so these are end-to-end **step walls**, not throughput, and the prefill
figure has **n = 2**. **MEASURED.**

**Both real tables are bit-exact to the shipped path at model level — so the default is not merely
"safe enough", it is exact.** `guarded` and `noguard_floor` each reproduce `shipped` on 24/24 token ids
and **0 of 3072** logprob cells with `max|d| = 0`. The tie cells are **reached and load-bearing** (24
routed tie events in one prefill forward; the amplified `tieprobe` table moves 24/24 positions), so the
honest statement is: **the `2⁻⁹ → 0` floor fix is a real, reached correction whose effect is below the
model's bf16 output resolution** — *not* that the cells were never hit, and *not* a safety margin.
**MEASURED.**

**Selection and activity, counted from the artifact.** The load path printed
`[v4-fp4-lut] LUT path ACTIVE: variant=… md5=… kernel=fp4_dequant_lutptr_rc_gaudi2` **four times per
LUT arm** (one per worker) with that arm's expected md5; the graph compiler's own handshake
`InstantiateTpcKernel LUT name='fp4_dequant_lutptr_rc_gaudi2'` fired **192×, 96 of them `-> 0`**; and
`InstantiateTpcKernel OURS` fired **0×** (the shipped `fp4_dequant_fwd_gaudi2` was not instantiated in
the LUT arms, which is what the LUT branch replacing it predicts). *Both counters come from the
`[fp4_wrapper]` diagnostic, which is emitted **only** under `FP4_WRAPPER_DEBUG=1` — so a zero from an
arm that did not set it is a claim about the instrument, not about the kernel (§3.10 item 23).* The
`zeros` falsifier moves **22/24 tokens and 740/740 cells**, which is what makes "selected" mean "and
doing something". **MEASURED.**

**On the restored scales the selector picks `noguard_floor` on every rank, in every arm.** The scales
are byte `[118, 126]` on ranks 0/2 and `[118, 125]` on ranks 1/3, so `top_guard_needed=False` and
`floor_fix_needed=True` (zero margin on the boundary byte 118) ⇒ **all four ranks print
`=> VARIANT: NOGUARD_FLOOR`**, in the `shipped` arm as well. **MEASURED.**

**The test seam used to produce these arms is default-off, provably inert, and not the shipping
mechanism.** `_install_fp4_lut` in `src/vllm-gaudi/vllm_gaudi/models/deepseek_v4.py` is the only `src/`
file touched; with **neither** env var set, the `else` branch executes the pre-edit line unchanged, and the
`noguard_floor` arm ran with `DSV4_FP4_LUT_TABLE_VARIANT` **unset** and installed exactly
`5a1e3296fa3a6ecf3379b688a18e1540` — the same md5 the builder produced before the edit. **If a
different variant should ship, that is a change to the decision logic, and the seam must not be how it
is selected.** **MEASURED / source read.** *(No line numbers for that file: it is under active edit and
this seam is uncommitted, so the symbol — not a line — is the stable anchor. The archived report that
first recorded the seam keeps its own line numbers: an archive is a record of what was true when it was
written; this manual is the living document.)*

*Artifact: `Archive/by-date/2026-10-13/fp4-lut-promotion-safety/REPORT.md` and its logs
(`logs_device_probe.txt:4-14`, `:50-61`, `:93-97`, `:102-109`, `:138-150`;
`logs_lut_model_safety.txt:46-55`, `:62-66`, `:71-73`; `logs_lut_tables.txt:4-6`;
`logs_variant_gate.txt:76-78`, `:87`, `:112`, `:122`). The variant semantics, the tie census and the
latent defects: `Archive/by-date/2026-10-13/variant-decision-semantics/REPORT.md` §1–§4. The 3.853×
timing and its bundle/vector-op census: `Archive/by-date/2026-10-13/fp4-lutptr-vliw/REPORT.md` §1–§2.
The late-write evidence: the archived `fp4-e2e-variant` report's `[latewrite] wiring verdict = applied`
and `THE FP4 OP DID NOT RUN: … synStatus=synStatus 26` probe output. **The whole-model TP=4 arms, the
routing measurement and the selection evidence:
`Archive/by-date/2026-10-13/fp4-tp4-model-safety/REPORT.md` — its VERDICT (the five-arm table), §2 (the
end-to-end numbers), §4 (routing and influence), §5 (the selection/activity counts), §6 (the seam) and
§7 (device hygiene); the reductions are `compare.py`, `tp4_decode_compare.py`, `tie_scan.py`,
`reduce_routing.py` and the JSONs `shipped.json`, `guarded.json`, `noguard_floor.json`, `zeros.json`,
`tieprobe.json`, `tie_experts.json`, `routing.json` in the same bundle.***

*The two code fixes and the restore-path finding:
`Archive/by-date/2026-10-13/fp4-code-fixes/REPORT.md` — §1.1 the late-write reproduction on the
unmodified tree, §1.2 the restructure onto the `vllm.general_plugins` hook, §1.3 the acceptance run
(the loaded path read from `/proc/self/maps`, both ops executing), §1.4 the pre-fix-vs-fixed
blind-spot pair, §1.5 the ordering guard, §2.1–§2.4 the `guard_floor` mapping and the structural
answer, §2.5 the restore-path finding. Logs: `accept_before_fix.txt`, `accept_after_fix.txt`,
`blindspot_before_after.txt`, `late_promotion.txt`, `variant_mapping_before.txt`,
`variant_mapping_after.txt`, `lut_tables_after.txt`, `gate_variant_selection_after.txt`, with
`MANIFEST.md5` over the bundle's 44 files and the pre-fix module copies in its `backup/`. Every
pre-fix line number in that report is the archive's own record; the manual cites the symbols.*

---

## 3.5 Specialization and graph count — what it means for a deep model

**This section is deliberately short, and it is not where the mechanism lives.**

- The **mechanism** — no dynamic shapes, integer specialization, and the graph-count arithmetic — is a
  **basics** fact that bites someone optimising a 20-line region. It is in **Chapter 1 §1.4**
  (§1.4.1–§1.4.2 for the two rules, §1.4.3 for the count, §1.4.4 for where it bites).
- The **region strategy** — graph breaks cost twice, prefer fewer and larger regions, and the
  coalescing trap (coalescing does *not* reduce the count if the region still guards on the same
  parameters) — is in **Chapter 2 §2.4** and **Chapter 1 §1.4.3**.

Those two are cross-referenced here rather than restated. What is **porting-specific** is the
*consequence of that arithmetic for a model with many layers*, and that is all this section covers.

### 3.5.1 Why a deep model lands at O(n_layers) compiled programs

*(Mechanism: **Chapter 1 §1.4.1–§1.4.2**. The graph-count arithmetic: **Chapter 1 §1.4.3**. Region
strategy and the coalescing trap: **Chapter 2 §2.4** and **Chapter 1 §1.4.3**.)*

- **Regional compilation compiles each decoder layer as its own compiled function.** So the number of
  compiled programs is **not** "one per region kind" — it is **per layer**.
- **Per-layer constants are computed *inside* the graph.** A layer index, a per-layer byte offset, a
  per-layer buffer slice: any of these is an integer in traced code, and therefore a specialization
  key (Chapter 1 §1.4.2).
- **⇒ The compiled-program count is `O(n_layers × region-kinds × buckets)`**, not
  `O(region-kinds × buckets)`. For a 43-layer model that is a factor of **43** on the layer axis alone,
  before any region-kind or bucket multiplier.

**This is structural, not a bug.** It follows directly from the compiler's static-shape model applied
to a model that genuinely has 43 *distinct* layers. **Do not treat a high count as a defect to
chase** — treat it as the price of the design, and decide deliberately whether any part of it is worth
removing (Chapter 2 carries the tradeoff, and it is not one-sided).

### 3.5.2 The worked example — and a warning about its numbers [DSv4F]

**An illustration of the arithmetic above, not a law.** A 43-layer model on our port has been reported
at **~194 distinct compiled recipes per decode step, of which 160 were used by exactly one layer** —
i.e. almost all of the specialization is per-layer, and the instance count scales with `n_layers`
rather than with the number of region kinds.

> **⚠ Those two numbers are NOT current, and a port should not quote them as a measurement.**
>
> - **Source (so you can check it):** the figure is recorded in the project's
>   `notes/specialization-and-compile.md` §B1, and in its archived twin
>   `Archive/by-date/2026-10-04/old-notes/intentional-specialization-params.md`. **It comes from a
>   single `with_stack=True` trace taken 2026-09-20**, on a code state ~3× slower than the current one.
> - **The project's own corrections note says so.** `notes/wall-bound.md` §9 lists it explicitly as a
>   stale figure, and gives the current values: **123 distinct recipe handles per 4 steps** for one
>   code family and **70–73** for another; and separately **246 compile ids over 25 regions, with a
>   per-region maximum of 43** — that per-region 43 is the one number here that directly supports the
>   per-layer claim, because 43 *is* `n_layers`.
> - **A later archived census** (`Archive/by-date/2026-10-11/node_law/out/*.json`) counts **83–130
>   distinct recipes per 4-step trace** across that archive's runs — again well below 194.
> - **The units are also not the same object.** "Recipe", "compile id" and "recipe handle" are three
>   different things and are counted on different clips (see **Chapter 2 §2.1** for the four nouns,
>   and **Chapter 2 §2.5.4** for how the three counts differ). A count quoted without its unit and
>   its clip is not a count.

**What survives, and what is the point of the example:** the *scaling* — the count is per-layer, and
most instances serve exactly one layer. That is what a port of a deep model must budget for. **The
exact number is code-state-, unit- and clip-dependent, so measure your own** rather than importing
ours.

### 3.5.3 The practical consequences for a port of a deep model

- **Budget device memory for the compiled programs, not only for weights and KV.** Each compiled
  program holds its own graph buffers. On our port the retained post-warmup footprint was **~11 GiB**
  above the post-build figure, with a transient of **~30 GiB** during the first decode's compilation
  — both **[DSv4F]**. The derived **~55 MB per compiled program** is an **INFERRED hypothesis**, not a
  measurement: if the footprint does *not* fall when the instance count falls, the memory is
  somewhere else and the arithmetic is void.
- **Budget warmup time and cache size.** The count is large enough to matter for both, and a cache
  limit that is exceeded causes recompile thrash.
- **Watch for compilation during steady state.** If the count grows while the workload runs, then some
  shape or some integer is **not covered by warmup** — and **any timing measured over that window is
  invalid.** This is a hard gate in the porting checklist (§3.10), and
  [`examples/03-porting/specialization_probe.py`](examples/03-porting/specialization_probe.py)
  measures it at both compile levels.
- **The per-layer structure is where the count comes from, so that is where to look** if you want to
  reduce it — and Chapter 2 explains why reducing it is not automatically a win.
- **A deep model is not a special case of a shallow one.** A 2-layer test model will not show this
  behaviour at all. If you prototype on a truncated model, you will not see the specialization
  pressure your real port will have.

---

## 3.6 What actually bounds a decode step

### 3.6.1 The two-term relation

> **`wall = 21.724 + 0.996812 · merged`, R² = 0.999931** (n = 12). **MEASURED**

where **`merged` ≡ `|union(kernel ∪ hpu_op Dma*)|` — device OCCUPANCY, not summed work.**
**MEASURED**

**The wall responds to `merged` and to nothing else measured. MEASURED**

| predictor | r² | sign |
|---|---|---|
| `merged` | **0.9999** | correct |
| `~device ops` | 0.565 | correct but far weaker |
| `~syncs` | 0.195 | **wrong sign** |
| `~launches` | 0.15 | **wrong sign** |
| `~interval count` | 0.10 | **wrong sign** |
| `~recipes` | 0.066 | **wrong sign** |

**The number that makes the point: 925.1 ms of summed device work fits inside a 70.5 ms of occupancy
— 13.1× parallelism. MEASURED [DSv4F]**

> **⇒ Reducing total device work does not reduce the wall. Only the UN-OVERLAPPED part counts.**
> A doctrine that optimises summed kernel time optimises the wrong quantity.

**Corollary, and we had to retract our own claim to learn it:** *"the binding resource is the
op-issue path"* is **refuted** for the current code state — a 25 % kernel-count reduction at a
code state whose device was only ~20 % occupied left the wall **flat** (0.307 → 0.316 s).
**MEASURED**

### 3.6.2 The caveats — and they are the point of this section

We got this wrong once by quoting the relation without its limits. Do not repeat it.

> **⚠ CAVEAT 1 — the slope is NOT validated out-of-sample.** A pure-extrapolation split gives
> **`b` = 1.298 instead of 0.997 — 30 % too steep**, under-predicting the slowest run by **94.8 ms**.
> **`b = 0.996812` is determined by one pair of runs (the E4 pair).** **Quote it with its run set.**
> **MEASURED**
>
> **⚠ CAVEAT 2 — read the evidence as "the kernel channel is REFUTED", not as "the DMA channel is
> CONFIRMED."** The kernel channel's pooled out-of-sample R² is **negative in every split** (−1.63,
> −4.18, −9.96, −0.16) while the alternative's is 0.999926 / 0.999898 / 0.996995 / 0.891593. One
> channel being bad does not make the other one *proven*; it makes it the best available.
> **MEASURED**
>
> **⚠ CAVEAT 3 — there is a second fit of the same phenomenon**: `wall = 21.63 + 0.9929·merged`,
> R² = 0.99994 (n = 8). Two fits, different run sets, slightly different constants. **Say which one
> you are quoting.** **MEASURED**
>
> **⚠ CAVEAT 4 (added 2026-10-12) — an `a + b·X` fit is only as identifiable as the corpus it was
> fitted to.** The relation above is a **cross-sectional** fit across runs, so it is not itself the
> case in point — but the project's own 2-D sweep shows how the *form* fails when it is fitted to a
> **1-D** one: `W = a + b·M` gets slope **3.22** when the swept knob is matmul *size* and **1.54** when
> it is matmul *count* — the same physical quantity, a **2× discrepancy** — and the resulting 2-point
> fit misses an intermediate size by **15 %**. **Read any `a + b·X` here as a description of its run
> set, not as a law.** Before extrapolating one, check that the corpus varied more than one dimension,
> or that the dimension you are extrapolating along is the one that was swept. **MEASURED**
> *(§2.7's 2026-10-12 update; `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §4.2.)*

### 3.6.3 What is *not* established about decode

- **Whether the engine's cross-engine intersections are REAL concurrency.** A TPC∥DMA intersection
  figure of **12.531 ms/step** has moved three times in one day, each move forced by a measurement. It
  is **UNRESOLVED**: measurement spread (0.09–0.89 %) puts the deciding margin **inside the noise**;
  the load-transfer extrapolation is uncertain by **tens of percent**; and the quantity that decides it
  **was never measured** — the deciding arm held the variable fixed, which made its own pre-committed
  falsifier **unevaluable by construction** (a design flaw, not a null result). Two harness-calibrated
  routes both give `O/M ≤ 1` and differ by **33 %** (0.99 vs 0.66) — **that spread IS the honest
  measure of how soft the estimate is.** **MEASURED**

  > **QUANTIFIED 2026-10-12 — the deciding arm was run, and it lands just over the boundary. The status
  > is otherwise unchanged: still neither established nor dismissed.** The quantity the paragraph above
  > says "was never measured" — `p`, the co-residency inflation that turns the measured DMA union `W`
  > into true DMA busy time `M = W/p` — is now **measured as a function of the TPC:DMA ratio**
  > (**19 arms** — 17 ratio points, the three no-co-tenant `sw_t0_*` arms collapsing into one; it is a 2-D sweep over TPC load × DMA bytes, which the archived 1-D sweeps could not do). The
  > engine's own ratio is `43.493/27.562 = 1.578`; the two arms bracketing it give **`p = 2.23–2.28`**
  > (r 1.51 → 2.28, r 1.77 → 2.23), hence **`M(W0) = 12.1–12.4 ms/step`** and **`O/M = 1.02`**
  > (band **1.01–1.04**).
  >
  > **`O/M` band reconciled 2026-10-12 — the source's band was internally inconsistent.** The band is
  > `O·p/W = 12.531/(27.562/p)` over the **two arms that bracket** the engine's ratio, `p = 2.23–2.28`
  > ⇒ `12.531/12.4 = 1.01` … `12.531/12.1 = 1.04` (exact quotients **1.014–1.037**). The source
  > report's §4.3 **table** prints this as **1.01–1.03** (a truncation of 1.037); its **answer box**
  > prints **1.01–1.59**. **1.59 is not a bracket of the engine's operating point:** it is `O/M` at
  > `p = 3.50`, the inflation measured at TPC:DMA ratio **1.96** — the **third**-nearest arm — whereas
  > the engine's ratio 1.578 lies *between* the two nearest arms (1.51 and 1.77). Widening to the
  > three nearest arms (`p = 2.23–3.50`) is what yields 1.59, and it is incoherent with the same
  > box's central `O/M = 1.02`, which is the two-arm figure (the source's own reducer prints
  > `O/M = 1.21` for the three-arm set). **The two-arm band 1.01–1.04 is what the data supports.**
  > *(Verified by re-running the source's reducer, `Archive/by-date/2026-10-12/engine-overlap/eo_analyze.py` §D.)*
  >
  > **The ceiling for the intersection to be real is `p ≤ 2.200`** (`= W_DMA/O = 27.562/12.531`):
  > above it, the intersection would claim more hidden DMA work than exists. The measured value sits
  > **just over** it. **So the intersection consumes ≈100 % of the DMA work that could possibly be
  > hiding under it** — against **≈45 %** (`12.531/27.562 = 45.5 %`) of the measured `hpu_op` DMA
  > union. *(The source's first mention of this fraction reads "~44 %" and its second "45 %"; the
  > quotient is **45.5 %**, so 45 % is the one the arithmetic supports.)*
  >
  > **Read that as a knife-edge, not as a verdict — in either direction.** Both readings survive every
  > measurement made: if the `hpu_op` window is the execution window (`p = 1`), then `M = 27.56` and
  > **45 %** of the DMA work is hidden — unambiguous overlap; if `p ≈ 2.25`, essentially **all** of it
  > is hidden and the intersection is saturated. **But the `O ≤ M` test does not come out the same under
  > the two readings, and this block previously said it did.** *(Corrected 2026-10-12 — the sentence
  > read "Under **both** readings the intersection is **≤** the available DMA work, so it passes the
  > `O ≤ M` test", which contradicts this block's own `O/M = 1.02` and its "sits just over it"
  > above.)* Under the `p = 1` reading the intersection is comfortably inside the available DMA
  > work (`O/M = 12.531/27.562 = 0.455`); at the bracketing `p = 2.23–2.28` it is **just outside** it
  > (`M = 12.089–12.360`, `O/M = 1.014–1.037`). **So the `O ≤ M` test is met by one reading and missed
  > by 1–4 % by the other — a knife-edge in either direction, and neither reading licenses a claim that
  > the intersection is legitimate.** What does **not** survive is the **magnitude**:
  > the DMA busy behind it is uncertain by **2.2×**, so any pricing that treats the engine's DMA union
  > (27.562 ms/step) as device time is wrong by that factor. **A self-consistent iteration**
  > (`p = f(r_true)` with `r_true = 1.578·p`) converges to `p ≈ 2.25`, so the two conventions agree.
  >
  > **What would settle it, stated plainly:** an **engine-side `DUMP_POST_GRAPHS` on one W0 decode
  > step**, which gives `M(W0)` directly and turns `O/M = 1.02` into a measurement instead of an
  > extrapolation. That needs a **vLLM run** — **NOT RUN**, and it is the single highest-value next
  > measurement here. *(Two smaller gaps the sweep leaves: `p` was not measured at a ratio of exactly
  > 1.578 — the nearest arms are 1.51 and 1.77 — and the engine's DMA **byte** counts are not
  > derivable from the archived traces. Source:
  > `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §4.1–§4.3.)*
- **The size of the ~21 ms host constant's sync-wait vs host-compute split.** `synchronizeEvent`
  carries a Synapse **stream** id, not an OS tid, so it cannot be intersected with the runner thread.
  **UNKNOWN** — settling it needs a runtime change (a `record_function` around each blocking sync at
  the point the runtime issues it).
- **Whether the DMA channel choice is right for *your* model.** Our verdict rests on three reasons
  (load-tracking, subset/definitional, and the wall fit); a fourth argument we originally gave
  (seriality/saturation-collapse) was **retracted**, and the supporting sentence in one note **reads
  inverted against its own conclusion**. Re-derive it on your traces. **MEASURED / OPEN**

### 3.6.4 The device-time decomposition, for orientation [DSv4F]

> **⚠ This table is a PRE-OPTIMISATION BASELINE, not a current profile.** It is one decode arm —
> **`lad_traceW0`, rank 0, 4 decode steps, `merged` = 66.1349 ms/step** — taken at a code state
> **before** the staging work of §3.6.5 and §3.6.6 landed. Its own step is **66.13 ms** against
> **10.6 ms** at bucket 1 on a later state of a parallel port of the same model, so it describes
> what was left to fix, not where a tuned step spends its time. **Re-take it before you steer by
> it.** The two families it ranks first by *union* (`DmaMemcpy`, `DmaTranspose`) are also the ones
> whose op counts fall fastest when staging is fused away — **treat the `Dma*` counts as the
> symptom to watch while fusing staging, not as a budget to optimise within.** *(A later
> decomposition of the same model — TPC 88.6 % of the union at bucket 8 — is a different arm on a
> different port state and does not supersede this table's rows; it only shows how far the mix can
> move.)*

**The column is the family's own UNION — its device *occupancy* — and it is NOT a removal price.**
That distinction was missing from this table until 2026-10-13, and the header even called the column
`merged`; it is not `merged` (the whole step's device union, **66.1349 ms/step** — measured on the
**`lad_traceW0`** arm, **rank 0**, 4 decode steps; a different trace of the same model gives a
different `merged`), it is the family's own busy-set union.

| family | ops/step | union ms/step | **leave-one-out (LOO)** | **LOO / union** |
|---|---:|---:|---:|---:|
| `fp4_dequant` | 9,288 | 22.6330 | **12.0862** | **53.4 %** |
| `DmaMemcpy` | 84,090 | 16.9399 | **10.7841** | **63.7 %** |
| `GEMM` | 8,924 | 11.4767 | **7.5155** | **65.5 %** |
| `DmaTranspose` | 50,790 | 7.0449 | **1.6471** | **23.4 %** |
| `DmaBroadcast` | 6,235 | 5.0750 | **1.5357** | **30.3 %** |
| `gather_nd_*_u8` | 3,096 | 4.3820 | **2.0692** | **47.2 %** |
| `fused_wqa_wkv` *(an event-subset cut, not a family)* | — | 0.2505 | **0.0000** | **0 %** |

**Three families are 46 % of `merged` (by the LOO column; 77 % by union).** **MEASURED [DSv4F]**. Note
the shape of this table: the two largest *op counts* are not the two largest *time* items, and the
family with the largest op count is not in the top three by time. **Count and cost are different
questions.**

#### The node-count census is not a cost model — and there is a family this table cannot show

Two facts hold at once, and a reader needs both:

- **Metadata nodes are numerous and free.** `Reshape`, `StaticReshape`, `logical_slice_fwd`,
  `ExpandDims`, `Split` and `Concatenate` appear in the post-graph dumps and are **absent from the
  device trace entirely**. A census reporting "74.6 % of node executions are metadata" is therefore a
  *graph-node* count with **no device-time implication**, and pricing a fusion from node counts
  overstates the win.
- **Many tiny integer nodes are numerous and expensive.** On a parallel port's **bs = 2** arm
  (81 recipes, 14.30 ms attributed), one family of integer layout arithmetic — `ge_i32` 609 µs,
  `div_mod_i32` 599, `where_i32` 542, `cast_i64_to_i32` 493, `less_i32` 446, `and_i8` 431,
  `constant_i64` 306, `add_i32` 213, `gather_elements` 114 — totals **4,356 µs, 30.5 % of the
  attributed step**: the **largest single term** on that arm. Each node is **1–3 µs**, so the family
  is invisible in any per-node ranking and in TPC lane-time, yet together it dominates. In one
  2-layer attention recipe, 78 integer TPC nodes cost ~124 µs of 182 µs per execution.

**They are distinguishable only by engine identity in the trace, not by op name.** That is why the
table above, which ranks node-name families over a *different* arm (W0, bucket 8), has no such row:
its families are named kernels, and a 1–3 µs node never reaches a ranking built from them. **Do not
add this family to the table above** — it is another arm, and it would mix two populations.

**⇒ The category this names, for the regime table.** "Many tiny integer nodes" is neither bandwidth-,
compute- nor launch-bound in the usual sense — it is **per-node cost paid ~78 times per recipe**. A
reader following a launch-bound diagnosis concludes "fewer, larger ops", which is right but does not
say *which work to delete*. The audit that does: rank nodes by **engine identity × count × mean
duration**, not by count alone, and look for a large count of sub-5 µs TPC nodes.
**MEASURED** (a parallel port's bs = 2 arm, TP4×PP1 — their port state, not ours).

> **The quantity the families add up to: `merged` is a SUM of per-recipe unions, and that is exact.**
> Because recipes execute sequentially — **cross-recipe overlap is exactly zero** — the **110
> per-recipe busy sets partition `merged`**: `25.295` (attention) + `40.642` (MoE/FFN) + `0.198`
> (head/embed/sampler) = **66.135** against `merged` = **66.1399**, a **0.007 %** difference.
> **MEASURED.**
>
> So the arithmetic that is *illegal* is summing **event** durations (§2.9 rule 1) — not summing
> recipe unions, which is the correct model and is what makes this section's tables additive. Two
> consequences worth carrying:
>
> - **The partition is the evidence for the seriality**, and the seriality is the reason the sum is
>   legal. Neither statement stands alone; quote them together.
> - **The sum is exact only when every device event belongs to a recipe.** Eager-placed nodes have no
>   recipe and sit outside the partition — on a Qwen decode trace the recipe-less events were **11 %**
>   of the device union. Where the partition does not close, the shortfall is the eager work, and it
>   belongs in its own row rather than folded into a family.
> - **Per device, never across devices**: in TP the ranks run in lockstep, so per-rank sums are
>   replicas; the step wall is the slowest rank plus the host term.

> **⚠ Read the LOO column, not the union column, when you are pricing a change.**
>
> **A family's union is an *occupancy*; the most a change that removes that family can free is its
> *leave-one-out* — how much `merged` actually falls when the family's intervals vanish.** The LOO
> is **below** the union, and the reason is arithmetic: families overlap. On this trace the **710**
> family unions sum to **88.65 ms** — the 16 named rows of the per-family ranking summing to
> **76.884** plus that ranking's own derived 694-family remainder of **11.768** — against a `merged`
> of **66.14 ms**, so **22.5 ms (25 %)** of the summed union is cross-family overlap.
>
> Over the **seven** event-subset cuts whose LOO is fractional — the six families above plus the
> `_internal` staging subset, which keeps 63.7 % — **the LOO spans 23.4 %–65.5 % of the cut's own
> union.** So the two readings are not interchangeable in either direction: **the union *overstates*
> what a removal buys** (by **1.5–4.3×** on these rows), and **the LOO *understates* the family's
> coverage** of that time, for the same reason. **Quote the LOO when you are pricing; quote the union
> when you are describing occupancy; never substitute one for the other.**
>
> The `fused_wqa_wkv` row is the sharpest case, and it is not a rounding artefact: **0.2505 ms of
> work and a marginal of exactly 0.0000** — the rest of its recipe still occupies that time, so
> removing it buys **nothing**. **You cannot cut inside a recipe and expect a saving.** That is why
> the span boundaries that matter are **recipe** boundaries.
>
> *Two controls from the same artifact, because the table alone cannot tell you which regime you are
> in.* **(a)** For a **handle-complete** span the trace price is essentially exact: the one clean
> single-variable ablation the archive holds (W1 = W0 with the `fp4_dequant` device body removed, same
> op, same signature, same launches, same recipes) takes `merged` **66.1349 → 51.5456** (the same
> `W0` arm, rank 0, as above), a **−14.5892** fall, of which
> `Δ own_union(MOE_ROUTED) = −14.5465` is **99.7 %**. **(b)** For a cut **inside a
> run** the trace is a **lower bound** and under-prices by a measured **21.1 %**
> (LSO predicts W1 `merged = 54.0936`, measured 51.5456 — the complement's union *shrinks* by 2.55 ms
> once 22.6 ms of TPC work vanishes, because the non-fp4 events **inside the same recipes**
> re-overlap). **The gap is intra-recipe, which is consistent with the measured fact that
> cross-recipe overlap is exactly zero.**
>
> **MEASURED** *(all rows and both controls re-derived for this table from the archived scripts:
> `Archive/by-date/2026-10-12/span-ranking/` — `REPORT.md` §3's event-subset table, `sr8_spans.py`,
> `sr11_final.py`, `sr9_ablation.py`; the family unions and `merged` cross-check against the
> per-family leave-one-out ranking in the same project's `notes/interval-census.md` §5.1.)*

### 3.6.5 The largest single node in the decode step is a KV-slab materialisation — BYTE-bound, and its removal by a span rewrite is MEASURED

*(Added 2026-10-13. A single-node finding from one decode run of the port at **TP=4 across all four
modules, unpinned**. It is a **compiled-graph** measurement plus trace arithmetic, not a timing run,
and it is `[DSv4F]`. **Extended the same day by the two-arm compile gate that discharged the rewrite's
conditionality — the verdict box here, the gate itself at the end of this section.**)*

> **VERDICT — the conditionality behind this span's rewrite is DISCHARGED.** The rewrite plan's floor
> for replacing this span — **`0.2324 ms/step`, 42.2× span-internal / 1.17× model** — rested on an
> **inference**: that spelling the slab as a **declared tensor operand**, read and written **through
> its base address**, **removes the staging pass** rather than moving it elsewhere. **That inference is
> now a measurement**, because the experiment designed to discharge it ran, and **its control
> succeeded — twice**:
>
> | arm | geometry | graph nodes | `memcpy_dma` staging nodes | nodes with non-zero `dma_expected_duration` |
> |---|---:|---:|---:|---:|
> | **control 1** — the shipped chain at this geometry, in the treatment's own process | 2,422 blocks × 1,600,000 B | 33 | **1** (`num_of_ROIs` **243**, dur **3719.180392235825**) | 3 |
> | **control 2** — the **full model at TP=4**, cold recipe cache, 21,769 post-graphs | real | 111 / 210 / 242 | **116** graphs carry it | **3,103** (116 of them the staging signature) |
> | **treatment** — the rewrite's spelling, same geometry, same process | same | 2 | **0** | **0** |
>
> **Control 1 reproduces this section's own node bit-for-bit** — same `guid`, same `num_of_ROIs`, same
> `dma_expected_duration` to the last digit, and the same `slice → memcpy → reshape →
> index_put/scatter_nd → index_select → gather` dataflow. **And the treatment, in the same process, on
> the same geometry, with the same compiler, produces no DMA node of any kind.** So the staging pass is
> a property of the **shipped expression**, and it does not exist when the slab is a declared operand
> accessed through its base address. **The rewrite's floor stands, and the span is worth rewriting.**
>
> **Two bounds on that, stated here rather than in a footnote.** **(1) It measures that the staging
> pass disappears — not what the replacement costs.** The treatment op is a **skeleton** at this
> geometry (the model itself may not be edited), so the rewrite's per-launch intercept — the plan's
> **N4**, its own named principal risk — is **still unmeasured**, and the floor's launch term still
> comes from a **harness kernel**, not from model nodes. **N4 remains UNKNOWN** (§3.9 item 22).
> **(2) It does not repair the denominator below.** The `1.81 TB/s` discussion bounds the **shipped**
> pass's byte floor; this gate removes the **pass**. Those are different claims, and the gate settles
> neither the envelope nor the unmeasured regime the stage lives in.
>
> *Artifact: `Archive/by-date/2026-10-13/compile-gate-run/` — `REPORT.md` §0–§3, §5, §7 and the per-arm
> extracts `evidence/control_SHIP_REAL.json`, `evidence/control_SHIP_REAL_READ.json`,
> `evidence/control_SHIP_MINI.json`, `evidence/treatment_R3_REAL.json`. The floor composition and the
> branch table are the rewrite plan's own
> (`Archive/by-date/2026-10-13/attn-preamble-rewrite-plan/REPORT.md` §0–§0.5; that archived copy
> **predates the gate**, so its §0 headline still reads *"conditional on an inference"* — the
> **pre-gate text, retained deliberately as the record**).*

**What the node is.** The decode `ATTN_PREAMBLE` recipe compiles to **one** 111-node graph
(engines: `None` 53 / `TPC` 47 / `DMA` 9 / `MME` 2), and **exactly one** of its nodes is a staging
copy — `24_memcpy_116119_internal`, engine `DMA`, type `DmaMemcpy`, guid `memcpy_dma`,
`num_of_ROIs` **243**, `dma_expected_duration` **3719.180392235825** µs, `origin_nodes [97147]`. It
is **99.96 %** of that graph's whole DMA budget (3719.18 of 3720.59 µs). The operation it performs,
read from the **pre** graph for the same recipe (which has **no** memcpy node at all — the staging is
inserted by placement):

```
pre graph:  slice_7/slice/16750   guid "slice"   in ["8"]   out ["24"]
   8  : bf16 [800000, 2422]  strides [2, 1600000, 3875200000]   <- STRIDED view of the int8 slab
   24 : bf16 [32768, 2422]   strides [2, 65536, 158728192]      <- DENSE
        32768 x 2 = 65536;  2422 x 65536 = 158,728,192          <- the density check
   31 : bf16 [512, 64, 2422] strides [2, 1024, 65536, 158728192]
post graph: the node, inserted by placement
   24_memcpy_116119_internal  (engine DMA)   in [24_memcpy_116119]   out [24]
   24_memcpy_116119 : bf16 [32768,2422]  alias_of "124"  strides [2, 1600000, 3875200000]
   24               : bf16 [32768,2422]  alias_of "124"  strides [2, 1600000, 3875200000]
         ^ identical descriptors — an ADDRESS-ASSIGNMENT ARTEFACT, which reads as a self-copy.
           The pre graph is what settles the operation.
downstream (both graphs):  reshape -> [512, 64, 2422] = [head_dim, block, n_blocks]
   -> index_put/scatter_nd_onnx_fwd_bf16 (TPC)   <- the SWA cache WRITE
   -> ... -> int8 slab 123 -> ... -> 125
   -> index_select_1/gather_fwd_bf16 (TPC)       <- the SWA window READ
   -> gather/gather_elements_fwd_bf16 (TPC)
```

**The origin is the paged KV-cache slab.** The base tensor is int8 `[1600000, 2422]` =
**3,875,200,000 B = 3.875 GB**, `is_dense`, `persistent`, in DRAM; the node's input is a **bf16
strided view** of it and its output is the **dense** form. So the operation is a **strided → dense
materialisation of `32768 × 2422 × 2 = 158,728,192 B (158.7 MB)` per layer-launch** — the whole slab,
all 2,422 blocks, as a bf16 tensor. `n_blocks = 2422` and `block = 64` are fixed by the
`[512, 155008]` input of the `gather_fwd_bf16` in the `ATTN_CORE` instance, since
`155008 = 2422 × 64`.

**It is BYTE-bound, not descriptor-bound — and that reframes the fix.** Byte accounting from the dump:

```
staged destination (dense, MEASURED)   32768 × 2422 × 2 B   =    158,728,192 B  = 158.7 MB / launch
  × 43 launches/step                                        =  6,825,312,256 B  = 6.825 GB/step (one way)
  round trip (strided read + dense write)                   =  13.65 GB/step
byte floor  13.65 GB / 1.81 TB/s  (ASSUMED: midpoint of a RETRACTED range)  =  7.542 ms/step
  the same floor at the four MEASURED DMA points, 16 / 64 / 256 / 1024 MB (§1.6):
    13.65 GB / 1.7751 / 1.7766 / 2.0213 / 2.0284 TB/s       = 7.6897 / 7.6832 / 6.7531 / 6.7294 ms/step
measured S4 union (this span's own stage)                   =      7.7751 ms/step
  -> 1.011× / 1.012× / 1.151× / 1.155× those floors        = 98.9 / 98.8 / 86.9 / 86.6 % of envelope
```

> **CORRECTED 2026-10-14 — the launch count in the block above is `43`; the step issues `64`.** The same
> staging node runs in **two** recipes — **43 `ATTN_PREAMBLE` launches and 21 `ATTN_CORE` launches**,
> **disjoint in time** — so **every byte figure here scales by `64/43`**: `158.7 MB × 64 = 10.1586 GB`
> one way, **`20.3172 GB` round trip**, and the four byte floors become **`11.446 / 11.436 / 10.052 /
> 10.016 ms/step`**. **The `98.9 / 98.8 / 86.9 / 86.6 %` band is unchanged**, because both the bytes and
> the measured union scale together — so this section's *verdict* (the pass sits at ~99 % of the
> envelope, and the lever is **traffic**) stands; only its denominators were **1.49×** too small. The
> `43`-launch arithmetic is kept above as the record. **Population evidence, the second node's identity,
> and the reconciled rate: §3.1.4** — the `ATTN_CORE` node is `46_memcpy_124917_internal` in
> `Archive/by-date/2026-10-13/staging-origin-nodes-dump/evidence/post_decode_ATTN_CORE_210node.json`.

**The pass runs at 87–99 % of the DMA envelope — and the `~97 %` that used to stand here rests on a
denominator that is ASSUMED, not measured.** It is **not inefficient — it is efficiently doing work
that should not need doing**, which is the same shape of finding as the fp4 decode's fold (§3.4.5) and
the same one-line lesson: *check the job before you optimise the kernel.* **The lever is therefore
TRAFFIC, not descriptor count.**

- **`1.81 TB/s` is not a measurement.** It is the arithmetic midpoint of the range `1,731–1,888 GB/s`,
  and **that clause was retracted on 2026-10-11**: those points are the **TPC `memcpy_u8`** path, not
  the DMA engine. **§1.6 carries no `1.81 TB/s` figure at all** — its DMA numbers are **1,775.1 /
  1,776.6 / 2,021.3 / 2,028.4 GB/s** — and §1.6 explicitly rules this size class out: *"Below 16 MB
  we have no DMA-engine measurement at all."* **ASSUMED.**
- **Against the four real DMA-engine points the honest figure is a band, not a point.** At the four
  measured points the byte floor is **7.6897 / 7.6832 / 6.7531 / 6.7294 ms/step**, so the stage sits at
  **98.9 % / 98.8 % / 86.9 % / 86.6 %** of the envelope. **The verdict weakens and does not survive as
  stated — but it does not reverse.**
- **The denominator is unmeasured in the regime this stage lives in.** Per descriptor the pass moves
  **130,640 B** (`158,728,192 ÷ 1,215`) — **128× below** the smallest DMA-engine point (**16 MB** per
  copy). So we **cannot tell** whether **7.7751 ms/step** is the DMA engine's floor for this access
  pattern. **UNKNOWN** — a negative from a completed search, not a failed one.
- **The ratio is circular, and that is the most instructive part.** At `1.81 TB/s` the measurement
  implies **14.07 GB** of traffic against the counted **13.65 GB** — a **3.1 %** excess that *is* the
  "**1.031×**". At `2.0213 TB/s` it implies **15.72 GB**. **The denominator was selected to make the
  floor land on the measurement. That is fitting, not bounding.**
- **And a stronger constraint comes free from this section's own data: a floor cannot exceed its
  measurement.** At `1.731 TB/s` the byte floor would be **7.886 ms/step, above the measured
  7.7751** — impossible — so the envelope is bounded **below** by
  `13.65 GB ÷ 7.7751 ms` = **1.756 TB/s**, which **excludes the low end of the cited range on our
  own evidence**.

**What survives is the structural claim, and it is the one that matters: the pass is a 13.65 GB round
trip serving ~2.8 MB of useful fp8 KV.** An earlier reading of this stage bounded its byte floor at
"≤0.72 ms/step" from an assumed ≤30 MB/layer staging budget and concluded that the **descriptor** count
(**5.425 ms/step**) was the binding floor; the dump's real budget **refutes that byte floor by ≥10×**
and displaces it. **MEASURED / DERIVED** for the traffic and the counted bytes; **ASSUMED** for the
envelope; **UNKNOWN** for the envelope below 16 MB.

> **Provenance note, because it is how the wrong number got in.** The `~97 %` came from
> `Archive/by-date/2026-10-13/staging-origin-nodes-dump/REPORT.md`, which states the `1.81 TB/s`
> denominator flatly; the rewrite plan
> (`Archive/by-date/2026-10-13/attn-preamble-rewrite-plan/REPORT.md` §0.8) carries the **same number
> with the retraction attached**. **This section inherited the weaker source on exactly this point.**

> **One apparent tension, and it dissolves — the `17.27` "concurrency" is a marker artefact.** The
> arithmetic above implies **~101.6 GB/s per busy engine** at a "concurrency" of **17.27** — but
> **17.28 descriptors in flight on a five-lane pool is impossible** (§1.2, *"five lanes in parallel"*),
> because the DMA marker **self-overlaps 14.5×** its own union (§2.9 rule 2): `Σdur/union = 17.28` is
> **self-overlap, not concurrency**, so neither the per-engine rate nor any floor built on it stands.
> At the real five-lane pool the descriptor floor is **26.04 ms/step = 3.35× the measurement**, and
> **a floor cannot exceed its measurement**, so the descriptor reading is refuted by its own data. *(That
> 26.04 is the plan's **52,460 descriptors/step**; this section's own identity below counts
> **52,245**/step, at which the same floor is **25.93 ms/step = 3.34×** — the two counts differ by 0.4 %
> and both refute the reading.)*
> **The only trustworthy S4 quantity is the union, 7.7751 ms/step** — which is exactly the quantity
> the byte reading uses, and why the *byte* reading is the one the dump supports. *(Artifact:
> `Archive/by-date/2026-10-13/dma-envelope-provenance/REPORT.md` §5.)*

**Two identities, and one UNKNOWN that must stay UNKNOWN.**

* **`1,215 = 243 ROIs × 5`, verified on two independent nodes.** A 243-ROI node contributes 9,720
  trace events over 8 launches = **1,215 per launch**; a **1**-ROI node contributes 40 events over 8
  launches = **5 per launch**, i.e. **5.0 per ROI**. The same ×5 closes the step total:
  `52,245 descriptors/step ÷ 43 launches = 1,215.0`. **The mechanism of the 5:1 ratio is UNKNOWN** —
  measured on two nodes, unexplained, and immaterial to the identity, which the node name, the budget
  and the count all fix independently. **Do not invent one.**
* **`dma_expected_duration` here is a computed value, not a measurement — and it OVER-predicts.**
  It is bit-identical (**3719.180392235825**) in all **118** instances of this node, across 4 ranks,
  both recipes, and both the decode and prefill compilations. And the trace's own `Σdur` for the
  `24_memcpy_*` population is **536.977 ms over 4 frames = 134.244 ms/step** (208,980 events =
  52,245/step) = **3.1220 ms/launch** — against the model's 3.7192: the model is **1.19× high**.
  **Quote both; use neither as the other.** *(Read this against §1.7's per-transfer reading of the
  same field, where the measured window is **1.03–1.16× the model** — the opposite sign. The two are
  not the same quantity: §1.7 compares one node's **window** with the model, this compares the model
  with a **sum of descriptor durations over a 243-ROI node**. Neither number transfers to the other
  case, and §2.10 trap 3's "optimistic lower bound" framing is therefore **size- and
  granularity-dependent, not a global sign**.)*
* **The census: 118 instances, ONE distinct staging origin.** Over **21,763** post-graph files,
  **118** staging nodes appear — **31** decode + **28** prefill in `ATTN_PREAMBLE`, **30** decode +
  **29** prefill in `ATTN_CORE` — and there is **one** distinct staging origin class: a strided bf16
  `[32768, 2422]` view of an int8 paged slab, materialised dense. In `ATTN_PREAMBLE` the slab is base
  `123`/`124`; in `ATTN_CORE` it is base `9` — two slabs of the same shape class and nothing else.
  **No weight or operand tensor appears in any staging node's dataflow.** **MEASURED.**

> **The instrument lesson, and it is worth more than the node: `origin_nodes` is a red herring.**
> Its value is `[97147]`, and **that id is not in the post graph's own node-id set** (the graph's ids
> run **97422…97661**), so `origin_nodes` alone **cannot name anything**. The instrument that *does*
> name the tensor is the node's **`input_tensors` / `output_tensors` resolved through the graph's own
> `tensors` table**, which carries `dtype`, `max_shape`, `strides`, `alias_of`, `offset`,
> `allocation` and `persistent`. **Read the tensors, not the origin ids.** (§2.10.)

**The generalisable lesson for any span ranking: report LAUNCHES-PER-SPAN.** `ATTN_PREAMBLE`'s
**9.8171 ms/step** own union is **79 % — one staging node that its boundary happens to enclose**,
not its own work — and it is simultaneously the largest span whose boundary passes the
acceptance tests *and* the one enclosing **43** large attention launches. A span's headline cost can
be dominated by what the boundary happens to contain, so **any span table should report, per span,
the number of recipe launches it holds**, alongside its milliseconds. **MEASURED / DERIVED.**

**And the instrument has a hard precondition: a COLD recipe cache.** This run pointed
`LAD_RECIPE_CACHE_ROOT` at a fresh empty directory, cold-compiled every decode recipe, and wrote
**21,763** post-graph files. **MEASURED** for the requirement and the run. The *reason* the
precondition is hard is a **stated mechanism, not an observation (`source read`)**: `DUMP_POST_GRAPHS`
is written by the graph compiler, and a Synapse **recipe-cache hit loads a compiled blob and builds no
graph** — so a warm cache is *stated* to yield an **empty dump directory and a false negative**, not an
error. So §2.10's last bullet is **refined**, not upgraded: the cold-cache requirement is measured, and
the failure mode is the mechanism the precaution is taken against.

**The two-arm compile gate — and why the earlier attempt could not settle it.** *(Added 2026-10-13.)*
The gate is a **standalone two-arm compile** at this section's own geometry, in **one process** with a
**cold recipe cache**: a **control** that spells the shipped expression exactly — the typed page view,
`page[blocks, offs] = kv`, `page.index_select(0, phys).to(bf16)`, `torch.gather(...)` — and a
**treatment** that spells the rewrite's access forms, with the slab as a **declared tensor operand**,
read and written **through its base address**, and **no `[32768, 2422]` intermediate anywhere in the
expression**. Three further arms are controls on the *instrument* rather than on the design:

* **`SHIP_MINI` — the *previous* falsifier's control, reproduced.** At its miniature geometry (8
  blocks, one contiguous page) it produced **the same two DMA nodes to fourteen significant figures**
  (`DmaBroadcast` **0.5516913659793814**, `transpose_dma` **0.25169136597938147**) and **no staging
  node**. **The cause is geometric, and it is the whole story: an 8-block contiguous page has nothing
  to materialise.** At 2,422 blocks × 1,600,000 B the same expression materialises 158.7 MB. So that
  attempt's zero was **a zero from a configuration that could not produce the thing being tested
  for** — which is why it neither confirmed nor refuted the removal.
* **`SHIP_REAL_READ` — the write removed, the read kept.** The `index_select` + `gather` alone, with no
  `page[blocks, offs] = kv`: **11 nodes, 2 DMA, and no `memcpy` node of any engine.** **The staging is
  therefore forced by the WRITE, not by the read** — which is where an implementer should aim: the
  out-of-place `index_put`/`scatter_nd` on the strided page is what has to materialise its base.
* **`R3_DENSE` — the treatment with a *contiguous* slab operand**: still **0 DMA nodes**, so the zero
  is not an artefact of declaring a strided operand.

**The generalisable rule, and it costs less than the experiment.** *A control that cannot fail in the
way the treatment is supposed to fail cannot license the treatment's result.* That is the same shape as
this section's own **"a floor cannot exceed its measurement"** — a check that costs nothing, applies
before the run, and would have caught the earlier attempt immediately.

**One caveat on the fresh full-model arm, and it is a row count, not a discrepancy.** Control 2's slab
is **2,421** rows — one fewer than the **2,422** this section carries — so its
`dma_expected_duration` is **3717.6702440399486** rather than **3719.180392235825**. The difference is
entirely the byte count at one bandwidth:
`32768 × 2422 × 2 B ÷ 3719.180392235825 µs = 42.678271893281 GB/s` and
`32768 × 2421 × 2 B ÷ 3717.6702440399486 µs = 42.677979913459 GB/s` — **the same bandwidth, agreeing to within ~7×10⁻⁶ relative.** The row count is a KV-cache-sizing outcome of the engine build, not something
the experiment controls, and **control 1 is the 2,422-row reproduction** — so the two durations are one
quantity evaluated at two row counts. **Do not read the one-row difference as a contradiction.**

*Artifact: `Archive/by-date/2026-10-13/staging-origin-nodes-dump/` — `REPORT.md` §1–§3, §5–§6,
`evidence/post_decode_ATTN_PREAMBLE_111node.json`, `evidence/pre_decode_ATTN_PREAMBLE_111node.json`,
`trace_census.json`, `staging_243.json`, `run.log`. The byte floor, the measured union and the 79 %
share are the rewrite plan's own §0.3/§1.3/§0.7 arithmetic, restated here.*

---

### 3.6.6 An in-place write costs what the functionaliser can SEE — a view-level write forces no second whole-buffer *allocation*, an opaque one is copied

*(Added 2026-10-15. Microbenchmarks at the model's own slab geometry — **no model, no vLLM, no server**,
so this is not a `[DSv4F]` workload measurement. One process per arm, cold recipe cache. The write is
verified to land in every arm **where a write is expected**, and verified *not* to land in the negative
control (`nonmut_unused`, which fails both its checks by construction). Two arms carry no such verdict:
`neg_readonly` writes nothing, and `internal_alloc`'s own `args[0]` check is void by construction — its
landing is read off the returned window instead.)*

> **VERDICT — in the currency of *peak allocation*, the cost of an in-place write is decided by whether
> the functionaliser can see it, not by whether it mutates.** On a **3.8752 GB** int8 slab:
>
> | how the write is spelled | peak allocation during the call | × buffer | compiled |
> |---|---:|---:|:--:|
> | `page[blocks, offs] = kv` — **view-level, ATen-visible** (**the shipped form**) | **262,144 B** | **0.00007×** | yes |
> | an **opaque** op declaring `Tensor(a!)`, mutating a **flat contiguous** view | **23.25 GB** | **6.00007×** | yes |
> | the same opaque op, run eager | **8 MiB** | **0.00216×** | **no** |
>
> **The two compiled spellings differ by ≈88,700× in that currency.** They do **not** hold the write's
> operand fixed — the opaque op requires a **flat contiguous** operand while the ATen write targets the
> **strided page**, so that pair is **not** single-variable. The pair that *is* single-variable is the
> opaque op **eager versus compiled** (8 MiB versus 23.25 GB — same arm, same operand), and that is what
> establishes **this is not a property of the runtime.** **MEASURED.**
>
> **Peak allocation is not the whole cost of a write, and "no extra allocation" is not "free".** The
> view-level spelling forces **no second whole-buffer allocation** — but it is the *same* spelling
> §3.6.5 measures forcing a **strided-page staging pass** of 158.7 MB per layer-launch (13.65 GB/step;
> **20.3 GB/step** under §3.6.5's own `43 → 64` launch correction). **Those are two different costs of
> one write**, and this section is about the second one only. Read the two together.

**What the two spellings are.** The shipped form is an advanced-index assignment into a **view** — the
page construction and SWA write of `_dsv4_page` / `_write_paged_swa_cache`:

```
page = slab.view(torch.bfloat16)[:, off:off + pe].view(-1, bs, hd)  # STRIDED 3-D view; the probe runs off = 0
page[blocks, offs] = kv                                             # view-level, in-place, ATen-visible
```

The expensive form is the *same write* performed inside a custom op that **declares its operand
mutated** — a `Tensor(a!)` argument, or a Python op registered with `mutates_args=`. That is the route
§3.6.5's rewrite would have taken, which is why this section sits here. *(The shipped page offset is
per-layer and wide; the probe below runs with `off = 0` — see the scope paragraph.)*

**The mechanism is torch's, and it needs no HPU to observe.** When a mutation is **opaque** — hidden
inside an op the functionaliser cannot decompose into view updates — AOTAutograd wraps the call in
`torch.ops.higher_order.auto_functionalized_v2`, **whose result is the whole buffer as a new value**, and
then writes that value back with `aten.copy_.default`. Read off a CPU-only run (torch 2.11, no device in
the process, storage traced):

```
auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(..., _all_bases=[arg0_1])
getitem_1: "f32[2097152]" = auto_functionalized_v2[1]                       # whole base, NEW storage
copy_:     "f32[2097152]" = torch.ops.aten.copy_.default(arg0_1, getitem_1) # whole base, back
```

A **visible** mutation is functionalised *in the same kind* — into a whole-base value — but by a
**structured** primitive, and that is the entire difference:

```
index_put:     "f32[256,32,32]" = aten.index_put.default(...)       # page-sized
slice_scatter: "f32[256,8192]"  = aten.slice_scatter.default(...)   # WHOLE BASE
copy_:         "f32[256,8192]"  = aten.copy_.default(arg0_1, slice_scatter)
```

So the difference between the two spellings is **structured primitive versus opaque hop**, *not*
copy-versus-no-copy — on CPU the structured form materialises a whole base too. **MEASURED.**

**A backend can decline to pay it, and this one does.** The HPU graph for the structured write carries
`slice_scatter/slice_insert`, `guid logical_slice_insert`, `engine None`, `is_logical True`, and its
output tensor is `alias_of` the **input**, in the **input's own memory section**, `persistent false`.
That is an in-place insert expressed as an alias rather than a whole-base materialisation, and it is why
the shipped form measures 262,144 B. **MEASURED.** *(Scope: the CPU arm and the HPU arm are structurally
but not geometrically equivalent — different dtype, shape and page size — so the supported statement is
"HPU lowers the structured write's whole-base value to an alias while the CPU AOT path materialises it",
**not** "HPU beats CPU".)*

**What the numbers are, and what they are not.** 23.25 GB is the **allocator's** peak delta
(`max_memory_allocated()`) in one run. It is **not** a traffic byte count: the dump carries **no byte
field on a node or in its `tensors` table**, so every byte figure for *data movement* read from a dump is
a declared shape × dtype size. (The graph object does carry a byte-valued `workspace_size`, but that is a
workspace *allocation*, not bytes moved — and in our own local census of this dump family it appears
in only 224 of 21,763 files.) And the dump's own total is **run-dependent** (6 declared entries / 23.25 GB in one keep run, 9 /
34.88 GB in another). By **distinct storage** the dump declares **at most 3** slab-sized storages (11.63 GB), and
"at most" is load-bearing: in the two keep-flag-`0` runs, two of the three entries share a memory section
at identical byte size, so they may be one memory re-typed, and the dump cannot say which. The reconciliation between the two readings is
**UNKNOWN**. The one genuinely independent instrument is **wall time**: the opaque arm costs **+11.77
ms/call** over the read-only control at the same geometry (11.988 vs 0.222 ms), which is **20.9–23.9 GB**
at the measured DMA band (§1.6) and **26–28 GB** at the TPC band
([Chapter 4](04-execution-model.md) §4.1.2) — note the whole-base nodes are TPC (`memcpy_bf16`, and a
`copy_fwd_i8` identity), not DMA. **The copies execute; they are not virtual.** *(Both bands are
imported, not measured in this artifact.)*

**Three things that do NOT explain it, all tested.** **`PT_HPU_KEEP_INPUT_MUTATIONS`** — `0` and `1` both
give **6.00007×** across three runs, with the setting verified applied from each run's own bridge config.
On HPU the flag does not move the number; on **CPU** it is measured to only *relocate* the copy-back
between the graph and the epilogue, leaving the functionalised whole-buffer value in place. **Exclusive
ownership** — allocating the buffer *inside* the compiled region, so nothing external aliases it, still
gives **6.00007×**. **Dtype** — on the **Python**-registered op, bf16, int8, fp32 and fp8 at a fixed
element count all give **1.00002×** across a 4× spread in bytes; the C++ op is bf16-only by construction,
so its behaviour at other dtypes is **UNKNOWN**. **MEASURED.**

**The escape hatch exists and is out of bounds.** Declaring the op **non-mutating** while it writes does
remove the cost — but the graph shows the read of the buffer has **no data dependency on the op's
output**, so the ordering that makes the write visible is **accidental, not declared**. The matched
control proves the accident is load-bearing: the same declaration with an **unconsumed** output has its
op eliminated and **the write does not land at all**. A false declaration that happens to give the right
answer on one compiler version is undefined behaviour with a lucky sample, not a design. **Do not use
it.** **MEASURED.**

**Scope: the shipped port is in the class that forces no second whole-buffer allocation.** The model's own compiled
dumps carry the `memcpy_dma` staging node §3.6.5 documents — `num_of_ROIs` **243**,
`dma_expected_duration` **3719.180392235825**, fed by a bf16 `[32768, 2422]` view of the slab — and both
this probe's view-level arm and §3.6.5's own compile-gate **control 1** reproduce that node exactly, the
latter from a control that spells the **shipped expression** itself. So the identification is made against
the model's own graph, and it is robust on the axis the snippet above fudges: the **118** instances in
the model's decode dumps share one duration, one ROI count and one in-tensor descriptor across **43
distinct `user_mem_offset` values**, `0` on only **three** of them. **MEASURED** *(the node
identification)*. `src/vllm-gaudi` and `src/custom_ops` declare **zero** mutating custom ops — not only
zero `Tensor(a!)`, but zero `Tensor!`, zero `Tensor(x!)` and zero `mutates_args=`, and every op in
`src/custom_ops` is declared non-mutating. In vLLM itself the
`Tensor(a!)` declarations are confined to the CPU and CUDA bindings (`csrc/cpu/torch_bindings.cpp`,
`csrc/libtorch_stable/torch_bindings.cpp`, `csrc/flashkda_registration.cpp`). **The hazard is latent, not
live.** **source read** (`grep -rn 'Tensor(a!)' src/vllm-gaudi src/custom_ops` → 0, and likewise for
`Tensor!`, `Tensor(x!)` and `mutates_args=`).

**The porting rule.** *Do not hide a mutation from the functionaliser.* Prefer a **view-level ATen
in-place write** — the compiler has a cheap path for it. Treat a custom op that declares `Tensor(a!)` on
a **large shared buffer** as a hazard and price it before adopting it: at this geometry it is 23.25 GB of
peak per call, not 262 KB.

**What this section does NOT establish.**

1. **Why the compiled opaque form is 6.00007× while the same pipeline's per-op-lowering form is
   3.00440×.** That residual has **no mechanism**, and the 3.00440 row must be read as *the
   `torch.compile` pipeline up to and including the HPU per-op lowering* — it emits HPU Synapse graphs
   including whole-slab copies — **not** as an isolation of AOTAutograd. Do not decompose the step.
2. **Any size dependence.** Graph structure is size-independent in **both halves** — HPU 4 KB–3.8752 GB,
   CPU 4 KB–1 GB — and only peak *residency* moves, with a cause that is **UNKNOWN**. Quote no threshold.
3. **What `inductor` pays.** Its runtime trace is inert and its wall time is ~6× lower on the same
   graph, so whether it materialises the whole base is **UNRESOLVED** — stated here deliberately, because
   a reader who knows `inductor` would otherwise infer the opposite claim from the rows above.
4. One sighting of a compiled **silent write-drop** on a strided view is recorded only as
   **UNRESOLVED — the observation was never recorded**. Four clean re-runs with preserved artifacts land
   the write every time, so neither "it is a defect" nor "it did not reproduce" is supported.

*Artifacts: `Archive/by-date/2026-10-15/cow-mutability/` — the device half: `REPORT.md`, `logs/*.json`,
the **per-run** censuses (`logs/census_pos_m2__hpu__m0__*.json`, where the "at most 3 distinct storages"
figure lives — the canonically-named per-arm census does not carry those keys) and
`logs/keep-run-bridge-config/*.probe.log`, which is the record that the keep-flag setting was applied.
And `Archive/by-date/2026-10-15/cow-mutability-torch/` — the CPU half: `REPORT.md`, `results.json`,
`evidence/*.txt`, which is where the graph text above is quoted from. The adversarial review behind the
retractions above is at `Archive/by-date/2026-10-15/cow-mutability-review/`. The staging node's own
record, and the control that identifies the shipped expression, are §3.6.5's
`Archive/by-date/2026-10-13/staging-origin-nodes-dump/` and
`Archive/by-date/2026-10-13/compile-gate-run/`.*

---

## 3.7 Multi-device, pinning, and concurrent processes

### 3.7.1 There are four numberings of one card — and the **module id** is the one to reason and pin in

- **Physical module id / AIP** — what `hl-smi -L` shows as each AIP's Module ID. **The card's
  identity, fixed per card** — and the numbering that the pin (`HLS_MODULE_ID`) and the
  device-acquire path both speak, so **it is the one to reason and pin in**.
- **Local device index** — a **dense `0..N-1`** numbering over the devices visible to your process;
  what `torch.hpu.device_count()` reports and what `torch.hpu.set_device(i)` indexes. It is a
  **position**, not an identity: it is an index *into* `HABANA_VISIBLE_MODULES`, so it moves when the
  allocation moves. **This is NOT the vendor tool's index order** — see the table below.
- **`hl-smi` card index** — the vendor tool's own numbering, and a **third** one. It is
  **position-dependent** as well: it is a host index, and it is not the local index.
- **`hl-smi`'s default per-AIP row number** — the **accel-enumeration ordinal**, i.e. the same
  numbering the tool prints as `accelN`. It is **dense** over the AIPs you can see, which is why it is
  neither the index nor the local index: on this box the default view numbers its four rows `0..3`
  while `-Q index` reads `0, 2, 3, 6` for the same four cards.

`HABANA_VISIBLE_DEVICES` is a **container-runtime input**, consumed **outside** the container.
Nothing in the image reads it, so setting it *inside* a container is **inert** — the container's
`--device` flags are what actually isolate the cards. **INFERRED**
`HABANA_VISIBLE_MODULES` **is not a filter in this stack** — no library in `/usr/lib/habanalabs/`
contains the string, and the one C++ consumer uses it only as a **presence flag** before falling
through to the by-type acquire. **INFERRED** *(grep + disassembly; `hpu_isolation` §2.3.)* Its real
effect is the **local-index → module-id table** that `torch.hpu.set_device(i)` indexes. **MEASURED**
*(`hls_module_id` §4.7 — the mapping was verified physically for all four indices.)*

> **"Not a filter" is scoped to the *image*, and the vendor's page is the other half of the same
> sentence — read them together, or they look like a contradiction.** The container runtime accepts
> **either** variable to select the mount — *"you can select a Gaudi device to mount on Docker by
> setting `HABANA_VISIBLE_DEVICES` or `HABANA_VISIBLE_MODULES` when the Docker container is started"*
> — and recommends the module-id form (same page and fetch as below). So **the runtime** is where the
> module-id spelling is preferred, and **the image** is where the variable is a table rather than a
> filter; the INFERRED finding above is scoped to `/usr/lib/habanalabs/` and says nothing about the
> runtime. Both halves are needed: the runtime is why the module-id form is the one to set, the image
> is why the module *set* is what you read back.

**`HABANA_VISIBLE_DEVICES` and `HABANA_VISIBLE_MODULES` are not two knobs and not alternatives: they
are ONE allocation written in two numberings.** The module set is the identity; the index list is its
**position-dependent image**. On this box, 2026-10-15:

```
HABANA_VISIBLE_MODULES=0,1,2,3          <- the allocation, named in module ids
HABANA_VISIBLE_DEVICES=2,6,0,3          <- the SAME four cards, as its equivalent index rendering
hl-smi -Q index,module_id -f csv   ->   index 0 -> module 2
                                        index 2 -> module 0
                                        index 3 -> module 3
                                        index 6 -> module 1
```

The indices `2,6,0,3` are **exactly** the host indices of modules `0,1,2,3`, so the two name the same
four cards — and the index list is **derived, not given**: note that it is *not* `0,1,2,3` in order,
because the allocation is dense in **module** space, not in index space.

**Which of the two is actually *set* is a separate question, and the honest answer is narrower than
"both".** The container launcher here sets **only** `HABANA_VISIBLE_MODULES` — while **its own header
comment names `HABANA_VISIBLE_DEVICES=2,6,0,3` as the mechanism**, which is the same allocation in the
other numbering. The index list `2,6,0,3` is therefore the **equivalent rendering** of what the
launcher sets: documented in that comment, and **observed set alongside the module variable in an
earlier launch**, where the two agreed. So do not read this as "the container sets both" — read it as:
*the module variable is the one to set and to read; the index form is a rendering of the same
allocation.* **⇒ reason and pin in module ids; treat any index list as a derived, position-dependent
rendering of the module set.** **MEASURED** *(`Archive/by-date/2026-10-15/` — the index→module→PCI map
and both `hl-smi` snapshots: `ordpost2/REPORT.md`'s "Device, before and after" table; the
serial-carrying capture and the by-serial comparison:
`device-host-map/HOST-AND-CONTAINER-DEVICE-MAP.md`. **Scope:** the launcher is **working-tree state in
this project, not a committed artifact**, and the "earlier launch" is a recorded observation — so this
one sentence is **deployment-local**, not a property of the stack. What is `MEASURED` is the
index→module map and the module table.)*

⚠ **The numberings disagree, and "card N == module N" is not a rule — it is a coincidence when it
holds.** On this box (**SUPERSEDED 2026-10-15** at the values below — they were right for their own
box and were overtaken by a later one; the rule did not move):

| what | order |
|---|---|
| `hl-smi` card index → module | **`0→2, 2→0, 3→3, 6→1`** |
| local torch index → module | **`0→0, 1→1, 2→2, 3→3`** |
| modules **visible to this container** | **`0, 1, 2, 3`** |

Read the arithmetic, because the *shape* of the failure has changed: the identity holds for **1 of the
4** `hl-smi` indices (only `3→3` coincides; `0→2`, `2→0` and `6→1` all fail), and for **all 4** local
torch indices — but that last row is an **accident of this allocation**: the visible module set
happens to be the contiguous `0,1,2,3`, so the table *is* the identity. On the previous allocation
(`0,3,6,7`) the same row read `0→0, 1→3, 2→6, 3→7` and failed for **3 of 4**. **Do not read the
current row as a law; read `HABANA_VISIBLE_MODULES`.**

**Note which rows are measured and which is derived, because the tags differ.** The first and third
rows are **MEASURED** — the `-Q` read-out, and the module table with its acquire probe
(`Archive/by-date/2026-10-15/ordpost2/` — `REPORT.md`'s "Device, before and after" table). The
**middle row is DERIVED, not measured on this box**: a local index *is* an offset into
`HABANA_VISIBLE_MODULES`, so with the table `0,1,2,3` the row is the identity **by construction** —
which is exactly why it must not be read as a law, and why the previous allocation's `0→0, 1→3, 2→6,
3→7` is that row's real shape.

Derive a local index from `HABANA_VISIBLE_MODULES`, **never** from `hl-smi -L`'s accel order (which
here is `2, 0, 3, 1` — the module ids listed in `hl-smi` index order, and neither of the other two
rows). The **fourth** numbering — the default per-AIP row number — is the one to be careful with,
because it is the one you read by eye: on this box the default view numbers its four rows `0..3` while
`-Q index` reads `0, 2, 3, 6`, so **the row you are looking at is not the index you need**.
**MEASURED** *(the default view verbatim:
`Archive/by-date/2026-10-15/ordpost2/hl-smi-pre-ordpost2b.txt`; the mapping: `REPORT.md` as above.)*

**The index numbering is not stable across a reboot; the module ids are — and we have now measured
that, on the card rather than on the numbering.** The vendor's public multi-container page states it:
*"…`HABANA_VISIBLE_MODULES` when the Docker container is started. The device index set with
`HABANA_VISIBLE_DEVICES` may change after a system reboot or driver reinstallation, while module ID set
with `HABANA_VISIBLE_MODULES` remains the same."* *(the vendor's
`Multiple_Dockers_each_with_Single_Workload` page:
`https://docs.habana.ai/en/latest/Orchestration/Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html`
— fetched 2026-10-15, HTTP 200, 81,909 bytes, sha256
`294b68dc8596f654a76b8151233b72c7dee55d9e583dbcc675261b8fca6dd344`.)* Across this box's 2026-10-15
reboot (with a driver reinstall, 1.24.1 → 1.24.2), **four of the four cards visible both before and
after kept their module ids exactly** — while the index map moved: host index `0` was module `0` before
and is module `2` now, and the host map now reads `0→2, 1→6, 2→0, 3→3, 4→7, 5→4, 6→1, 7→5`.
**MEASURED, and scoped — and the scope only parses once the two dumps are named.** The *before* read
is our container's **4-card `hl-smi -L`** (whose per-AIP blocks report Serial Number and Module ID);
the *after* read is the host's **8-card `hl-smi -Q index,module_id,serial,uuid -f csv`**. So the
comparison covers the **four** cards our container could see in both — not the host's eight, and not
two (the container's own 4-card view is the *before*; it is not a second container dump). Both dumps,
the post-reboot map and the by-serial comparison are in
`Archive/by-date/2026-10-15/device-host-map/HOST-AND-CONTAINER-DEVICE-MAP.md`.

**Reconcile two `hl-smi` dumps by SERIAL — not by index, and not by module id.** `hl-smi -Q` exposes a
per-card **serial** (and **UUID**), and those are the only fields that say whether two dumps describe
*the same card*:

```
hl-smi -Q index,module_id,serial,uuid -f csv,nounits
```

A card present in both dumps can then be checked for **which of the two numberings moved** — its
module id (identity) or its index (position). Reading the module *set* or the index instead is how
"the same card, renumbered" gets mistaken for "a different card": in a dump that does not carry the
serial, the two look identical. **MEASURED**

> **A module is acquired EXCLUSIVELY — one module, one process.** When two processes request the
> same physical module, the second **hard-fails at device-acquire** with
> `RuntimeError: synStatus=8 [Device not found] Device acquire failed.` (exit 3). There is **no
> silent sharing and no state in which both run**: a 4-way arm requesting one module for all four
> gave **exactly one winner and three identical failures**, and a **12-second-staggered** arm still
> failed while the other modules sat free. **⇒ a pin set must be a PARTITION — one distinct module
> per job.** **MEASURED** *(Archive/by-date/2026-10-11/hls_module_id/REPORT.md §4.3–§4.5.)*

### 3.7.2 Setting `HABANA_VISIBLE_DEVICES` inside a container that already limited you is INERT — and confusing

**Check first, then decide.** On a bare host you will see 8 AIPs (untested here — we have only ever
run in the 4-AIP container); the operative advice is to pin. **In a container that has already
limited you to 4, the four visible AIPs are already yours — setting `HABANA_VISIBLE_DEVICES` in
here has NO effect** (nothing in the image reads it; it is a `habanalabs-container-runtime` input
consumed **outside** the container, and the container's `--device` flags are what actually isolate
the cards). Do not rely on it, and do not try to pick global devices. Use relative numbering.
**INFERRED** *(whole-image grep + disassembly of the one device-acquire site:
`Archive/by-date/2026-10-11/hpu_isolation/REPORT.md` §2.3.)*

On this 4-AIP box the container's allocation is **`HABANA_VISIBLE_MODULES=0,1,2,3`** — the host
indices `2,6,0,3`, i.e. the same four cards (§3.7.1) — a clean dense local view over a physical
subset. **The module set is box-specific: read it, do not assume it.** That sentence is the durable
one; the *value* moves — **four times** on this project, and the history is two paragraphs down.
**Every one of the four is a real box's set**, which is the point: the value is not a fact about
Gaudi, it is a fact about the box in front of you. **SUPERSEDED 2026-10-15** at the value; the rule
did not move. That is why 4-device runs "just work" there — and why the check below is not optional.
**MEASURED** *(`Archive/by-date/2026-10-15/ordpost2/` — `REPORT.md`'s "Device, before and after"
table and the `hl-smi` snapshots.)*

⚠ **Then check the set against the vendor's recommended combinations — because a set that is merely
unusual is a set you will end up debugging.** Read the page precisely, because the distinction is
load-bearing: *"only 2-Gaudi and 4-Gaudi scenarios are supported. **It is highly recommended** to set
module ID in `HABANA_VISIBLE_MODULES` using the combinations listed below: 2-Gaudi: `0,1`, `2,3`,
`4,5` or `6,7` — 4-Gaudi: `0,1,2,3` or `4,5,6,7`"* — and *"Avoid mounting the same index on multiple
containers."* ⇒ **"supported" is the vendor's word for the *count* of processors; the combinations are
a *recommendation*.** The lists are **module IDs**: **2-Gaudi = `0,1`, `2,3`, `4,5` or `6,7`; 4-Gaudi
= `0,1,2,3` or `4,5,6,7`.** And the page's **own first example is non-adjacent** — it mounts four
processors with module IDs `5,4,7,6` — so a non-adjacent set is not forbidden by the page; it is
simply **outside the recommended list**. *(Same vendor page and fetch as §3.7.1.)*

**Our own container's allocation, and it has moved four times:** `4,5,6,7` — which **is** one of the
two recommended 4-Gaudi combinations — until **2026-10-07** (archived as `4,5,6,7` on 2026-09-26 and
2026-10-04; the first `0,3,6,7` record is 2026-10-07); then `0,3,6,7` from 2026-10-07, then `0,2,3,6`
after the 2026-10-15 driver bump, and now `0,1,2,3`. **So the recommended regime is the LONG one and
the non-recommended regime is the short one: the switch is at 2026-10-07, ~8 weeks after this project's
first archived work (2026-08-13), and the non-recommended sets are 8 days old.** The HPU measurements
this manual reports therefore fall in **both** regimes, and its CPU-only figures have no topology at
all. A reader who copies the allocation published in this section builds a TP4 run on a set **outside**
the recommended combinations; **this manual cannot tell them what that costs, because we have not
measured it.** *(The September `4,5,6,7` runs are the natural comparison for anyone who wants the
number, but they are code-state-confounded.)* **⇒ For any collective or interconnect figure, check
which regime it came from before attributing anything to the model.** **ASSUMED (vendor
documentation)** for the recommendation — the quotation above is the citation — and **MEASURED** for our
own allocations: `Archive/by-date/2026-09-26/runA.env` and `runB.env` (the `4,5,6,7` era),
`Archive/by-date/2026-10-04/old-notes/picking-hpus.md` (the same set, as *"physical set = host modules
4-7"*), `Archive/by-date/2026-10-15/device-host-map/HOST-AND-CONTAINER-DEVICE-MAP.md` **§2** (the
pre-reboot block, which is the `0,3,6,7` record) and **§3** (the post-reboot `0,1,2,3` map), and
`Archive/by-date/2026-10-15/ordpost2/REPORT-01-device-blocked.md` (the `0,2,3,6` state).

**Check before you assume:**

```
hl-smi -L                                   # 8 AIPs on a host, 4 in a limited container
hl-smi -Q index,module_id,memory.used -f csv,nounits   # the only honest placement read-out
hl-smi -Q index,module_id,serial,uuid -f csv,nounits   # ... and the serial: "same card" vs "different card"
env | grep -iE "HABANA_VISIBLE"
python -c "import torch, torch_hpu; print(torch.hpu.device_count())"
```

**An idle device reads ~768 MiB used; much higher means a process holds it.** **MEASURED**

### 3.7.3 Pinning a process

**`HLS_MODULE_ID` is the primitive.** It names a **physical module id**, not a local index.

- **Export it to pin a process:** `HLS_MODULE_ID=6 ./venv/bin/python harness.py`. This is the
  out-of-process pin, and the only one **we have** that works for a job whose launcher must set the
  module *inside* the worker. **MEASURED** *(A second in-worker path exists in the image —
  `hccl._setup_module_id` writes the variable from `HABANA_VISIBLE_MODULES[local_rank]` on HCCL
  import — but it keys off `LOCAL_RANK`, which a plain launcher does not set, and the `LOCAL_RANK`
  injection route was tried and reverted. **INFERRED** (code analysis).)*
- **It is read LAZILY**, at device-acquire time — so it works when set *after* `import torch,
  torch_hpu`, as long as it precedes the first HPU tensor. A non-numeric value fails with
  **`ValueError: stoul` at allocation**, not at import. **MEASURED**
- **`torch.hpu.set_device(i)` as the FIRST HPU call** is the in-process wrapper: it writes
  `HLS_MODULE_ID = HABANA_VISIBLE_MODULES[i]`. In one 4-AIP container — **the `0,3,6,7` allocation of
  §3.7.1, not the current one** — the observed mapping was **0→0, 1→3, 2→6, 3→7**. Called after a
  device is already selected it is **silently ignored**. **MEASURED**
- **It does not override an existing pin — *except* when that pin is `0`.** An existing
  `HLS_MODULE_ID` wins, so keep the environment pin and any `set_device(i)` **consistent**
  (**INFERRED** — code analysis of `hpu/__init__.py:352-367`, `:381`; not executed). The exception is
  `HLS_MODULE_ID=0`: the guard is keyed on the Python helper `_get_module_id_from_environ()`, whose
  `if not device_index: device_index = -1` (`hpu/_utils.py:68`) maps `"0"` to `-1`, so the guard
  does **not** fire and `set_device(i)` **overwrites** the pin — pre-set `0` plus `set_device(3)`
  leaves the environment reading `"7"`. **MEASURED** *(`hpu_pinning/mechanism2_run.log` M3b.)* Note
  the **driver** still honours `0` — it pins to module 0 (`set_device(0)` writes `"0"` and lands on
  module 0; `hls_module_id` §4.7, `setdev_0/`); it is only this Python-side helper that treats `0`
  as `-1`. Never assume `set_device` is harmless when the pin is `0`.
- **The pin set must be a partition.** Because acquisition is exclusive (§3.7.1), N jobs need N
  *distinct* module ids; giving two jobs the same id fails the second one. **MEASURED**
- **A multi-rank job needs one distinct module per rank.** One shared `HLS_MODULE_ID` for the whole
  job **breaks** it — **3 of 4 processes died**, measured on four bare `torch_hpu` processes; the
  environment is inherited by spawned workers, so the same value reaches every rank (**INFERRED** —
  code analysis, not executed under a spawn). Distinct per-rank values work (four processes, one
  per module → `0,3,6,7`). **MEASURED** *(the bare-process arms; **not** a `vllm serve` job — see
  §3.7.6.)*
- **No self-report API works** — `current_device`, `current_device_index` and `tensor.device` all
  report **0**, even on processes physically on modules 3, 6 and 7. **Read
  `os.environ["HLS_MODULE_ID"]`** instead — and remember it is only meaningful if something *set*
  it, so **unset means "not pinned"**, not "module 0". (A literal `0` in the environment *is* a pin,
  to module 0 — see the `set_device` bullet above.) **MEASURED**

**What unpinned does — and it is not "module 0".** With `HLS_MODULE_ID` unset the acquire path is
`synDeviceAcquireByDeviceType`, i.e. **first free module, skipping modules already held** (on this
box: module `0`, then module `3`). Whether that order comes from the module table or from the
driver's own enumeration is **not settled** — the two coincide here, so this arm cannot separate
them (§3.9, item 16(a)). Two unpinned processes landed on *different* modules, and the pairing is
**not stable run to run** — it is a race. So unpinned concurrency already spreads; `HLS_MODULE_ID`
buys **determinism**, not collision avoidance. **MEASURED**

**`synStatus=8` is NOT diagnostic.** The identical string arises from **at least three** causes: a
module id that is **not visible** (`HLS_MODULE_ID=1` where the table is `0,3,6,7`), a module id that
is **already held** (above), and the **transient simultaneous-launch race** (3 of 5 at zero
stagger). The first two are **MEASURED here**; the launch race is recorded in the repo but was
**deliberately excluded** from the pinning experiment (every lock arm was staggered), so it is not
re-confirmed here. Never read it as "device taken", and never as "the pin failed". **Retry, then
check the process table.**

### 3.7.4 Concurrent processes do not perturb timing much — a 4-concurrent schedule is safe

- **< ~2 % device, < ~5 % host** perturbation from concurrency. **MEASURED**
- **Two gotchas**: (a) **stagger launches by ~2 s** — a simultaneous start lost a process to
  `synStatus=8` in **3 of 5** attempts; (b) **never sample `hl-smi` during startup or teardown** — it
  shows a transient (98304 MiB). **MEASURED**
- **You cannot share a module.** Acquisition is exclusive (§3.7.1), so "contention" between HPU
  processes on one module does not exist. A module reading above the ~768 MiB baseline has a process
  **holding** it — ordinarily **another HPU job** (another agent's, another user's, or your own),
  and only secondarily a non-HPU holder or a stale/interrupted run (an interrupted sandbox held a
  module for ~35 s). **MEASURED**

### 3.7.5 Worker-process hygiene

- **Let the platform choose the worker launch method.** On HPU the multiprocessing method is
  auto-selected (`spawn`, because `fork` can hang on exit). **Forcing `fork` is a known dead end.**
  **MEASURED**
- **Workers live in a separate process group from the driver leader.** After killing the leader's
  process group you must also kill the worker PIDs explicitly, **or they orphan**. **MEASURED**
- **Never `pkill -f` on a shared machine** — it kills other users' servers. Track the PID you spawned.
  **MEASURED** (and it is a shared-server rule, not just a performance rule)

### 3.7.6 What this section does not know

- **How a real `vllm serve` pins its workers end-to-end is UNKNOWN.** vllm-gaudi never references
  `HLS_MODULE_ID` (grep, whole tree) and calls `torch.accelerator.set_device_index(local_rank)`,
  which does **not** route to the acquire path. So a TP=N job spreads **by the by-type race**, not
  by a mechanism — our TP=4 runs complete, which under the exclusivity rule implies four distinct
  modules, but **no run read their placement back**. What *is* measured is the env-var path on a
  bare `torch_hpu` process. The **UNKNOWN register (§3.9, item 16)** carries this with the
  experiment that would settle it.
- **Whether the module-id table also restricts the driver's unpinned choice is UNKNOWN.** The binary
  analysis finds no `HABANA_VISIBLE_MODULES` string in `libhl-thunk.so`, and the PT-bridge's by-type
  function does not consult it (**INFERRED**), but the driver/thunk-level candidate set is left open
  (`hpu_isolation` §10.2). The two-process arm we *have* (two unpinned processes, watching which
  modules they take) used the container's **full** table, so it cannot separate the table from the
  container's `--device` cgroup: both processes land inside the table either way. A **discriminating**
  arm is still open. *(§3.9, item 16(a); Chapter 1 §1.12.)*
- **What `torch.device("hpu", N)` does when N ≠ the pinned module's index** — **MEASURED for an
  unpinned process**: the index is ignored, the bridge warns (`"hpu:X" notation is not supported
  ...`), and every allocation lands on the process's one acquired module (`hls_module_id` §1.2). The
  *pinned* variant is untested. Until it is settled, keep the environment pin and any
  `set_device(i)` **consistent**.
- **The `~2 %` device / `~5 %` host figures above (§3.7.4) are *better* controlled than they
  might read — but they are a bound, not a zero.** Each arm ran `torch.hpu.set_device(idx)` as
  its first HPU call, so it was **pinned**, and that pin is read back from the arm's own
  `HLS_MODULE_ID` (`0`, `3`, `6`, `7` for idx `0..3`). The run also samples `hl-smi` (`hl-smi
  -Q module_id,memory.used,utilization.aip -f csv,nounits`, before / during / after, around
  rather than during the arm's own run), which resolves the **neighbour**'s placement — the
  dev-heavy process on idx 3 reads as module **7** with its own 8389 MiB / 88 % footprint —
  but **not the arm's own**: module 0 reads the 768 MiB idle baseline in the samples that
  bracket the arm's run, and the single 98304 MiB reading on it is the init/teardown transient
  of §3.7.4 gotcha (b), not a steady footprint. What is *not* controlled is the box's other
  tenants: the run's own drift is **±5–7 %**, which is the dominant error term. **MEASURED**
  *(`Archive/by-date/2026-10-11/hpu_pinning/`: `timing_arm.py:33`, `timing_run.log`,
  `run_timing.sh`.)*

---

## 3.8 The dead-end register

Every entry below cost real time. They are recorded so a port does not re-walk them. **This register is
as valuable as the wins.**

| # | dead end | why it is dead | tag |
|---|---|---|---|
| 1 | **"Remove the `cat`"** | The graph compiler inserts `concat_logical` + `memcpy_dma` for GEMM operand staging regardless; a GEMM recipe contains DMA layout ops whether or not the user's cat is there. And the IR cannot decide critical-path membership. | **MEASURED** |
| 2 | **A 256-entry LUT for the fp4 decode** | **Corrected 2026-10-12 — the dead end was the *diagnosis*, not the LUT.** The narrow observation was right **for the 4-argument call form** (equivalently the 6-argument form with `predicate = 1`) we tested: `v_i8_lookup_1c(idx, 0, 0, tbl, 1, 0)` emits `lookup_1c BV8_0 %V1, %V0, 0x0` — **no table operand** — so the caller's `lut` tensor is never loaded and four different tables give byte-identical output. **The generalisation to "not expressible on this ISA" was wrong.** What made a call-form bug look like an ISA limit: the kernel left the predicate at the header's **default of `1`** — a **constant predicate** — so the table load was **dead-code-eliminated**: the listing of `src/custom_ops/fp4_dequant_lut/kernels/gaudi2/fp4_dequant_lut_gaudi2.c` contains **no load of tensor `0x2` at all** (`rg 'ld_tnsr'` → only `0x1` and `0x0`), i.e. it was measuring a **no-LUT kernel** — **and the device run confirmed the symptom** (four byte-identical tables, `lut != base`, `CHECK: FAIL` in `Archive/by-date/2026-10-09/w1_lut_timing/logs/check.log`, whose archived kernel copy is byte-identical to the tree's at md5 `044e0a8d20437981fec6c416638d5db1`), **while the diagnosis went to the ISA instead of the call form.** The 6-argument form `v_i8_lookup_1c(idx, 0, SW_LUT_PTR, tbl, 1, 0)` — which needs `set_lut_256(2)` to write tensor 2's base into `LUT_FUNC256_BASE2_ADDR_LO/HI` — emits `lookup_1c BV8_0 lut_ptr …` and **does** read the table: with a `0..255` table and an index that is provably the column, the device returns **exactly the 256 table bytes** (256 distinct values, multiset identity), and against the shipped kernel the corrected arm is **bit-exact (0 differing bytes / 50,331,648 at the largest production shape)**. `lookup_1c` is a 128-lane instruction, so a 256-lane lookup is **2 instructions**; the corrected arm costs the **same 22 instructions/iteration** as the broken one and is **faster** — at `[4096,2048]`, span median over 60 launches, **32.16 µs vs `arith2`'s 50.51 µs**. ⇒ **This is not a dead end; it is a live, faster option.** *(Scope — and it is wider than "the lazy path is untested": (i) the standalone GC/recipe harness only, so the engine's lazy path is still untested; (ii) **the table covers `k ∈ [−8,7]` and the `min/max` clamp wraps silently outside it** — the `0 / 50,331,648` cell above is production shapes at **`k ∈ [−8,−1]`**, and the out-of-domain row of the same test is **NOT exact, max byte diff 6…130** — **and this checkpoint reaches `k = −9`** (byte 118, **1,070 groups**), so the arm as measured **wraps to the `k = −8` row on those groups**, a **load-time domain guard is mandatory**, and as things stand **that guard would not pass**; (iii) **the table is a new op input** — the signature becomes `(blocks, kexp, lut) -> (o0, o1)`. Device run and timing: `Archive/by-date/2026-10-09/lut_device/`; the domain limit is that README's §2 table and §6 condition 2; `k = −9` / 1,070 groups: §3.4.5 above and `Archive/by-date/2026-10-12/fp4-op-contract/REPORT.md` §3.2.)* | **MEASURED** (the narrow observation, and its falsification — our own compile listing + our own device run) |
| 3 | **Re-expressing the dequant in native torch ops** | Bit-exact, and **21.7× slower** — 76 % of the cost is the `torch.gather` **int32 index** (a 100 MB index for a 25 MB payload). No chain of elementwise ops matches one fused memory-bound pass. | **MEASURED** |
| 4 | **Pre-dequantizing experts at load time** | Doubles the weight bytes: **36.8 → 71.4 GB/rank** on a 96 GB card with 12.3 GB free. HBM-blocked. | **NOTE-CLAIM** |
| 5 | **Two HPU streams for cross-engine overlap** | `CROSS = 0.0000` **with a valid precondition** (119.1 ops outstanding, both queues provably loaded). **CORRECTED 2026-10-12 — the precondition is superseded as this row's cited evidence; there are *two* distinct two-stream arms, so this row's arm was *not* the null test; and the "host cost explodes" clause has been moved to the arm it belongs to.** The "0/40 ops in flight" reading is the **E5 audit's `07_streams_indep`** (`Archive/by-date/2026-10-12/e5_method_audit/AUDIT.md` §6 — host-bound, 10.9 % device busy, 0.3 ops outstanding at `H_end`), a **null test**. But the arm *this row* quotes is the **`t2_graph_overlap` arm `t1_02_streams_tpc_mme`** (`Archive/by-date/2026-10-10/t2_graph_overlap/`, README §5), which is **device-bound** — 36 ops/step, **119.1 ops outstanding at `H_end`, 97.8 % device busy, both queues provably loaded** (`precondition_ok = True`) — so the "119.1 / both queues loaded" precondition was **valid for its own arm**, and the 0/40 null belongs to a *different* arm. A **device-bound** re-run (`str_bound`) adds a **third** such arm: `CROSS = 0.0000` **exactly, both runs**, **18.5 ops outstanding / 83.5 % device busy** (run 1) and **55.9 / 97.0 %** (run 2). The dead end therefore stands on **two** device-bound arms. **The "host cost explodes" clause is *not* supported by this row and is now attributed to arm 7.** *(Corrected 2026-10-12 — the clause's provenance is the E5 audit's `07_streams_indep`, the **host-bound** arm: 10.9 % device busy, 0.3 ops outstanding at `H_end`. It is overstated even there: the ~2 ms stream-switch cost that motivated it is profiler **inclusive self-time summed over nested frames**, not a real cost — two `torch.hpu.set_stream` calls measure **~10–24 µs** across **eight** archived calibrations — 9.8 µs in `t2_graph_overlap` (`Archive/by-date/2026-10-10/t2_graph_overlap/raw/calibration.json`), 12.6 / 13.1 / 21.5 µs in the three `recipe_dump` runs, and 22.4 / 22.6 / 22.8 / 23.9 µs in the four `overlap_dma_probe` runs (`Archive/by-date/2026-10-11/{recipe_dump,overlap_dma_probe}/raw*/calibration.json`) — so **9.8 µs is a 2.3× low outlier against a ~22–24 µs mode**. All eight share **one** timing method (the archived `t2_graph_overlap.py` `bench()`, `n=20, w=3`, imported by every driver that produced one of the eight), so the spread is **run-to-run, not methodological**. This row's own arm cannot carry it: `t1_02_streams_tpc_mme`'s record reads `host_issue_span_ms = 131.07` against `device_span_ms = 156.83`, `host_bound_unprofiled = False`.)** *(§2.12 item 6; `Archive/by-date/2026-10-12/engine-overlap/REPORT.md` §3.1.)* | **MEASURED** (the re-run's precondition, asserted and passing) |
| 6 | **"Avoid custom ops" as a heuristic** | **Refuted by the model's own op.** Three arms held customness fixed and flipped the outcome purely on **submission count**: custom op → 0.2217 at 1 submission; custom op → 0.0000 at 2; native op → 0.4179 at 1. The discriminator is the submission count. | **MEASURED** |
| 7 | **Region coalescing alone** | Grouping 4 layers into one compiled unit **moved no operational count** — the *parameter*, not the region boundary, sets the instance count. | **MEASURED** |
| 8 | **Launch count / sync count / recipe count as the lever** | r² 0.15 / 0.195 / 0.066, **all with the wrong sign**. | **MEASURED** |
| 9 | **Reaching the 48 MiB SRAM from your own code** | Every route is closed. `synDeviceMalloc` at an SRAM address returns **`synSuccess` and relocates to DRAM**; `MEMORY_ATTRIBUTE_SRAM` is `// currently not supported`; a TPC kernel **cannot name the address** (32-bit pointers — a 64-bit constant cast to a pointer compiles to `ld_l`/`st_l`, i.e. **local memory**, and *appears to work*); `AuxTensor.sramAlloc` is a **dead field with no readers**. Placement is the compiler's decision, not yours. **Chapter 7** is the whole map. | **MEASURED** |
| 10 | **A node-count census as a cost model** | Metadata nodes (`Reshape`, `logical_slice_fwd`, `Split`, `Concatenate`, …) are numerous and **absent from the device trace**; conversely a family of 1–3 µs integer nodes can be **30 % of a step** while never reaching a per-node ranking. Node counts have **no device-time implication in either direction** — rank by **engine identity × count × mean duration**. §3.6.4. | **MEASURED** |
| 9 | **Optimising for "the step is launch/serialization-bound"** | **Two readings exist, and they point at opposite levers — so name both.** *(i) **launch/serialization-bound**: device idle ~80 %, host busy through most of it, ~573 graph launches and ~1048 host syncs per step.* *(ii) **device-occupancy-bound**: the step is ~76 % device-occupancy and the wall tracks device busy.* **These describe different CODE STATES, not two measurements of one** — the 573/1048 numbers are a 2026-09-19/20 `with_stack=True` state ~3× slower than the current one, and the launch/sync/recipe **counts** fail to predict the wall on the current corpus (wrong sign, r² ≤ 0.2) while `merged` fits at r² 0.9999. A later archived fit re-tests the launch-side model directly and **refutes it again** (`Archive/by-date/2026-10-11/node_law/`: wall = k × node-executions gives R² **0.346** through the origin with a per-run `k` spread of **171 %**, against wall = 21.703 + 0.99713 × `merged` at R² **0.99995**). **⇒ Do not optimise for launch count.** But if you are on a code state with the *other* symptom, the other reading is the right one — check which regime you are in before choosing a lever. | **MEASURED** |
| 10 | **`torch.stack([a, a], dim=-1)` — the same tensor twice** | Costs **22,532 µs vs 301 µs** for two *distinct* tensors — **75×** — on a 50 MB output. It silently destroyed a component benchmark (read 22,500 µs inside a 4,300 µs whole-op measurement). | **MEASURED** |
| 11 | **Profiling with CPU activity enabled** | Inflates the trace's host span **~3×** and collapsed a queue-depth metric from 23.3 to 1.67, manufacturing a false empty queue. **Use HPU-activity-only profiling.** | **MEASURED** |
| 12 | **`**`-globbing a dump directory** | Silently **skips the hidden `.graph_dumps/`** directory — one recon saw **24 of 1,141** files on its first pass. Use `os.walk`/`find`. | **MEASURED** |
| 13 | **Counting "device ops" as a work-unit count** | A device event is a **replication count** — one event per (launch × TPC/DMA engine) — so kernel events ÷ recipe launches was **544 / 569 / 971** across three traces. **Not even a fixed multiple.** | **MEASURED** |

---

## 3.9 The UNKNOWN register

Stated, not omitted. Each has a note on what would settle it. **Six entries have been closed since
the first draft — items 1, 2, 4, 10, 14 and 21 — and all are kept in place, marked CLOSED, so the
resolutions are visible rather than silently deleted.** *(Items 10 and 14 state the fp8 format as
`e4m3fn` codes, bias 7. That is the **configured default** — the bias is **selectable**; see the bias
subsection in §3.4.1.)*

**Items 17–20 are new, opened 2026-10-12 by the vendor-MXFP4-op conformance probe (§3.4.7) — they are
additions, not closures.** **Item 21, opened 2026-10-13 by the fp4 LUT promotion (§3.4.8), is now
CLOSED — the whole-model TP=4 run resolved it (below).** **Item 22 is new, opened 2026-10-13 by the
two-arm compile gate (§3.6.5), and it is the *bound on that gate's result* rather than a new question.**
**Item 23 is new, opened 2026-10-21 from a parallel port's batched-decode notes (§3.1.5), and
HALF-CLOSED 2026-10-24 by that port's own scaling study — its routed-expert mechanism is refuted, and
what replaces it is measured; what remains open is a `merged`-per-token curve on *our* arm.**
They are registered here rather than left inside §3.4.7 so that a reader who
consults this register to find out what we do *not* know gets a complete answer.

> **The habit that keeps this honest, and it is cheap.** When a later section **corrects or supersedes**
> a pattern, the **earlier** section that **prescribes** it must be re-checked **in the same pass**. A
> chapter that says *"do X"* in §A and *"X is wrong"* in §B is **worse than one that says nothing**,
> because the reader stops at §A. A mechanical check — *extract every `§X.Y` reference and assert the
> target heading exists* — found four broken references in this chapter, one of them **§3.4.4
> prescribing exactly the decode that §3.4.5 says is wrong**. Run that check after any edit — and
> **scope it to refs whose preceding token is not a path**: a document-qualified ref written bare
> (`` `REPORT.md` §2.1 ``) resolves against *this* chapter's §2.1 and **passes silently against the
> wrong target**, while a bare `§7` or `§4.4` in the same style **false-fails**. Both are the check
> lying about what it checked, in opposite directions.

| # | UNKNOWN | what would settle it |
|---|---|---|
| ~~1~~ | ~~**SRAM size.** The only figure anywhere ("48 MB") is marked **unverified by its own erratum**; do not import it.~~ **CLOSED — RESOLVED 2026-10-11. The SRAM is 48 MiB on Gaudi 2** (`SRAM_SIZE = 0x3000000`); the "unverified" marking was right, and the figure survived checking. What *is* measured is unchanged — SRAM staging exists and is visible in the compiler's own output (`"allocation": "SRAM"`). *(Source, public and checkable: the upstream Linux kernel's habanalabs driver — `drivers/accel/habanalabs/include/gaudi2/gaudi2.h:41` (`SRAM_SIZE`); line numbers move between kernel releases, so check the symbol, not the line.)* **Full statement: Chapter 1 §1.5.** | — **done** |
| ~~2~~ | ~~**L1/L2 cache sizes and behaviour.**~~ **CLOSED — RESOLVED 2026-10-11. There is NO L1/L2 cache.** The public vendor documentation is consistent: the TPC's memory hierarchy is **HBM (96 GB) → 48 MB SRAM (on-die, shared, compiler-scheduled staging) → per-TPC local memory (1 KB scalar + 80 KB vector per core, private, kernel-programmer-managed)**, and **no cache level exists between them**. The TPC local memory is *not* the SRAM and is *not* a cache. *(the vendor's TPC user guide: `docs.habana.ai/en/latest/TPC/TPC_User_Guide/Processor_Architectural_Overview.html`.)* **⇒ A bandwidth model must not assume reuse from a cache: reuse has to be engineered into the dataflow (tiling into local memory / SRAM) or it does not happen.** *Substantive material is routed to Chapter 1; this entry is closed here so the register is not stale.* | — **done** |
| 3 | **Whether the TPC cost model consumes the *general* HAL reader** (2000 MHz) rather than an engine-specific one (1650 MHz). If it does, it is **~21 % optimistic**. The MME has its **own** reader at 1650 MHz and is accurate to 0.5 %. **The DMA cost model is EXCLUDED — it makes no clock call at all** (no vtable slot-7 call in its constructor or in its duration/cost functions), so the clock question does not apply to it; the **TPC** cost model's clock remains **UNCHECKED** because its constructor is not exported. | Disassemble the TPC cost model's clock path the way the MME's was done. *(MME clock and family table: `Archive/by-date/2026-10-11/synapse_cpp/GC_COST_MODEL_CLOCK.md` §8; the DMA exclusion and the TPC absence: same file, §10.)* |
| ~~4~~ | ~~**The 512 MiB C++ bench point (1,646 GB/s)** — an outlier below the measured 256 MB and 1 GB points.~~ **CLOSED — RESOLVED 2026-10-11. The point is RETRACTED; both falsifiers ran.** **(a)** the bench **syncs per repetition**, so its wall time carries the launch+sync overhead (device union **530.5 µs** vs bench median **622.1 µs**, the ~90 µs gap being exactly that overhead); **(b)** the published number is an **arithmetic error** — the true quotient is **1,726 GB/s**, and 1,646 corresponds to **652.3 µs**, which is no bench statistic at all. The **corrected point is 2,023 GB/s** (flat 1-D 2,022.9 / shaped 2-D 2,023.5 — the **2-D hypothesis is DEAD**, 0.03 %) and **the curve is monotone**. *Artifact: `Archive/by-date/2026-10-11/dma_open_items/OPEN-ITEMS-c2-and-512MiB.md` §2; full treatment, including how the wrong point survived four passes, in **Chapter 1**.* | — **done** |
| 5 | **Whether the engine's cross-engine intersections are real concurrency** (§3.6.3). **NARROWED 2026-10-12 — still OPEN, in both directions.** The deciding quantity is now measured rather than flagged: the ceiling on the co-residency inflation is **`p ≤ 2.200`**, the `p` **bracketed at** the engine's own TPC:DMA ratio is **2.23–2.28**, and `O/M = 1.02` — i.e. the intersection sits **just over the ceiling `O = M`** (≈100 % of available DMA work hidden, against ≈45 % of the measured `hpu_op` DMA union), so it **misses the `O ≤ M` test by 1–4 %** at the measured `p` and is **neither established as real nor dismissed as an artefact**. The knife-edge is quantified; the verdict is not changed. | **An engine-side `DUMP_POST_GRAPHS` on one W0 decode step** — it gives `M(W0)` directly and turns the extrapolation into a measurement. **NOT RUN** (needs a vLLM run). The deciding arm that was *missing* has now run and varies the quantity; what is still missing is the **engine's own `M`**. |
| 6 | **The ~21 ms host constant's sync-wait vs host-compute split.** | A runtime change so `synchronizeEvent` carries the calling OS tid, or a `record_function` at the point the C++ runtime issues the sync. |
| 7 | **The per-recipe / per-step / per-queue split of the DMA window's fixed component.** Every sweep arm held nodes, recipes and queues constant, so it is **confounded**. **PARTIALLY NARROWED 2026-10-12.** The 2-D sweep does not vary recipes, but it settles two of the three axes: the window is **invariant to DMA queue load** (per-node `runs/step = 1.00` — one node execution is one contiguous window, marker fan-out 25×–130×) and it is **sensitive to a TPC co-tenant as a step, not a proportion** (the same 3072² node measures **8.89 µs** with no co-tenant (identical in both runs) and **22.4 µs** with one TPC mul (**run 2; run 1 measures 23.12 µs**), yet is **flat** across 1 / 2 / 4 TPC muls at fixed DMA). So most of the "fixed component" is a **first-co-tenant step**. | A sweep that varies **recipes** independently — that axis is still confounded (queue load and co-tenant are now partly covered; see §2.7's 2026-10-12 update and §2.12 item 5). |
| 8 | **Composition for unequal cross-engine groups, or > 2 engines in one recipe.** **HALF-CLOSED 2026-10-12 — the `> 2 engines` half is measured; the unequal half is not.** Three engines in one recipe, independent: `union(all) = 0.7400` against `Σ[kernel] = 1.1232` and `max(single) = 0.6380`, so **all three co-schedule** and the third engine costs only **+0.102 ms** of union (pairwise, **this arm's own record** — corrected 2026-10-12, this parenthetical previously read `TPC∥DMA 0.63–0.99, MME∥DMA 0.79–0.89`, which is not the 3-engine arm's range: TPC∥DMAk 0.632, TPC∥DMAh 0.978, MME∥DMAk 0.549, MME∥DMAh 0.873, so **MME∥DMA here is 0.55–0.87**; over the wider set (**the twelve arms of the source report's §3.2 `ov` table**, `Archive/by-date/2026-10-12/engine-overlap/REPORT.md`) the spans are TPC∥DMAh 0.464–0.994 and MME∥DMAh 0.787–0.963). **Unequal cross-engine groups remain unmeasured** — every arm is duration-matched by construction, so the condition-4 failure regime was never reproduced. | For the unequal half: an arm with a deliberately unequal pair (e.g. 4:1 durations). For the 3-engine half: a **bottleneck-varied** 3-engine recipe (all measured arms are independent bodies). |
| 9 | **Whether the three extra DMA tids (4073, 4082, 4083) are device or host.** Our reading splits an earlier "host PCIe" ruling on 4082/4083. | A targeted probe. |
| ~~10~~ | ~~**Which description of the fp8 top bin is literally correct** (§3.4.1).~~ **CLOSED — RESOLVED 2026-10-11.** **The device's fp8 is `e4m3fn` codes (bias 7) truncated at 240; it is NOT `e4m3fnuz`.** `torch.float8_e4m3fnuz` is **not a supported dtype on HPU** (`is_supported_type` rejects it), the device's own **encoder** emits bias-7 codes (1.0 → `0x38`, 240 → `0x77`), and `cast_from_fp8` matches the CPU `e4m3fn` table **242/256** vs `e4m3fnuz` **1/256**. So the "storage = fnuz, decode = fn" split does **not** apply — the device does not *write* fnuz codes either. `float8_e4m3fnuz` contributes exactly **one** useful number to this stack: the ceiling **240**. **Full statement, the poison-code set, and the guard rule: §3.4.1.** *Provenance: settled by a decode+encode sweep on one HPU (one disclosed 11-second device probe), because the archive could not answer it.* | — **done** |
| 11 | **Whether the ~55 MB per compiled instance scales with the instance count.** If it does not, the memory is in transient activation workspace or per-launch state, and the specialization/memory tradeoff is void. | De-guard one region and re-measure the post-warmup footprint. |
| 12 | **Whether the `cat` is on the critical path.** The IR cannot decide it (no unique node ids), and the one synthetic arm was inconclusive and changed the engine mix. | A GC-side post-graph dump from the real model, or an arm that holds the engine mix fixed. |
| 13 | **The exact tensor/line that introduces a data-dependent size into the graph** on our own port. Candidate identified, not pinned. | Per-region guard-key inspection. |
| ~~14~~ | ~~**Whether the fp8 reserved-bin rule is ENGINE-DEPENDENT** — i.e. whether the MME could decode the reserved bin differently from the TPC.~~ **CLOSED — RESOLVED 2026-10-11. NO: the rule is a property of the device, not of an engine.** **`MME == TPC byte-for-byte over all 256 codes: True`**, same 14-code mismatch set, measured **within one process on one code set** (so a within-run two-engine comparison, not two measurements days apart). Calibration passed first — **242/242 exact through the MME** — so the readout is a pure decode. **`e4m3fnuz` is excluded twice over on the MME**: it reads `0x76` as **224** (fnuz would be 112) and `0x80` as **0** (fnuz would be NaN). **⇒ the format statement needs no engine qualifier, and the guard's byte set `0x78–0x7E` / `0xF8–0xFE` is confirmed for BOTH engines.** *Mechanism worth keeping: the fp8 decode happens **inside the GEMM node** — a post-graph dump shows `MME GEMM ['hfloat8','hfloat8'] → ['bf16']` consuming fp8 directly with **no intervening cast**, so **no engine-attribution pass can ever see it** and would have reported "MME, no fp8 cast" and been silently wrong.* **Full statement: §3.4.1** (see also the closed item 10 above for the format itself). *Provenance: a second disclosed ~11-second device probe, with its own post-graph dump.* | — **done** |
| **15** | **What SELECTS the fp8 exponent bias in a given recipe, and whether it is readable back.** The bias is **selectable** (`{3, 7, 11, 15}`, default **7**) and is carried **per op in the H2D scale tensor** (§3.4.1), but *which* mechanism sets it for a given op, and whether a compiled recipe's chosen bias is **observable** from a trace or a dump, is **not established**. This matters because §3.4.1's poison set is stated **for the default bias** — a recipe that selects another bias has a **different reserved bin**. | A recipe/dump inspection for a bias field, or a decode readout on an arm known to select a non-default bias. *(**Opened** by the bias finding — recorded rather than left implicit, so the closures of items 10 and 14 are not read as "fp8 is fully understood".)* |
| **16** | **How a real `vllm serve` pins its workers end-to-end — and therefore whether a TP=N launch is deterministic or a race.** vllm-gaudi **never references `HLS_MODULE_ID`** (grep, whole tree) and calls `torch.accelerator.set_device_index(local_rank)`, which **does not route** to the device-acquire path (`HPUHooks` does not override `setCurrentDevice`). The measured per-rank mechanism (`HLS_MODULE_ID` distinct per rank) was proven on bare `torch_hpu` processes only; the vLLM integration was deliberately not run. Two further device-pinning questions sit in the same place: **(a)** whether the module-id table also restricts the driver's unpinned by-type choice (**UNKNOWN** — the PT-bridge does not consult it (**INFERRED**), but the driver/thunk level is not settled; `hpu_isolation` §10.2), and **(b)** what `torch.device("hpu", N)` does when N ≠ the pinned module's index (**MEASURED** for an unpinned process — the index is ignored and the bridge warns; only the *pinned* variant is untested). | Launch the **real TP=4 server** with and without `HLS_MODULE_ID` exported per rank, and read each worker's device off `hl-smi`'s process table / `hl-smi -Q index,module_id,memory.used -f csv,nounits`. For (a): a **discriminating** two-process arm — a per-job `HABANA_VISIBLE_MODULES` that *excludes* a cgroup-exposed module, with **no** pin, watching which modules the processes take (the two-process arm we already have used the container's *full* table, so it cannot separate the table from the cgroup). For (b): one probe that pins module 6 and then allocates on `hpu:1` / `hpu:2`. *(Opened 2026-10-12; sources: `Archive/by-date/2026-10-11/hls_module_id/REPORT.md` §8, `hpu_isolation/REPORT.md` §10.1–§10.2.)* |
| **17** | **The on-device semantics of the vendor MXFP4 op's GPT-SwiGLU / clamped path** — `bias_mxfp4_fused_weights` (`gpt_swiglu = true`, `mixture_of_experts_gen.cpp:1553`): the `limit`, the `alpha`, the `beta` / `(up+1)` term, and **the gate/up split layout**. **This is a different op and a different code path from the one we measured** — the non-bias `mxfp4_fused_weights` is `gpt_swiglu = false` (`:1441`) and its split is **measured as `halves`** (§3.4.7, row 3). The bias path's own split is **not** measured: a bridge test docstring (`tests/pytest_working/any_mode/test_hpu_mixture_of_experts_mxfp4.py:745`) says *interleaved*, while **vllm-gaudi**'s `swiglu_oai` reference (`vllm_gaudi/models/minimax_m3_sparse.py`) says split/first-half — a **doc-against-doc** disagreement we have not resolved. Do not read either document as evidence about the non-bias path. | A `bias_*` arm on a **non-degenerate** configuration (≥ 2 experts, random weights and biases — the vendor's own test shape). The probe's corner configuration (1 expert, `experts_min == experts_max`, identity `w12`, zero biases) **wedged the compute queue** with a 30 s timeout and was deliberately not retried. The cheapest first step is the **bf16** `bias_fused_weights` op, which traverses the same cguid GPT-SwiGLU lowering **without** the fp4 cast nodes, so it isolates whether the hang is fp4-specific. *(Opened 2026-10-12 by the vendor-op probe; §3.4.7; `Archive/by-date/2026-10-12/a1-probe/REPORT.md` §6 and `STATE_AFTER_HANG.md`.)* |
| **18** | **The vendor MXFP4 op's decode *implementation* — two things about it are measured behaviour only.** **(a)** Whether its byte-255 e8m0 scale is `inf` or `NaN`: even `(code 0, byte 255)` — a weight of exactly `0.0` — returns NaN, which requires the *scale* to be non-finite, and the op **cannot distinguish** the two (`0 × either = NaN`). **(b)** The **mechanism of its flush-to-zero**: byte 0 → scale `2^-127` (an fp32 subnormal) is flushed to `0.0` for every nibble, and a bf16-subnormal product flushes too — measured behaviour, unmeasured cause. | Kernel source, which we do not have. *(Contrast our own stack, which **refuses** `0xFF` at load: §3.4.5; the checkpoint histogram showing `0xFF` is unreachable here — 0 occurrences in 8,858,370,048 routed-expert bytes — is at `src/gates/step_e8m0_byte255_gate.py`.)* *(Opened 2026-10-12; §3.4.7; `Archive/by-date/2026-10-12/fp4-builtin-conformance/REPORT.md` §2.1 and `REPORT.md` §7 items 1–2.)* |
| **19** | **Multi-expert correctness of the vendor MXFP4 fused op.** The conformance probe used **one** expert. The vendor's own test uses **3 experts with `topk(2)`**, which is a **different contract**, and a naive all-zeros routing table gave garbage — so `experts_min` / `experts_max` with a real routing table is **not established**, and every axis in §3.4.7's table is single-expert. | The vendor test's own configuration (3 experts, `topk(2)`, random weights and biases) driven against a known-good reference. *(Opened 2026-10-12; §3.4.7; `Archive/by-date/2026-10-12/fp4-builtin-conformance/REPORT.md` §7.)* |
| **20** | **A like-for-like cost of the vendor fused op against our dequant + fp8-GEMM chain — and its cost at the real production shape.** The only microbenchmark we have compares the vendor **whole fused op** (routing + 2 GEMMs + activation, **0.591 ms**) against our **dequant alone** (**0.409 ms**), so **no ratio may be quoted from it** (§3.4.7); the GEMM term — the term that would make the comparison meaningful — is **not measured at all**. Separately, the shape is **synthetic** (`T = 512, H = 4096, F = 2048`, one expert), not the real production shape with a real expert count. | Build the chain: our `fp4_dequant` **plus** the `fp8_gemm_v2` consumer at the same shape, timed under one protocol; then repeat at the production shape and expert count. *(Opened 2026-10-12; §3.4.7; `Archive/by-date/2026-10-12/fp4-builtin-conformance/REPORT.md` §4.4 and `REPORT.md` §7 items 5–6.)* |
| ~~21~~ | ~~**Whether the promoted fp4 LUT path is safe — and as fast — end to end on the model.**~~ **CLOSED — RESOLVED 2026-10-13: the whole-model TP=4 decode RAN, it is bit-exact, and it is faster by 1.13× / 1.36× — not 3.853×.** Five arms, one engine process each, **four modules held simultaneously**, W0-anchor config, cold per-arm recipe cache. **Both real tables are bit-exact to the shipped path at model level** — `guarded` and `noguard_floor` each 24/24 token ids and **0 of 3072** logprob cells, `max|d| = 0` — and the tie cells are **reached**: **24 of 132,096 routed (token, layer, slot) events** land on a tie-carrying expert, and the amplified `tieprobe` table moves **24/24 positions**, so the `2⁻⁹ → 0` floor fix is a real, reached correction whose effect is **below the model's bf16 output resolution**. The **end-to-end** speedups are **1.1337× on the decode step** and **1.358× on the prefill step**; the **3.853×** is **kernel-local** (the dequant kernel alone at `[4096,2048]`) and is **not** a model-level speedup. Scope: one 35-token prompt, greedy, 24 steps × 2, `max_num_seqs = 1` — step walls, not throughput, and the prefill figure has **n = 2** (the measured-request-#1 ratio). *(§3.4.8; `Archive/by-date/2026-10-13/fp4-tp4-model-safety/REPORT.md` VERDICT and §2–§7; the earlier block-level and promotion evidence stays at `Archive/by-date/2026-10-13/fp4-lut-promotion-safety/REPORT.md` §5–§6, §9.)* | — **done** |
| **22** | **The rewrite's per-launch cost — the plan's N4 — is unmeasured, and the two-arm compile gate does not measure it.** The gate (§3.6.5) establishes that the KV-slab staging pass **disappears** when the slab is a declared operand read and written through its base address; it does **not** establish what the replacement kernel costs. Its treatment op is a **skeleton** at the model's geometry (the model itself may not be edited), and the rewrite's floor still carries a per-launch term read off a **harness** kernel (§4.2.10's bundle law), not off model nodes. So *"the staging pass is removed"* is **MEASURED** while *"R costs `0.2324 ms/step`"* remains **DERIVED**. | Compile the **real** fused per-layer kernel at the model's own geometry and read its per-launch term off its own launch, the way §4.2.10's arms were read. *(Opened 2026-10-13 by the two-arm compile gate; §3.6.5; `Archive/by-date/2026-10-13/compile-gate-run/`.)* |
| **23** | **HALF-CLOSED 2026-10-24 — the routed-expert half is answered (negatively); the `merged`-per-token half is not.** ~~The mechanism is INFERRED: on a sparse model the weight set is the **union of the routed experts**, which grows with the batch.~~ **REFUTED** by a direct test on a parallel port: routing diversity moves step time by **0.1–0.4 ms** at every n, and **D = 6 vs D = 36** is within **1 %**; the routed MoE explains **at most 1.16× of the step's 3.13×** growth. What the step *is* a function of is the **compiled bucket** — 10.4 / 25.9 / 32.5 ms at buckets 1/2/4, **flat in rows inside a bucket** (§3.1.5). **Still open:** this manual has **no** `merged`-per-token curve **on its own arm** — the campaign's serving arm never ran a concurrency sweep (`--max-num-seqs 1`, sweep **NOT RUN**) — and the parallel port's curve ends at bucket 4 because **bs > 4 does not compile** there. | The original ask stands, minus its second half: one arm at bs = **1, 2, 4, 8** on **our** code state, reading **`merged` per token** (§3.6.1) **and** the bucket each batch size maps to — the bucket, not the expert union, is the quantity to separate "the weights amortised" from "we crossed a step". *(Opened 2026-10-21; half-closed 2026-10-24 by the parallel port's own scaling study; §3.1.5.)* |

---

## 3.10 The porting checklist

**In order. Each step is the next one's precondition.**

**A. Get a trustworthy baseline.**
1. `hl-smi -L`; count AIPs, and read the module set with
   `hl-smi -Q index,module_id,serial,uuid -f csv,nounits`. **Decide the pinning story from what you
   see**, not from what you expect — the module set is box-specific, and the **module id** is the
   numbering to pin in (§3.7.1). **Check the set against the vendor's recommended combinations before
   you build a TP group on it — in module IDs: 2-Gaudi = `0,1` / `2,3` / `4,5` / `6,7`; 4-Gaudi =
   `0,1,2,3` or `4,5,6,7`.** The recommendation covers the *combinations*, and the page's own example
   is non-adjacent — so a set outside the list is not forbidden, but **the manual cannot price it:
   what a non-recommended set costs is not measured here** (§3.7.2).
2. If a container already limited you, **do not set `HABANA_VISIBLE_DEVICES`** — inside the container
   it is **inert** (nothing in the image reads it), so it neither helps nor isolates.
3. **Pin by module id, and make the pin set a partition.** Export `HLS_MODULE_ID=<module>` per job.
   It is read **lazily** at device-acquire time, so code in the worker may set it after `import`
   (and a launcher may set it in the environment it hands to a spawned worker); either way it must
   precede the worker's first HPU tensor; or use `torch.hpu.set_device(i)` as the **first** HPU
   call, which writes it for you. For a multi-worker job, give each rank a **distinct** module —
   one shared value kills the workers. Read back `os.environ["HLS_MODULE_ID"]` (the self-report
   APIs all lie; unset means **not pinned**).
4. Confirm a clean idle baseline (~768 MiB per device) before and after every run.
5. If scheduling several runs, **stagger them ~2 s** and never sample `hl-smi` during
   startup/teardown.

**B. Know the floor before you write code.**
6. Compute the floor **from the config and the checkpoint headers** — bytes per step, per card, and
   `ms/card` at the achievable bandwidth. **Do not take it from a trace.**
7. **Label each term by share**, and check whether the largest term is the one you were about to
   optimise. *(In our case it was not: attention 40.9 % vs MoE 30.7 %.)*
8. Check whether the FLOPs term binds at all (`R < BW × AI`); for M = 1 decode it does not.
9. Cost every byte at the right **engine** and the right **size** (§3.2.2).

**C. Get the formats right at load.**
10. Inspect **every scale tensor's dtype**. Handle raw `uint8` exponent bytes explicitly.
11. Apply the fp8 conversion **on the host tensor**, and **assert structurally** that you did.
12. Never pre-dequantize weights at load unless the HBM arithmetic closes.

**D. Get the graph structure right.**
13. **No dynamic shapes, and no varying Python ints in traced code.** If a size is data-dependent,
    make it static with fixed buffers and masks.
14. **Both compile levels must be zero after warmup.** Count `compileGraph` events — they are the
    *Synapse* level, not dynamo.
15. Minimise graph breaks; prefer fewer, larger regions — **but measure the specialization count
    after coalescing**, do not assume.
16. Record the **specialization count, launches, syncs, wall** as standing metrics.

**E. Only then optimise, and price in `merged`.**
17. Report **`merged`** (device occupancy), not summed kernel time.
18. Quote the wall relation **with its run set and its caveats** (§3.6.2).
19. Check the dead-end register (§3.8) before starting anything on the list.
20. For a fused-expert idea, **first check whether the op reads the routed experts or all local
    experts** — the read-set ratio is **10.7×** (6 of 64 vs all 64); against §3.4.6's 44.64 GB
    materialised path, routed-only is **12.9×** and all-local collapses to **1.21×**. *(Corrected
    2026-10-14 — this line read `12.9× vs 1.21×`; §3.4.5.)*
21. **When you fix a defect, sweep every place that *prescribes* the pattern you changed** — the
    loader, the tests, your own notes, and the docs. In §3.4.5 each of the three defects was found by
    reading **a different file from the one that was wrong**, and one of them was a document telling a
    reader to paste the broken line.
22. **When you substitute a vendor-built artifact for a reference one, assert its symbols in *both*
    directions.** A shared library's *undefined* set is what it deliberately leaves for runtime
    resolution; its *export* set is what the rest of the process will look for **in it**. Comparing
    only the undefined set can pass a library that cannot load. **MEASURED:** the patched
    Bridge passed a clean 2,410-vs-2,414 undefined comparison while exporting **0** `HlMlMemory*`
    symbols against the shipped **34**, and `import torch` then died on
    `_hpu_C.so: undefined symbol: HlMlMemoryReporter::GetPath` — a symbol the *installed wheel's own*
    pybind extension needs from that library. The cause was one **optional** CMake variable
    (`SPECS_EMBEDDED_ROOT`, absent from the project's own required-env list), so CMake logged a single
    `STATUS` line and built on. **An optional build variable is a silent failure mode: assert the
    feature's symbols, not the absence of a log line.** Working example: `1cat/build.sh`, whose
    self-check prints exports, `MISSING` against the reference library, and the feature's symbol count.
    *(The measurement, with the counts in both directions: `1cat/REPORT.md` §5.1 and §5.3, and
    `1cat/evidence/COUNTS.txt`.)*
    **`ldd -r` is the mirror of the same trap:** on a plugin library it reports every symbol that the
    *other* libraries in the process will supply as `undefined` (75 lines on our backend), so it reads
    as authoritative while proving nothing about loadability — and it says nothing at all about what
    the library fails to *export*. Neither tool is a substitute for comparing both symbol sets against
    the reference.
23. **A zero count in a log is a claim about the instrument first.** Before reading "0" as "it never
    happened", check whether the emitter is env-gated. **MEASURED:** the graph compiler's own
    instantiation handshake for the promoted fp4 LUT kernel prints **only** when
    `FP4_WRAPPER_DEBUG=1` (`src/custom_ops/fp4_dequant/src/entry_points.cpp`, a variable read
    nowhere else), so one arm's **0** `fp4_wrapper` lines read as "the kernel was never instantiated"
    when the debug flag was simply off and the kernel **had** been instantiated.
24. **Check for the artifact before believing the exit code.** **MEASURED:** `HPU_DUMP=1` stalls the
    first forward past the EngineCore's RPC timeout, so the harness reported `TimeoutError` then
    `EngineDeadError` and the workers leaked (all four modules held at ~52.9 GB each — held at
    `15:34:04Z`, free by `15:34:55Z`) — **yet the workers had already written their four dumps**, which
    is why that run's routing measurement exists at all. The failure is the harness's, not the device's:
    the work was done while 52.9 GB per module stayed held. On a harness-level failure, **look for the
    output file** before writing the arm off, and check `hl-smi` for leaked modules before the next launch.
25. **A detector that filters on a list which excludes the default cannot see the default.**
    **MEASURED:** the variant wiring's `too_late` detector filtered `/proc/self/maps` against
    `known_kernel_libs()`, a list that deliberately omitted the image default — so "no promoted library
    is mapped" stayed true while the kernel database was already fixed, and a late write reported
    `applied` for as long as it existed. Read your detector's allow-list for what it *leaves out*.
    *(§3.4.8.)*
26. **A harness that sets environment flags measures a configuration, not the product.** **MEASURED:**
    a `vllm serve` (HTTP) boot of the port **failed on all four ranks**, in the framework's
    bucketed prompt warm-up — bucket **11 of 90** (**INFERRED** from the progress bar; a **lower bound** on
    progress, not rank-exact, because the per-rank lines interleave across four processes), the first
    `(bs=1, query=256)` prefill bucket — with
    `RuntimeError: Not Implemented` from the graph compiler's **symbolic-shape execution**
    (`habana_frameworks/torch/dynamo/compile_backend/symbolic_execution.py`'s
    `int(sym_expr.subs(sym_value_dict))`, via symengine `Basic.evalf`). **The defect is not
    server-specific — the harness had been masking it.** The offline harness's own env
    applier sets `VLLM_T_COMPILE_DYNAMIC_SHAPES=0` and `VLLM_SKIP_WARMUP=1`, while
    `dynamic_shapes_compilation` **defaults to `True`**: so **every offline figure this manual reports was
    measured with dynamic shapes OFF and warm-up SKIPPED**, and the server run used the product defaults.
    **CORRECTED 2026-10-14 — that universal is false as written; the *universal*
    above is kept as the record, and the scope is narrower than it says.** **(a) The manual's offline
    *microbenchmark and analysis* campaigns never run the engine at all, so neither variable is theirs to
    set.** Their own headers say so: `engine-overlap`'s harness reads *"No vLLM, no model weights, no
    engine run"*, and `dependency-semantics` (*"Microbenchmarks and hand-built Synapse C-API graphs only.
    No `vllm serve`, no model weights, no engine run"*), `eocensus-t2-rereduce` (*"pure data reduction. No
    device, no HPU, no ports, no vllm"*), `span-ranking` (*"CPU-only"*) and `variant-decision-semantics`
    (*"CPU-only, code-reading"*) say the same of themselves. **(b) The offline *engine* figures older than
    2026-09-21 predate the line.** The `VLLM_T_COMPILE_DYNAMIC_SHAPES` `setdefault` lives in `_apply_env` in
    `src/load_and_decode/engine.py` and arrived with the **2026-09-21** harness commit (`VLLM_SKIP_WARMUP`
    has been there since 2026-09-03); the archived **09-15…09-18** engine logs carry `VLLM_SKIP_WARMUP` and
    **no** `VLLM_T_COMPILE_DYNAMIC_SHAPES` — i.e. the product default. **That leg is UNKNOWN, not
    disproved** — the 09-19/09-20 runs behind this chapter's pre-09-21 engine figures kept no engine log
    (there is no `Archive/by-date/2026-09-20/`, and the sequence jumps 09-19 → 09-26; 09-26 is the earliest
    archived log that carries the variable), while the 09-15…09-18 logs above license only their own window
    — so the operator could in principle have exported the variable by hand on those runs — but the
    **2026-09-21** record measures the setting *changing* the decode wall (**0.324 s → 0.318 s**), which is
    only possible if the pre-09-21 runs were not in it. **What still stands, and it is the point of this
    item:** the **default dynamic-shape path is a latent defect**, and **every archived offline *engine*
    run since 2026-09-21 went through that harness**, so its figures belong to that configuration — quote
    them with that configuration. The
    correction **narrows the scope; it does not retire the claim.** *(Leg (a):
    `Archive/by-date/2026-10-12/engine-overlap/eo_harness.py` header;
    `Archive/by-date/2026-10-13/{dependency-semantics,eocensus-t2-rereduce,variant-decision-semantics}/REPORT.md`;
    `Archive/by-date/2026-10-12/span-ranking/REPORT.md`. Leg (b): the 09-15…09-18 engine logs under
    `Archive/by-date/`; `notes/wall-bound.md`, the L6 de-guard row.)* **Read a harness's env before you
    quote its numbers, and say which
    configuration they belong to.** `VLLM_T_COMPILE_DYNAMIC_SHAPES=0` aligns a server run with the measured
    offline configuration — a **configuration choice, not a fix**; the default dynamic-shape path stays an
    **open defect**. And **do not reach for `VLLM_SKIP_WARMUP=1`**: it does not avoid the compile, it moves
    it to the first request that lands in that bucket — **a failed boot becomes a server that dies on its
    first use**. **What the same run did establish, in a real serving process:** the **KV budget check
    passed** at `--max-model-len 20480` (`GPU KV cache size: 27,364 tokens`, 1.34×) while `262144` cannot
    boot; the **plugin hook fired pre-device in all six processes**, so the fp4 promotion installs **before
    the first HPU op** — the ordering it depends on (the *rewrite* branch was not exercised: the script
    already exported `GC_KERNEL_PATH`); and the **LUT was active on 4 of 4 ranks** (`variant=noguard_floor`,
    md5 `5a1e3296fa3a6ecf3379b688a18e1540`, **96** `LUT` / **0** `OURS` handshakes **over that run's whole
    log**). **The HTTP layer was
    never reached** — no listener came up, so no response exists and none is claimed. **SUPERSEDED
    2026-10-14 — a retry of the same launch reached it.** *This sentence is retained as the record of what
    was true before the retry ran.* **MEASURED, on the retry:** the server **booted** and **answered HTTP
    coherently** — the reference request returned a grammatical, on-topic completion naming **Rayleigh
    scattering** as the mechanism, so the **2026-08-27 echo failure mode is ruled out**. It needed the
    `VLLM_T_COMPILE_DYNAMIC_SHAPES=0` configuration of the paragraph above — the offline path's own setting,
    a **configuration choice, not a fix** — and **still not** `VLLM_SKIP_WARMUP=1`. **It also needed a long
    engine-ready timeout:** ready at **2,118 s** against the stock **600 s** default, so the default would
    have aborted a correct boot as an engine-ready timeout ~1,500 s early — **a cold-recipe-cache DSv4
    `vllm serve` must raise the readiness budget, or it reports a timeout that is not one.** All **90 prompt
    + 9 decode warm-up buckets completed**, and the `Not Implemented` count is **0** (it was **8** here).
    The retry also re-confirmed, and counted, the serving facts above: the **KV budget** identical
    (`27,364 tokens`, 1.34×), the **plugin hook pre-device in 6 of 6 processes**, and the **LUT active on
    4 of 4 ranks** (`variant=noguard_floor`, same md5) with **960** `LUT` / **0** `OURS` handshakes **over
    that run's whole log** — the same rate as this run's 96 over its 10 completed buckets (96 / 10 =
    **9.6**; the retry's 960 over its 99 warmed buckets = **9.7**).
    *(`Archive/by-date/2026-10-13/vllm-serve-smoke/REPORT.md` §3–§4 and §7; the retry:
    `Archive/by-date/2026-10-13/vllm-serve-smoke-retry/REPORT.md` §1–§4 and §9.)*
27. **A serving path that is not a function of its request is not an instrument.** **MEASURED:** ten
    identical **seeded** greedy requests to the same running server gave **two distinct results**, and the
    split is **not random** — for one prompt class it is a deterministic function of the **request's ordinal
    within that class** (even → A, odd → B, **26 of 26** of the requests after the first, p ≈ 1e-8; the very
    first request after readiness is a third form, and for the other prompt class there is **no rule found**
    — which is not the same as "random"). **Those 26 are not one fixed request**: the no-`logprobs`, the
    `logprobs=20` and the `logprobs=20, seed=0` variants are mixed among them, so this is a regularity of
    the **observed sequence**, not yet of a single request. A merely *racy* mechanism would not hold a
    perfect parity over 26 requests; a *stateful, toggling* one fits better — **hypothesis, not
    measurement**. Four obvious
    explanations are **excluded by the records**: it is **not a seed effect** (6 seeded vs 6 unseeded, 4
    distinct results each), **not prefix caching** (**0.0 %** hit rate on every stats line), **not text
    post-processing** (the divergence is in the API's own `logprobs.tokens` / `top_logprobs`, not only
    `text`), and **not batch composition** (all **19** engine-stat lines in the serving window read
    `Running: ≤1`, `Waiting: 0`, and the probe client is synchronous). It is **not the fp4 LUT** — same
    `variant` and same table md5 on both sides. **The mechanism is UNKNOWN, and none is asserted.** The
    **strongest surviving structural candidate is KV-cache block allocation** — the one quantity in the
    serving configuration that differs from the offline one *and* is stateful, cyclic and per-request. The
    remaining not-excluded candidates, in the report's own order: the post-warm-up graph set (weakened —
    the offline path instantiates the same per-request kernel profile and is reproducible), a different
    compiled graph selected under the *same* shapes (not logged today), a heuristic kernel choice
    (`KernelConfig(enable_flashinfer_autotune=True, …)`), a non-deterministic reduction or atomics
    (weakened by the offline control), `enable_chunked_prefill`, and the HTTP layer's own logprob assembly
    (a serving-time-compiled `logsoftmax_fwd_f32` — a **different code path** from the offline arm's).
    **What it does NOT invalidate — the over-claim risk, stated explicitly:** the **performance** figures
    (a difference of order 1e-5 … 0.5 in a returned logprob cannot move a step wall), and this chapter's
    **bit-exactness** claims, which are scoped to the **offline** path (§3.4.8). **What it DOES
    invalidate:** any **HTTP-vs-offline numeric comparison** — the server is not a function of its request,
    so such a comparison cannot attribute a difference to anything, including to the LUT.
    **The offline control, with its own caveat.** The offline engine path **is** reproducible in-process —
    `meas1 == meas2` exactly, on both token ids and logprobs, in every arm — and the harness calls that arm
    its determinism control. **But that control is n = 2 per process**, which is weak on its own: at the
    server's ~30 % A-rate, five independent n = 2 arms would all read "identical" about 6.6 % of the time.
    **Do not overstate the offline path's determinism from it either.**
    **On magnitude, do not repeat the "1–4 bf16 ULP" story.** Recomputed as exact rationals, **51 of 51**
    residuals are **not** multiples of `0.125`, the coarsest grid any residual needs is **2⁻¹⁵**, and both
    sides' logprobs are **float32-exact** — so these are ordinary float32 differences, spanning
    **3e-5 … 0.53**. *(§3.4.8; `Archive/by-date/2026-10-13/serve-nondeterminism/REPORT.md` §1.1, §3–§5;
    the measurement: `Archive/by-date/2026-10-13/vllm-serve-smoke-retry/REPORT.md` §5–§7 and §11 item 2.)*

28. **An env knob that "does not exist" usually lives in the plugin, not in vLLM.** **MEASURED:** a
    parallel port of this model concluded that `VLLM_T_COMPILE_DYNAMIC_SHAPES` "appears **zero**
    times in the vLLM tree we actually run" and that setting it "is a no-op" — from a grep of
    **upstream vLLM core** (`vllm/`). The knob is defined in the **plugin**, as a feature flag:
    `vllm_gaudi/extension/features.py` carries
    `Value('dynamic_shapes_compilation', True, env_var='VLLM_T_COMPILE_DYNAMIC_SHAPES',
    env_var_type=boolean)`, and it works — it is the same variable item 26 above is about. Upstream
    vLLM core cannot contain a plugin's knob, so **the zero count is a property of where the grep
    ran, not of the build.** Grep the plugin tree (`vllm_gaudi/`) for `env_var=`, and check the
    **product default** rather than the absence of the name. This is item 23 in a new costume, and it
    is worth its own item because the wrong conclusion here is *reassuring* — "the switch does
    nothing, so dynamic shapes are not the problem" — and it retires a live hypothesis without
    testing it. *(The knob's definition site: `vllm_gaudi/extension/features.py`; the consumer:
    `vllm_gaudi/utils.py`'s `HPUCompileConfig`, which reads `get_config().dynamic_shapes_compilation`.
    **And "the default is on" does not imply "the default reached *your* graph".** The knob reaches
    `torch.compile` through exactly **one** consumer — `HPUCompileConfig.__init__` — and a port that
    passes `dynamic=False` at its own compile site bypasses it completely. A parallel port found
    exactly that: every layer-group / bucket entry is compiled at a site that hardcodes
    `dynamic=False`, so `VLLM_T_COMPILE_DYNAMIC_SHAPES=0` changed nothing there — **not because the
    knob was absent, but because the call site overrode it.** Read the consumer before believing the
    default. **Source read.**)*

29. **A compile failure surfaces at a FLUSH POINT, not at the culprit — bisect with `synchronize()`
    markers, and two rounds is usually enough.** **MEASURED:** a graph-compile failure presented at a
    **D2H copy** (`token_host.copy_(token, non_blocking=True)`) — which is simply where the lazy graph
    was flushed, and therefore *not* the location of the defect (§3.4.7's `synStatus 26` shows the
    same pattern from the other side). Inserting `torch.hpu.synchronize()` at successive statement
    boundaries, behind an env flag, took it down in **two runs and no theory**: round 1 bracketed the
    model forward and **proved the forward was fine**, collapsing the search to four statements;
    round 2 split those four and collapsed it to two. The theory being pursued in parallel (a
    collective inside the sampler) was **wrong**, and the markers did not care — which is the point:
    this instrument does not depend on being right about the cause. **Rules that make it safe:** the
    flag **defaults off** and warns once when enabled; every marker costs a **device
    synchronisation**, so an arm carrying markers measures a different thing and must not be used for
    timing; and the markers are removed by their `PROBE EDIT` marker before commit, not left behind a
    flag.

---

## 3.11 What this chapter deliberately does not claim

- **No general Gaudi 2 law from a single workload.** Every `[DSv4F]` number is one model on one box.
- **No firm prefill saturation target.** The MME rate (430 TFLOP/s) is MEASURED but pending a **2× unit
  ambiguity**.
- **No claim that the 44.64 GB materialised traffic is measured** — it is a model, and its largest
  component is uncorroborated.
- **No claim that cross-engine overlap is absent *or* present.** The 12.531 ms figure is UNRESOLVED.
  *(**Narrowed 2026-10-12, and still unresolved:** the figure now sits **just over the ceiling
  `O = M`** — ceiling `p ≤ 2.200`, `p` **bracketed at** the engine's own ratio **2.23–2.28**, `O/M = 1.02` —
  so it **misses the `O ≤ M` test by 1–4 %** at the measured `p` and is **neither established as real
  concurrency nor dismissed as an artefact**, and this bullet's
  "no claim either way" is **unchanged**. §3.6.3.)*
- ~~**No SRAM size, no L1/L2 behaviour** (UNKNOWN).~~ **RESOLVED — both are settled: the SRAM is 48 MiB on Gaudi 2** (`drivers/accel/habanalabs/include/gaudi2/gaudi2.h:41`), **and there is no L1/L2 cache.** *Stated in §3.9 (items 1–2) and Chapter 1 §1.5.*
- **No claim that removing the `cat` would help** — see §3.3.5; it is a recorded dead end.
- **No claim that a fix in the tree is a *validated* fix.** The §3.4.5 review found the replacement
  test uninstalled, the headline check arm-dependent, and the first version of the fix changing an
  untested input class. **Re-run the contract on your own build; do not inherit a status block as
  evidence.**
