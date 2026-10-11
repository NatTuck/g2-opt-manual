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
| 1 | **Pin devices and get a reproducible single-rank baseline.** | Everything downstream is a comparison. A mis-pinned or contended run invalidates it. | §3.7 |
| 2 | **Establish the *floor* for your model at your shapes.** Compute it from the config and the checkpoint headers, not from a trace. | It tells you which term is largest *before* you write any code. | §3.1 |
| 3 | **Classify each phase: compute-bound or weight-read-bound.** Prefill is M ≫ 1; decode at bs=1 is weight-read. | The doctrine inverts; the wrong choice optimises nothing. | §3.1 |
| 4 | **Get the quantization formats right at load time.** | Format bugs are silent, and one class of them is unrecoverable. | §3.4 |
| 5 | **Get the graph structure right: few regions, no dataflow-edge surprises, watch the specialization count.** | Graph breaks and specialization both cost, and they are hard to retrofit. | §3.5 |
| 6 | **Only then price optimisations, in `merged` (device occupancy).** | Summed kernel time is not the quantity that moves the wall. | §3.6 |
| 7 | **Keep a dead-end register** so nobody re-walks a closed path. | §3.8 — ours is 12 entries long. | §3.8 |

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

**The structural prefill moves follow from §3 of Chapter 2 (the overlap law):**

1. **Co-schedule MME with TPC in one recipe** — they share only HBM, which the MME barely touches
   (13 %), and absorb **82 %** of each other. This is the highest-value prefill scheduling move.
2. **Budget DMA for the MME** — MME operands stage via `memcpy_dma`, so **the MME is DMA-dependent by
   construction**, and those staging nodes ride the same queues as your explicit copies. This is why
   MME×DMA absorbs only **48 %**.
3. **Keep ops clusterable** or you fragment recipes and lose the `MAX`. **MEASURED** (synthetic C-API
   recipes; the percentages do **not** transfer to a real model).

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
> 2. **A 512 MiB C++ bench point reading 1,646 GB/s sits BELOW the measured 256 MB (2,021) and 1 GB
>    (2,028) points.** It is an **OUTLIER and is OPEN** (two untested explanations: 2-D vs flat 1-D
>    memcpy, and per-rep sync including launch latency). **Do not treat it as the ceiling.** **OPEN**

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

Read from the source. **MEASURED** (the DSv4F implementation)

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
**G separate 2-D calls**. **MEASURED** (schema read from the installed library)

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

**The K-axis order is FREE**, and this is the single most useful layout fact in the MoE path: because
the contraction is a sum over `K`, **any permutation of `K` applied to *both* operands leaves the
result unchanged.** The reference implementation exploits exactly this — it lets the dequant kernel
emit its **native** 64-byte-chunk interleave and permutes the much smaller **activation** to match,
rather than un-permuting the large weight. **MEASURED**

**⇒ A producer can emit the operand in whatever chunk order its kernel naturally writes, provided the
consumer permutes the activation identically.** Verify the identity holds for your shapes before
relying on it.

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

> **⇒ Verdict: "remove the cat" is NOT an actionable lever.** Not because the cat is free — because
  the graph compiler re-inserts the equivalent layout work for the GEMM's operand staging, so removing
> the user-level `cat` cannot remove the DMA layout ops from a GEMM-containing recipe. **Do not spend
> a run on it.** Recorded here so nobody re-walks it. **MEASURED**

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

---

## 3.4 Quantization formats and their hazards

**This is the section most likely to cost you a week.** Format bugs here are silent, and one class is
**unrecoverable**. *(§3.4.1 — what the device's fp8 actually *is* — was carried as an UNKNOWN in the
first draft and is now **SETTLED** by measurement. Read it before anything else in this section: the
wrong format *name* gives the wrong poison set, inverted.)*

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

#### The operational rule — the part a port needs

- **Poison on any HPU decode: `0x78–0x7E` and `0xF8–0xFE` — 14 codes.** The whole `E=15` bin is
  reserved, with `M == 0` → ±inf and `M != 0` → NaN. Measured decode: `0x78` → `+inf`, `0x79–0x7E`
  → NaN, `0xF8` → `−inf`, `0xF9–0xFE` → NaN. **MEASURED**
- **NOT poison:**
  - **`0x7F` / `0xFF`** — NaN in `e4m3fn` *and* NaN on the device. They **agree**; they are **not**
    mismatches.
  - **`0x80`** — the device reads it as `±0.0`, exactly as `e4m3fn` does.
  - everything else: the two agree on every `|v| ≤ 240`.
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

The **three new measurements** are disclosed rather than presented as archive-sourced. Device hygiene
for both device probes: one HPU, no vLLM, `GC_KERNEL_PATH` untouched, all four AIPs at the 768 MiB /
0 % idle baseline before and after.

### 3.4.2 The conversion, and the wrapper that implements it

The standard conversion that makes checkpoint fp8 valid on Gaudi 2 multiplies **fp8 weights by 0.5**
and **fp32 scales by 2.0**. The real (dequantized) value is preserved while the fp8 bit pattern is
**range-fitted into the device's representable range** — i.e. the checkpoint's top bin (`0x78–0x7E` /
`0xF8–0xFE`, §3.4.1) is moved out of the device's reserved `E=15` bin. **It is NOT an
`e4m3fn → e4m3fnuz` format conversion** — see the warning in §3.4.1. **MEASURED**

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
- **`fp4_dequant_fwd_gaudi2` is bit-exact** on the full domain (all 256 byte values × all 256 `k`
  values, 8 388 608 bytes) and at six production shapes, including the consumer operand. **MEASURED**

#### The exponent fold is UNCLAMPED — a LATENT defect, not an active one

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
> **Recommendation: add the `t ≥ 15` upper clamp anyway.** It is cheap, and the failure mode is
> **silent** (a poison byte in the operand, §3.4.1). Defensive work, not a bug fix.

**The lower bound: `k < −8` DOES occur — and it is benign, not a poison.** The kernel's own comment
only reasons about the lower bound ("exact for `k ≥ −8`"). It is exceeded: **1,070 routed-expert
bytes**, and **365,620 shared-expert bytes** down to `k = −13`. What happens there is **not
corruption**: the subnormal branch's `sh2 = max(t + 2, 0)` makes the emitted code **saturate to the
e4m3fn subnormal floor `2⁻⁹`** (byte `0x01`), the smallest value the format can represent — an
**overestimate** of weights whose true scaled value is below the floor, not a poison code.
Magnitude-tiny. Recorded so nobody re-derives it. **MEASURED**

> **⚠ And a test-staleness warning worth carrying.** A `test_fp4.py` in the source tree calls
> `fp4_dequant(b)` with **one** argument against the shipping **two**-argument schema, and its
> reference implementation **ignores `kexp`** entirely. **It therefore does not validate the shipping
> kernel.** Do not treat a green `test_fp4.py` as evidence about the kernel you are shipping.
> **MEASURED**

**Don't pre-dequantize at load.** Converting fp4 → fp8 at load time **doubles the weight bytes**
(0.5 → 1 B/value), taking the routed buffers from **36.8 → 71.4 GB/rank** (+34.6 GB), on a 96 GB card
with **12.3 GB free** ⇒ **HBM-blocked**. **NOTE-CLAIM [DSv4F]**

**Read only the routed experts.** Reading the 6 routed experts of 64 local gives a **12.9×** traffic
win; reading **all 64 local** experts gives **36.8 GB/step** and the win collapses to **1.21×**. **I**
— *this is the single check that decides whether a fused-expert optimisation is worth anything.*

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

## 3.5 Specialization and graph count

**This is where a port silently loses memory and gains compile churn.** Read it before you structure
your model's forward.

### 3.5.1 There are no dynamic shapes, and **ints are specialized too**

- **`dynamic=True` is turned into `{"options": {"force_static_compile": True}}`**, and
  `dynamic_shapes_compilation` defaults True. So every tensor shape is **baked into the compiled
  graph** — you cannot pass a dynamic slice bound or shape and have one graph serve several cases.
  **MEASURED**
- **A Python int used in the traced code becomes part of the guard set** and produces a **separate
  compiled graph per distinct value**, exactly like a new shape does. A per-layer byte offset, a layer
  index, a constant — all of them. **There is no "dynamic argument" escape hatch for ints.**
  **MEASURED**

> **⇒ The graph count multiplies by layer × constant × shape.** This is the single most important
> structural fact in this section, and it is the one a port is most likely to violate accidentally by
> baking a per-layer index into the traced code.

### 3.5.2 Regional compilation, and the observed scale

- **Regional compilation compiles each decoder layer as its own `torch.compile` function.** **MEASURED**
- **Observed scale on the DSv4F port: ~194 distinct recipes per decode step, 160 of them used by
  exactly one layer** — i.e. almost all specialization is **per-layer**, and the instance count scales
  with `n_layers` (43), not with the number of region kinds. **MEASURED [DSv4F]**
- **The recipe cache limit is inflated by `n_layers + 2`** to accommodate this. Watch for a single
  region exceeding the cache size limit — that causes **recompile thrash**. **MEASURED**
- **Two independent compile levels exist, each with its own cache and key**: the **recipe**
  (`recipe_key` — the IR plus declared shape *ranges*) and the **Synapse graph** (`graph_key` —
  recipe plus concrete shapes plus the bucket and dynamic-bucket info). **Both must be zero after
  warmup**, or any timing measured over that window is invalid. **MEASURED**
- **Naming trap: the profiler event called `compileGraph` is the *Synapse graph* compile, not a
  dynamo compile.** A guard can be clean at level 1 while level 2 is wide open. **MEASURED**
- **The shapes can be data-dependent without your noticing.** One port's decode window showed **93
  `compileGraph` events per step** while level 1 was clean, because a live-entry count was reaching
  the graph as a tensor size. **MEASURED [DSv4F]**. The fix is code: make those dims static
  (fixed-size buffers + masks).

### 3.5.3 Graph breaks cost twice

**A graph break splits the compiled program into more regions, which means more graph launches — and
each launch blocks the host in a `synchronizeEvent`.** So a break costs you on both sides. **MEASURED**

**⇒ Prefer fewer, larger compiled regions.** But:

> **⚠ Coalescing regions does NOT reduce the specialization count if the coalesced region still guards
> on each layer's ints.** The *parameter*, not the region boundary, is what sets the instance count.
> Confirmed empirically: grouping 4 layers into one compiled unit **moved no operational count**.
> **MEASURED**

**⇒ Always check the specialization count after a coalescing change; do not assume it helped.**

### 3.5.4 The memory side

- Retained graph/activation footprint after warmup was **~11 GiB** on one port; transient demand during
  the first decode's compilation was **~30 GiB** above post-build. **MEASURED [DSv4F]**
- **~55 MB per compiled instance** is the arithmetic (`11 GiB / ~194 instances`) — a **hypothesis, not
  a measurement**. If the footprint does not fall when the instance count falls, the memory is
  somewhere else and the hypothesis is void. **INFERRED**
- **The tradeoff is device-side on both sides**: specializing buys device *runtime* (baked constants
  fold into immediates; exact shapes give better fusion/tiling) and costs device *memory*; reducing
  instances **saves** host work and memory and can only cost runtime. **INFERRED**

**⇒ Keep an instance only where it measurably cuts device time.** Everything else should be collapsed.

### 3.5.5 Standing metrics for a port

Track these together, before and after every change. Any one alone has misled us:

- **ops/step by family** — and note that a **device event is a replication count, not a launch**:
  kernel events ÷ recipe launches was **544 / 569 / 971** across three traces of the same model —
  *not even a fixed multiple*. **MEASURED**
- **`merged`** = `|union(kernel ∪ hpu_op Dma*)|` — device occupancy (§3.6).
- **graph launches/step** and **host syncs/step**.
- **distinct recipes** (the specialization count) — and see the probe in
  [`examples/03-porting/specialization_probe.py`](examples/03-porting/specialization_probe.py).
- **the step wall**, unprofiled.
- **`compileGraph` events** after warmup (should be zero).

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

### 3.6.3 What is *not* established about decode

- **Whether the engine's cross-engine intersections are REAL concurrency.** A TPC∥DMA intersection
  figure of **12.531 ms/step** has moved three times in one day, each move forced by a measurement. It
  is **UNRESOLVED**: measurement spread (0.09–0.89 %) puts the deciding margin **inside the noise**;
  the load-transfer extrapolation is uncertain by **tens of percent**; and the quantity that decides it
  **was never measured** — the deciding arm held the variable fixed, which made its own pre-committed
  falsifier **unevaluable by construction** (a design flaw, not a null result). Two harness-calibrated
  routes both give `O/M ≤ 1` and differ by **33 %** (0.99 vs 0.66) — **that spread IS the honest
  measure of how soft the estimate is.** **MEASURED**
- **The size of the ~21 ms host constant's sync-wait vs host-compute split.** `synchronizeEvent`
  carries a Synapse **stream** id, not an OS tid, so it cannot be intersected with the runner thread.
  **UNKNOWN** — settling it needs a runtime change (a `record_function` around each blocking sync at
  the point the runtime issues it).
- **Whether the DMA channel choice is right for *your* model.** Our verdict rests on three reasons
  (load-tracking, subset/definitional, and the wall fit); a fourth argument we originally gave
  (seriality/saturation-collapse) was **retracted**, and the supporting sentence in one note **reads
  inverted against its own conclusion**. Re-derive it on your traces. **MEASURED / OPEN**

### 3.6.4 The device-time decomposition, for orientation [DSv4F]

| family | ops/step | merged ms/step |
|---|---|---|
| `fp4_dequant` | 9,288 | 22.63 |
| `DmaMemcpy` | 84,090 | 16.94 |
| `GEMM` | 8,924 | 11.48 |
| `gather_nd_*_u8` | 3,096 | 4.38 |
| `DmaTranspose` | 50,790 | 7.05 |
| `DmaBroadcast` | 6,235 | 5.08 |

**Three families are 46 % of `merged`.** **MEASURED [DSv4F]**. Note the shape of this table: the two
largest *op counts* are not the two largest *time* items, and the family with the largest op count is
not in the top three by time. **Count and cost are different questions.**

---

## 3.7 Multi-device, pinning, and concurrent processes

### 3.7.1 There are two distinct notions of device identity — and confusing them is the classic failure

- **Local device index** — what `torch.hpu.device_count()`, `HABANA_VISIBLE_DEVICES` and `hl-smi`
  call `accel0..N` **in your view**. Dense, starts at 0.
- **Physical module id / AIP** — what `hl-smi -L` shows as each AIP's Module ID. **Fixed per card.**

`HABANA_VISIBLE_DEVICES` **re-numbers the visible set into a dense `0..N` local view**.
`HABANA_VISIBLE_MODULES` is a **physical** filter and can differ in ordering and value.

> **When two processes pick the same PHYSICAL module under different LOCAL views they collide
> silently** — both think they own a free device. That is the `synStatus=8 [Device not found]` /
> `c10::Error` worker-init failure. **MEASURED**

### 3.7.2 `HABANA_VISIBLE_DEVICES` is dangerous when a container has already limited you

**Check first, then decide.** On a bare host you will see 8 AIPs and you must pin. **In a container
that has already limited you to 4, the four visible AIPs are already yours — do not set
`HABANA_VISIBLE_DEVICES` and do not try to pick global devices.** Use relative numbering. **MEASURED**

A container that hands you host modules 4–7 typically shows **`HABANA_VISIBLE_DEVICES=0,1,2,3` AND
`HABANA_VISIBLE_MODULES=4,5,6,7`** — a clean dense local view over a physical subset. That is why
4-device runs "just work" there. **MEASURED**

**Check before you assume:**

```
hl-smi -L                                   # 8 AIPs on a host, 4 in a limited container
env | grep -iE "HABANA_VISIBLE"
python -c "import torch, torch_hpu; print(torch.hpu.device_count())"
```

**An idle device reads ~768 MiB used; much higher means a process holds it.** **MEASURED**

### 3.7.3 In-process pinning

- **`torch.hpu.set_device(i)` as the FIRST HPU call** maps to `HLS_MODULE_ID =
  HABANA_VISIBLE_MODULES[i]`. In one 4-AIP container the observed mapping was **0→0, 1→3, 2→6, 3→7**.
  **MEASURED**
- **No self-report API works** — `current_device`, `current_device_index` and `tensor.device` all
  report **0**. **Read `os.environ["HLS_MODULE_ID"]`** instead. **MEASURED**

### 3.7.4 Concurrent processes do not perturb timing much — a 4-concurrent schedule is safe

- **< ~2 % device, < ~5 % host** perturbation from concurrency. **MEASURED**
- **Two gotchas**: (a) **stagger launches by ~2 s** — a simultaneous start lost a process to
  `synStatus=8` in **3 of 5** attempts; (b) **never sample `hl-smi` during startup or teardown** — it
  shows a transient (98304 MiB). **MEASURED**
- **Treat one module as contended** if you are sharing the box. **MEASURED**

### 3.7.5 Worker-process hygiene

- **Let the platform choose the worker launch method.** On HPU the multiprocessing method is
  auto-selected (`spawn`, because `fork` can hang on exit). **Forcing `fork` is a known dead end.**
  **MEASURED**
- **Workers live in a separate process group from the driver leader.** After killing the leader's
  process group you must also kill the worker PIDs explicitly, **or they orphan**. **MEASURED**
- **Never `pkill -f` on a shared machine** — it kills other users' servers. Track the PID you spawned.
  **MEASURED** (and it is a shared-server rule, not just a performance rule)

---

## 3.8 The dead-end register

Every entry below cost real time. They are recorded so a port does not re-walk them. **This register is
as valuable as the wins.**

| # | dead end | why it is dead | tag |
|---|---|---|---|
| 1 | **"Remove the `cat`"** | The graph compiler inserts `concat_logical` + `memcpy_dma` for GEMM operand staging regardless; a GEMM recipe contains DMA layout ops whether or not the user's cat is there. And the IR cannot decide critical-path membership. | **MEASURED** |
| 2 | **A 256-entry LUT for the fp4 decode** | The lookup instruction's table is a **built-in coefficient table selected by `funcid`** — the caller's `lut` tensor is **never loaded**, and four different tables give byte-identical output. Not expressible on this ISA. | **MEASURED** |
| 3 | **Re-expressing the dequant in native torch ops** | Bit-exact, and **21.7× slower** — 76 % of the cost is the `torch.gather` **int32 index** (a 100 MB index for a 25 MB payload). No chain of elementwise ops matches one fused memory-bound pass. | **MEASURED** |
| 4 | **Pre-dequantizing experts at load time** | Doubles the weight bytes: **36.8 → 71.4 GB/rank** on a 96 GB card with 12.3 GB free. HBM-blocked. | **NOTE-CLAIM** |
| 5 | **Two HPU streams for cross-engine overlap** | `CROSS = 0.0000` **with a valid precondition** (119.1 ops outstanding, both queues provably loaded), and the host cost explodes. | **MEASURED** |
| 6 | **"Avoid custom ops" as a heuristic** | **Refuted by the model's own op.** Three arms held customness fixed and flipped the outcome purely on **submission count**: custom op → 0.2217 at 1 submission; custom op → 0.0000 at 2; native op → 0.4179 at 1. The discriminator is the submission count. | **MEASURED** |
| 7 | **Region coalescing alone** | Grouping 4 layers into one compiled unit **moved no operational count** — the *parameter*, not the region boundary, sets the instance count. | **MEASURED** |
| 8 | **Launch count / sync count / recipe count as the lever** | r² 0.15 / 0.195 / 0.066, **all with the wrong sign**. | **MEASURED** |
| 9 | **The "device idle ~80 % / launch-bound" framing** | **Stale** for the current code state: the step is ~76 % device-occupancy. It described a code state 3× slower. | **MEASURED** |
| 10 | **`torch.stack([a, a], dim=-1)` — the same tensor twice** | Costs **22,532 µs vs 301 µs** for two *distinct* tensors — **75×** — on a 50 MB output. It silently destroyed a component benchmark (read 22,500 µs inside a 4,300 µs whole-op measurement). | **MEASURED** |
| 11 | **Profiling with CPU activity enabled** | Inflates the trace's host span **~3×** and collapsed a queue-depth metric from 23.3 to 1.67, manufacturing a false empty queue. **Use HPU-activity-only profiling.** | **MEASURED** |
| 12 | **`**`-globbing a dump directory** | Silently **skips the hidden `.graph_dumps/`** directory — one recon saw **24 of 1,141** files on its first pass. Use `os.walk`/`find`. | **MEASURED** |
| 13 | **Counting "device ops" as a work-unit count** | A device event is a **replication count** — one event per (launch × TPC/DMA engine) — so kernel events ÷ recipe launches was **544 / 569 / 971** across three traces. **Not even a fixed multiple.** | **MEASURED** |

---

## 3.9 The UNKNOWN register

Stated, not omitted. Each has a note on what would settle it. **Two entries have been closed since the
first draft — items 10 and 14 — and both are kept in place, marked CLOSED, so the resolutions are
visible rather than silently deleted.**

| # | UNKNOWN | what would settle it |
|---|---|---|
| 1 | **SRAM size.** The only figure anywhere ("48 MB") is marked **unverified by its own erratum**; do not import it. What *is* measured is that SRAM staging exists and is visible in the compiler's own output (`"allocation": "SRAM"`). | A vendor document that survives cross-checking, or a staging-size sweep. |
| 2 | **L1/L2 cache sizes and behaviour.** | Not attempted. |
| 3 | **Whether the TPC or DMA cost models consume the *general* HAL reader** (2000 MHz) rather than an engine-specific one (1650 MHz). If they do, they are **~21 % optimistic**. The MME has its **own** reader at 1650 MHz and is accurate to 0.5 %; the general reader's consumption by the other engines is **UNCHECKED**. | Disassemble the TPC/DMA cost models' clock path the way the MME's was done. |
| 4 | **The 512 MiB C++ bench point (1,646 GB/s)** — an outlier below the measured 256 MB and 1 GB points. Two untested explanations (2-D vs flat 1-D memcpy; per-rep sync including launch latency). **OPEN** | A 1-D version of the same bench, and a version that syncs once outside the rep loop. |
| 5 | **Whether the engine's cross-engine intersections are real concurrency** (§3.6.3). | An arm that **varies** the deciding quantity — the one that ran held it fixed, making its own falsifier unevaluable. |
| 6 | **The ~21 ms host constant's sync-wait vs host-compute split.** | A runtime change so `synchronizeEvent` carries the calling OS tid, or a `record_function` at the point the C++ runtime issues the sync. |
| 7 | **The per-recipe / per-step / per-queue split of the DMA window's fixed component.** Every sweep arm held nodes, recipes and queues constant, so it is **confounded**. | A sweep that varies those independently. |
| 8 | **Composition for unequal cross-engine groups, or > 2 engines in one recipe.** | New arms. |
| 9 | **Whether the three extra DMA tids (4073, 4082, 4083) are device or host.** Our reading splits an earlier "host PCIe" ruling on 4082/4083. | A targeted probe. |
| ~~10~~ | ~~**Which description of the fp8 top bin is literally correct** (§3.4.1).~~ **CLOSED — RESOLVED 2026-10-11.** **The device's fp8 is `e4m3fn` codes (bias 7) truncated at 240; it is NOT `e4m3fnuz`.** `torch.float8_e4m3fnuz` is **not a supported dtype on HPU** (`is_supported_type` rejects it), the device's own **encoder** emits bias-7 codes (1.0 → `0x38`, 240 → `0x77`), and `cast_from_fp8` matches the CPU `e4m3fn` table **242/256** vs `e4m3fnuz` **1/256**. So the "storage = fnuz, decode = fn" split does **not** apply — the device does not *write* fnuz codes either. `float8_e4m3fnuz` contributes exactly **one** useful number to this stack: the ceiling **240**. **Full statement, the poison-code set, and the guard rule: §3.4.1.** *Provenance: settled by a decode+encode sweep on one HPU (one disclosed 11-second device probe), because the archive could not answer it.* | — **done** |
| 11 | **Whether the ~55 MB per compiled instance scales with the instance count.** If it does not, the memory is in transient activation workspace or per-launch state, and the specialization/memory tradeoff is void. | De-guard one region and re-measure the post-warmup footprint. |
| 12 | **Whether the `cat` is on the critical path.** The IR cannot decide it (no unique node ids), and the one synthetic arm was inconclusive and changed the engine mix. | A GC-side post-graph dump from the real model, or an arm that holds the engine mix fixed. |
| 13 | **The exact tensor/line that introduces a data-dependent size into the graph** on our own port. Candidate identified, not pinned. | Per-region guard-key inspection. |
| ~~14~~ | ~~**Whether the fp8 reserved-bin rule is ENGINE-DEPENDENT** — i.e. whether the MME could decode the reserved bin differently from the TPC.~~ **CLOSED — RESOLVED 2026-10-11. NO: the rule is a property of the device, not of an engine.** **`MME == TPC byte-for-byte over all 256 codes: True`**, same 14-code mismatch set, measured **within one process on one code set** (so a within-run two-engine comparison, not two measurements days apart). Calibration passed first — **242/242 exact through the MME** — so the readout is a pure decode. **`e4m3fnuz` is excluded twice over on the MME**: it reads `0x76` as **224** (fnuz would be 112) and `0x80` as **0** (fnuz would be NaN). **⇒ the format statement needs no engine qualifier, and the guard's byte set `0x78–0x7E` / `0xF8–0xFE` is confirmed for BOTH engines.** *Mechanism worth keeping: the fp8 decode happens **inside the GEMM node** — a post-graph dump shows `MME GEMM ['hfloat8','hfloat8'] → ['bf16']` consuming fp8 directly with **no intervening cast**, so **no engine-attribution pass can ever see it** and would have reported "MME, no fp8 cast" and been silently wrong.* **Full statement: §3.4.1** (see also the closed item 10 above for the format itself). *Provenance: a second disclosed ~11-second device probe, with its own post-graph dump.* | — **done** |

---

## 3.10 The porting checklist

**In order. Each step is the next one's precondition.**

**A. Get a trustworthy baseline.**
1. `hl-smi -L`; count AIPs. **Decide the pinning story from what you see**, not from what you expect.
2. If a container already limited you, **do not set `HABANA_VISIBLE_DEVICES`**.
3. Pin in-process with `torch.hpu.set_device(i)` as the **first** HPU call; read back
   `os.environ["HLS_MODULE_ID"]` (the self-report APIs all lie).
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
    experts** — 12.9× vs 1.21×.

---

## 3.11 What this chapter deliberately does not claim

- **No general Gaudi 2 law from a single workload.** Every `[DSv4F]` number is one model on one box.
- **No firm prefill saturation target.** The MME rate (430 TFLOP/s) is MEASURED but pending a **2× unit
  ambiguity**.
- **No claim that the 44.64 GB materialised traffic is measured** — it is a model, and its largest
  component is uncorroborated.
- **No claim that cross-engine overlap is absent *or* present.** The 12.531 ms figure is UNRESOLVED.
- **No SRAM size, no L1/L2 behaviour** (UNKNOWN).
- **No claim that removing the `cat` would help** — see §3.3.5; it is a recorded dead end.
