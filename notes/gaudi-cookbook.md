# Working notes: `thediymaker/gaudi-cookbook`

**Status:** investigation notes for our own team, not a manual chapter.
**Subject:** [`github.com/thediymaker/gaudi-cookbook`](https://github.com/thediymaker/gaudi-cookbook),
cloned read-only to `src/gaudi-cookbook`.
**Provenance:** commit **`1540263772481343d92ce34a2734a34b44dbbda7`** ("Show the original maker's logo on
third-party quantizations", 2026-10-25), **23 commits, all inside a two-day window** (2026-10-24 →
2026-10-25), single branch `main`, 13 MB. **Licence: Apache-2.0** (`LICENSE`, unmodified Apache text,
no NOTICE file).
**Method:** I read the code, the kernels and the glue, not just the prose. I ran **one** thing — a
CPU-only check of their layout arithmetic (below) — and nothing on an HPU. Every figure they publish is
**their claim**; I could not reproduce any of it and neither can anyone else from this tree.

---

## 0. Verdict, up front

**Yes, there is material we do not have — more than I expected from a "cookbook", and three items are
worth acting on.** This is not the same project as `autoresearch-gaudi` (different licence, different
subject, no shared text), and it is a *much* better artifact: real TPC-C kernels, real parity gates, real
CI. But **it publishes no measurement artifacts at all**, so every performance number in it is prose.

Ranked by what it is worth to us:

1. **A constant-fold that runs a *clamped* SwiGLU exactly on the vendor `bias_mxfp4_fused_weights` op**
   — and it is shipped code, for **our model family** (DeepSeek-V4.1-Flash). Our manual currently
   *rejects* the vendor op for DSv4F on the grounds that its clamped path is unreachable (§3.4.7 check
   1). **That rejection is wrong, or at least premature.** §2.1.
2. **Their kernel is not built with `--pre-RA-sched=vliw-td`** (`build.sh:128` is plain `-O2`), and by
   their own analysis it is *vector-issue-bound*. That is our biggest measured lever, untested on a real
   streaming MXFP4 kernel. §2.2.
3. **A batched epilogue that amortises the transcendental over 32 passes by lane-stuffing the horizontal
   sums into one vector register** — a direct refinement of our "8→16 accumulators" lever and its
   "sign inverts when the accumulator is expensive" caveat, with the A/B switch already in the source.
   §2.3.

Then a set of smaller but clean items: the **lane id exists** (contradicts §4.1.4), a **fp32 division
defect** with a quantified consequence (cheapest high-value probe in the repo, §2.4), the **interleaved
gate/up layout** for the bias path (closes §3.9 item 17's doc-vs-doc disagreement), **E8M0→bf16 by `<<7`**
(which explains our §3.4.7 UNKNOWN 18b's flush-to-zero mechanism), and four **lazy-mode custom-op
hazards** that are silent-wrong-result class.

**On SRAM (chapter 7): nothing.** They do not touch the 48 MiB pool, the slicer or placement. The one
memory-hierarchy item is the *TPC's own* local memory, which our chapters barely cover (§2.5).

---

## 1. Is this the same project as `autoresearch-gaudi`?

**No.** Same author, unrelated content. `rg -i autoresearch` over the tree: **no hits**. Different
licence (MIT there per our §3.9 item 23, **Apache-2.0** here). The subject here is vLLM-on-Gaudi serving
(recipes, runtime fixes, ports, Kubernetes) plus one TPC kernel family; there, a training/research loop.

**But the scepticism still applies, and differently.** Item 23's complaint was *"no tests, no CI and no
issues ever opened"*. That is **not** true here: there is a real CI (`.github/workflows/ci.yml` — ruff,
yamllint, `mkdocs build --strict`, a CPU `pytest` run, and a `fixes-on-release` job that fetches two
pinned upstream commits and applies every fix against them), and the kernels carry genuine gates. The
failure mode is different: **the CI explicitly skips everything that needs a device**
(`ci.yml`: *"skips what needs torch, vLLM or a device"*), so **no kernel parity test, no gate and no
benchmark in this repo has ever run in CI**, and **no output of any of them is checked in**. What is
missing is not tests — it is *evidence*.

---

## 2. What is genuinely new, ranked, with the experiment that would settle it

### 2.1 ⭐ The clamp fold — a clamped SwiGLU on the stock vendor op (corrects §3.4.7)

**What they do.** DeepSeek-V4.1's experts compute `silu(min(g, 10)) · clamp(u, ±10)` — *the same clamp
our manual records for DSv4F*. They run it on the **stock** `bias_mxfp4_fused_weights` op, not a custom
kernel, by folding constants into the op's own GPT-SwiGLU path.

**READ** — `ports/deepseek-v4-1-flash/dsv41_hpu/dsv41_hpu/vllm_model.py:148-186`. The docstring states
it and the code does it:

```python
k = SWIGLU_K                     # = 7   (:69)
assert int(s13.max()) + k <= 254, "E8M0 scale headroom exceeded"   # (:156)
s13.add_(k)                      # (:157)  E8M0 byte += 7  == weight x 2^7, exactly
b13 = torch.zeros(2 * self.I, ...); b13[1::2] = -1.0               # (:159-160)  up rows only
op = VllmMixtureOfExpertsOpMXFP4(..., has_bias=True,
                                 alpha=2.0**-k, limit=self.limit * 2.0**k)   # (:161-171)
self._wscale = 2.0 ** (-2 * k)   # (:186) applied to the routing weights
```

**Why it works** (INFERRED — my arithmetic, and it checks out). The op computes
`min(g,L')·sigmoid(α'·min(g,L'))·(clamp(u,±L') + 1)·ascale` on `g' = 2^k g`, `u' = 2^k u − 1`:

```
L' = 10·2^k,  α' = 2^-k  ⇒  min(g',L') = 2^k·min(g,10),  α'·min(g',L') = min(g,10)
up bias −1 cancels the op's hard-coded "+1"  ⇒  (clamp(u',±L') + 1) = 2^k·clamp(u − 2^-k, ±10)
ascale 2^-2k  ⇒  output = silu(min(g,10)) · clamp(u − 1/128, ±10)
```

So it is exact **except** that the *up* clamp acts on `u − 1/128` rather than `u`: within 1/128 of the
up-clamp boundary it clamps early, i.e. by at most **0.078 %** of the limit. Their word "exactly"
(`docs/ports/deepseek-v4-1-flash.md`) is very nearly right and not quite right. **INFERRED**, arithmetic
only; nobody has measured it.

**Why the fold is exact where it matters.** `s13.add_(7)` multiplies the weight by 2⁷ *through the E8M0
exponent byte*, which is a pure exponent — so it is **lossless**, no requantisation. That is a genuinely
nice trick and it is only available because MXFP4's scale is E8M0.

**Two hazards they hit, both of which we already document:**
- The `assert s13.max() + k <= 254` guard is exactly our §3.4.7 **trap 1** ("one out-of-range scale NaNs
  the entire layer"). The fold can push a byte to 255 = `inf` → NaN. **Any fold by +k needs this guard.**
- `vllm_model.py:173-174`: *"Every expert needs its OWN bias tensors: the Synapse MoE multiplexer
  registers per-expert inputs, and one shared tensor object fails the graph compile with 'MOE multiplexer
  weights were partially registered'."* New compile-error trap. **THEY SAY.**

**Follow-up experiment.** On one card, with one real DSv4F expert layer: build the folded op as above,
and compare against a CPU fp32 `silu(min(g,10))·clamp(u,±10)` reference over (a) random activations at
RMS 1 and 4 and (b) captured decode frames. Then time per call at T = 1/4/8/16 against the unfolded
stock path.
**Settles:** whether our port needs a custom clamped-activation kernel *at all* for the stock-op path,
and what the fold costs. **If it works, our §3.4.7 checklist item 1 ("this check alone rejects DSv4F")
must be rewritten** — see §4.1.

### 2.2 ⭐ Their kernel is issue-bound and does not use our biggest lever

**READ** — `kernels/mxfp4-moe/build.sh:128`:

```bash
"$TPCCLANG" -march=gaudi2 -O2 -x c++ ${3:+-I"$3"} "$1.c" -c -o "$1.o"
```

**No `-mllvm --pre-RA-sched=vliw-td`.** Their whole kernel family is built with the default `source`
scheduler, which our §4.2.2 measures as leaving **1.19×–3.45×** of bundle count on the table.

**And by their own analysis that is the term that binds.** **THEY SAY**
(`docs/kernels/mxfp4-moe.md`, "Why it is issue-bound"): one 512-value chunk-row costs ~17 vector
instructions; a route runs at 0.44 TB/s against a ~2.2 TB/s HBM floor; and the decisive ablation is that
*"a w13 pass with every load removed still took about 25 µs per expert, against 34 µs with loads and a
9 µs HBM floor"* — so **the loop is limited by vector issue, not memory**. If that is right, it is
precisely the regime where co-issue (which hides load/store/scalar behind vector slots) should pay.

**A caution, from our own chapter.** Our §4.1.1/§4.2.7 floor is `#vector ops × 0.556 ns` — co-issue
**cannot** multiply vector throughput. Their ~17 vector ops per chunk-row is a floor of 17 bundles; if
the default build emits ~25, the available headroom is ~1.4×, not 3×. So the prediction is
**1.2–1.4×, not 3×** — a useful, falsifiable number to carry in.

**Follow-up experiment.** Add the flag to `build.sh:128` (and to the `-S` line), rebuild one variant,
count bundles with `--tpc-asm-format=2`, gate on `tests/run_parity.sh --variant gptoss` (`PARITY PASS`
+ `SIG_DEVICE PASS`, bit-exactness is our §4.2.5 requirement), then `bench/bench_moe.py --compare`.
**Settles:** whether our bundle lever generalises to a real streaming dequant kernel, and whether their
"issue-bound" diagnosis has ~1.3× sitting unclaimed in it. **Cheap — hours, not days.**

### 2.3 ⭐ The batched epilogue — a third answer to "the accumulator is expensive"

**Their problem, and it is our caveat exactly.** The w13 pass produces four horizontal sums per pass
(4 weight rows), and each sum needs a `v_f32_reduce_add` plus the activation's two `v_sigmoid_f32`.
**THEY SAY** (`docs/kernels/mxfp4-moe.md`): run once per column pair, *"the two sigmoids cost about a
quarter of a pass"*. That is the "expensive accumulator" regime our §7.6 caveat warns about.

**Their answer is not fewer accumulators — it is to batch the epilogue.** **READ**
`kernels/mxfp4-moe/src/mxk8_w13.h:105-190` with `mxk8_core.h:110-120`:

- each pass's four horizontal sums are *not* consumed; they are **written into lane `m` of two `float64`
  VLM registers** by a predicated `v_f32_mov_vb(h[k], 0, G, to_bool64(la), 0)` where `la` is
  `lid4 == m` — i.e. one scalar result per lane, 32 results per vector;
- the activation then runs **once per run of up to 32 passes**, on whole vectors;
- the store is one `v_bf16_st_tnsr_partial` for the whole run.

So: **32 scalar epilogues become 1 vector epilogue**, and the same lane-stuffing is what makes the bias
fold possible (4 vector loads at the flush instead of 4 scalar loads per pass — **THEY SAY**
`mxk8_w13.h:17`: *"scalar global loads cost about a quarter of w13 at the gpt-oss shapes"*).

**The accumulator count, read from code.** `float128 acc[4]` (`mxk8_core.h:215`) — four logical
accumulators of 128 fp32 lanes = **8 × 256-byte vector registers of fp32 accumulator**, fed by 4 bf16
planes per chunk, with `bA[4]`/`bB[4]` = 8 packed-weight registers in flight. So their hand-tuned point
is **4 independent MAC chains / ~8–16 loads in flight** — which is *consistent with* our §7.6 knee
(16 loads in flight; 32 did not help) but **does not test it**: nothing in the repo sweeps rows-per-pass.

**Follow-up experiment — a 2×2, and the prediction is our own caveat.** `MXK8_ACT_BATCH` is a
compile-time switch, so:
- **2×2:** `MXK8_ACT_BATCH ∈ {0,1}` × rows-per-pass `RPP ∈ {4,8}` (RPP is hard-coded 4 — `bA[4]`,
  `acc[4]`, `MXK8_ROWS`, the epilogue — so this half is a real code change, ~a day).
- **Prediction (from §7.6 + the caveat):** at `ACT_BATCH=1` the epilogue is cheap per accumulator, so
  8 rows should *keep* paying (more ILP, ~1.2–1.5×); at `ACT_BATCH=0` the per-accumulator epilogue cost
  doubles and the gain should shrink or invert.
- **Settles:** whether "the sign inverts when the accumulator is expensive" is a real law, on a second
  and completely different kernel family — and it gives the manual its second data point, which is what
  §7.6 says one point cannot supply.

### 2.4 The fp32 division defect — cheapest high-value probe in the repo

**THEY SAY** — `ports/deepseek-v4-1-flash/dsv41_hpu/dsv41_hpu/numerics.py:125-128`:

> *"Gaudi's fp32 division is not correctly rounded (reciprocal-multiply), and with an arbitrary e4m3
> divisor a 1-ulp error flips FP4/FP8 rounding at grid midpoints (**3.6 % of compressed-KV elements on
> the first HPU run**). Midpoints of these grids have <= 5 significant bits and the divisors <= 4, so
> midpoint × divisor is EXACT in fp32 and every decision compares the dividend against it."*

They then round without dividing at all: `round_e2m1_ratio` counts `|x| > mid_k·s` and adds a tie term
(`numerics.py:133-150`); `round_e4m3fn_ratio` settles the approximate quotient against exact midpoints
(`:152-170`).

**This is a hardware fact we do not have in any chapter**, and it is a *one-line probe* to settle:
`a/b` on device versus a float64 CPU quotient over random pairs, count the 1-ulp disagreements. If it
holds, it applies to **every** quantisation path that rounds a quotient — including fp8 KV-cache
quantisation, which is exactly the area our §3.4.1–§3.4.5 work lives in. **UNKNOWN** whether our own
fp4 path is exposed: our dequant *multiplies* by the scale (no division), so probably not — but any
quantiser we write will be.

### 2.5 The TPC's own local memory (VLM) — our chapters do not cover it

Our §1.5 carries the vendor row **ASSUMED**: *"TPC local memory: 1 KB scalar + 80 KB vector … if the
program utilizes special functions such as tanh, sin, or cos, only 16 KBs are available."* No chapter
*uses* it: `rg -i 'VLM|SLM'` over the seven chapters finds **nothing**, and `vector local`
appears only inside that vendor quote.

The cookbook is a working data point on that row:

- **READ** `mxk8_core.h:54-58`: the staged activation row is `__local__ bfloat128 v8_x[4*12]` = **48
  vectors = 12 KB**, and `MXK8_XCH` is capped at 12 chunks.
- **THEY SAY** `mxk8_core.h:57`: *"14 staged chunks do not fit: VLM is shared with the batch buffers and
  the register spills."*
- **THEY SAY** `docs/kernels/mxfp4-moe.md`: *"vector local memory is 16 KB, and static arrays plus spills
  beyond that fail at compile time"*.
- **And they use a special function** — `v_sigmoid_f32` in the epilogue (`mxk8_core.h:307`) — so the
  **16 KB** regime is the one that applies, not 80 KB.

**⇒ This is the strongest independent corroboration we have of the 16 KB special-function regime**, and
it upgrades the row from "vendor doc, ASSUMED" to "vendor doc + an independent implementation that hits
the wall at 13–14 KB". It is still not our measurement. **INFERRED** that the 12-chunk cap is the 16 KB
wall and not something else — the arithmetic (14 × 1 KB = 14 KB, plus the w2 accumulators
`float64 v8_acc[8]` = 2 KB, plus routing tables) fits, but the compiler error was not quoted.

**Follow-up:** a 20-line kernel that declares `__local__` arrays of 12/14/16/17 KB **with and without**
`v_sigmoid_f32` in the body, and record which fail to compile. **Settles** the 80 KB vs 16 KB boundary
and where the "shared with batch buffers" budget really goes. Half a day.

### 2.6 The lane id exists — `read_lane_id_{1,2,4}b_b`

**READ**, used pervasively in three kernels:
`ushort128 lid = read_lane_id_2b_b();` (`mxk8_core.h`, `mxk8_w13.h`, `mxk8_w2.h`),
`uchar256 lid8 = read_lane_id_1b_b();` (`mxk_deq.c`), `uint64 lid4 = read_lane_id_4b_b();`
(`mxk8_w13.h`, `mxk8_w2.h`).

The naming is **N bytes per element**: `1b` → 256 lanes, `2b` → 128, `4b` → 64 — which independently
confirms §4.3.1's lane table. And it is load-bearing: `pat0 = (lid >> 3) | 0x80` is the scale-broadcast
pattern (§2.7) and `v_u32_cmp_eq_b(lid4, m, …)` is the lane-selective accumulator (§2.3). **If the lane
id were not the true per-lane index, every one of these kernels would produce garbage** — and their
parity tests, which compare against a CPU fp32 reference, are said to pass.

**Our §4.1.4 says: *"there is no lane id either: the compiler's own string table carries `v_lane_id_8/16/32`
and the call is rejected."*** What was measured there is that the **`v_lane_id_*` spelling** is rejected.
That is not the same statement. See §4.2.

### 2.7 Scale broadcast without a general shuffle — and a refinement of §4.4

**The pattern.** E8M0 → bf16 is a **7-bit shift** (a scale byte *is* a bf16 exponent field), and the
per-lane scale for a 512-value chunk is selected by `v_u16_shuffle_b` with a pattern that is pure lane
arithmetic:

**READ** `mxk8_core.h:29-36`:
```c
ushort128 u_ = as_ushort128(v_u8_unpack_b(sb, SW_UNPACK_8_TO_16|SW_GROUP_0, zero8)) << 7;
bc = v_u16_mov_dual_group_all_b(u_, 0xFFFFFFFF, 0,0,0,0, MkWrA(...), u_);   // broadcast the 16 scales
#define MXK8_PICK(bc, PAT) as_bfloat128(v_u16_shuffle_b(bc, as_uchar256(PAT), 0, bc))
```
with `pat0 = (lid >> 3) | 0x80` and `pat1 = pat0 | 0x2020` (`mxk8_core.h:214`).

**Why `i>>3` is the right index — verified arithmetically.** Value `k = 512c + 4i + p` sits in scale
block `k/32 = 16c + (4i+p)/32 = 16c + i/8`. So lane `i` needs scale byte `i>>3`, and **the 4-way
interleaved host permutation is exactly what makes that true** — no per-lane shuffle is needed to line
the planes up with the activation.

**Their claim is that the whole trick rests on the host permutation.** **READ**
`python/mxfp4_tpc_moe_rt.py:48-64` (`perm_index`, `permute_x`).

**And I ran it** (`/tmp/cookbook-investigator/verify_layout.py`, CPU torch 2.13.0a0, no device):

| claim | result |
|---|---|
| permuted position `512b+128p+i` holds natural index `512b+4i+p` | **True** |
| `permute_x` applies exactly that map | **True** |
| `(4i+p)//32 == i//8` for all `i∈[0,128)`, `p∈[0,4)` | **True** |
| the 4 nibble planes × `i>>3` scales rebuild a random packed row bit-exactly | **True** (max\|diff\| = 0) |
| `(s<<7)` as bf16 == `2^(s−127)` for all finite `s` | **False at `s=0` only** |

**RAN.** That is their layout arithmetic verified independently — the one part of this repo I could
check without hardware, and it holds.

**The `0x2020` bit.** `pat1 = pat0 | 0x2020` selects the *other* 16 scale bytes for the odd chunk of a
pair, and **THEY SAY** *"for a 16-bit shuffle the source-half bit must be set in both bytes (0x2020)"*.
Our §4.4 measured `shuffle.i8` as **4-byte-group-local** and refuted 16-element replication. This is a
**different dtype and a different intrinsic** (`v_u16_shuffle_b`), and it reaches **16 values**, with
bit 5 as a source-half selector — consistent with a **32-lane window per dual group**. It does not
contradict §4.4 (which explicitly says *"not 'shuffle is a 4-byte permute, full stop'"*), but it is a
**second, larger reach on a sibling instruction** and it supplies a *meaning* for the bit our probe found
"ignored" (bit 6 for i8; bit 5 here). **Follow-up:** rerun our flat-ruler `shuffle` probe for
`v_u16_shuffle_b` / `v_i16_shuffle_b` with a non-zero `income`. **Settles §4.4 UNKNOWN item 5's reach
question per dtype**, which is an open, load-bearing item in our chapter 4.

### 2.8 E8M0 → bf16 by `<<7`: exact, and it explains our UNKNOWN 18b

**RAN**, the table row above: for `s ∈ [1,254]` the shift is **exactly** `2^(s−127)`; `s = 255` gives
`inf`; and **`s = 0` gives bf16 `0.0`, not `2^-127 = 5.877e-39`**.

Our §3.4.7 UNKNOWN **18b** records, for the vendor op, *"byte 0 → scale 2^-127 … is flushed to 0.0 for
every nibble … measured behaviour, unmeasured cause"*. The cause is now available: **constructing the
scale by placing the E8M0 byte in the bf16 exponent field makes byte 0 the zero/subnormal encoding, so
any implementation built this way must flush it.** (A bf16 *can* hold 2^-127 as a subnormal — bit
pattern `0x0040` — the shift simply never produces it.) **INFERRED, demonstrated on CPU.** It does not
prove the vendor does the same construction, but it makes the measured behaviour the *expected* one, and
it is the natural implementation. Likewise `s=255 → inf` matches UNKNOWN **18a** (byte 255 is
non-finite), which their kernel inherits.

### 2.9 Interleaved gate/up for the bias path — closes §3.9 item 17's layout half

Our §3.9 item 17 records a **doc-against-doc disagreement** on the gate/up split of
`bias_mxfp4_fused_weights` (`gpt_swiglu = true`): the bridge test docstring says *interleaved*,
vllm-gaudi's `minimax_m3_sparse.swiglu_oai` says *halves*. Three independent sources now say
**interleaved**:

1. **READ (our tree)** `vllm/csrc/libtorch_stable/activation_kernels.cu:409-421` — vLLM's own
   `swigluoai_and_mul` carries the comment *"Interleaved gate/up: input has [gate0, up0, gate1, up1,
   …]"*, and its formula is `g = min(gate, limit); u = clamp(up, ±limit); (u+1)·g/(1+exp(−g·alpha))`
   with `alpha = 1.702`, `limit = 7.0` (`ops.h:570-571`). **This also confirms the cookbook's SwiGLU-OAI
   constants from our own tree** — they are not "their claim", they are canonical.
2. **READ (cookbook)** `ports/deepseek-v4-1-flash/dsv41_hpu/dsv41_hpu/quant.py:26-31` — the checkpoint is
   converted to `w13_layout = "interleaved_gate_up"` because *"the clamp-capable bias kernel needs
   interleaved_gate_up"*; and `vllm_model.py:160` puts the up-row bias at `b13[1::2]`, i.e. odd rows,
   which only makes sense interleaved.
3. The bridge docstring our manual already cites.

And the dissenting source is self-labelling: `vllm-gaudi/vllm_gaudi/ops/hpu_fused_moe.py:64` sets
`_SWIGLU_OAI_ACTIVATION = "swigluoai_**un**interleave"` — vllm-gaudi's *own name for the other layout*,
for MiniMax-M3, computed unfused because *"the Habana fused-MoE kernel only supports silu/gelu"*.

**⇒ item 17's layout half can be closed** (interleaved; the halves reading is a different, uninterleaved
path). The *device* semantics of the vendor op's gpt_swiglu path is still unmeasured — but note the
cookbook **runs that op in production for DSv4.1-Flash**, which is at least a working existence proof.
**Follow-up:** the `bias_*` arm our item 17 already specifies. Note the cookbook gives a cheap variant:
use `has_bias=True` with **zero** biases to isolate the layout without the clamp.

### 2.10 MME-vs-TPC engine choice by shape, with numbers

**THEY SAY** — `ports/deepseek-v4-1-flash/dsv41_hpu/dsv41_hpu/mhc.py:20-23`:

> *"`DSV41_MHC_TPC_DOT=1` for decode-sized inputs (<= 16 rows) the mix projection `x [N, 4 x 5120]` fp32 ·
> `fn [24, 20480]` fp32 as a multiply + row sum on the TPC instead of an fp32 GEMM on the MME
> (**~54 → ~8 us per call**, 80 calls per decode step; **max relative error vs fp64 1.9e-7 instead of
> 3.4e-6**)."*

That is a **6.75× win for the TPC over the MME at M ≤ 16, N = 24, K = 20480**, and **the TPC arm is also
more accurate** — which is the interesting half: an fp32 GEMM that accumulates 20,480 terms on the MME
loses ~18× more accuracy than a TPC row-sum. Our §3.1/§3.2 have the M=1 GEMV ceiling and the
prefill/decode engine split, but I do not think we have a **small-N** datapoint with a like-for-like
A/B *and* an accuracy comparison. **Follow-up:** reproduce at that exact shape with our own harness
(cheap, one card); it is a direct test of "MME is the wrong engine for tall-skinny-and-narrow".

**The prefill counterpart of the same law** is their `mxk_deq` kernel: unpack each local expert **once**
to bf16 and then run the **stock MME** MoE op — TTFT −22 %/−26 % at EP 8, and **bit-identical** output
(**THEY SAY**). So the engine choice flips with shape *within one kernel family*: TPC-fused-GEMV at
decode, TPC-unpack-then-MME at prefill. That is a clean, quotable statement of the law.

### 2.11 Four lazy-mode custom-op hazards (silent-wrong-result class)

All **THEY SAY** (`docs/kernels/mxfp4-moe.md`, "Limitations and pitfalls"; corroborated in code):

1. **Legacy lazy-mode custom ops are cached by input *signature*, not by GUID.** *"In one process, a
   second custom op with the same input shapes silently returns the first op's result."* Their two
   kernels have 5 and 6 tensor inputs so they cannot collide, and `tests/parity_hpu.py --impl sig` is a
   gate on it. **This is the single most dangerous item in the repo for us** — it is silent, it is
   wrong-answer, and any custom-op pair we add is exposed. Our manual has nothing on it
   (`rg "input signature"` → no hits).
2. **`PT_HPU_LAZY_ACC_PAR_MODE=0` is required.** With the lazy accumulation thread on, a custom op whose
   input is a view of a tensor produced in the same lazy graph **fails intermittently**. Not in our
   manual.
3. **A view taken in an *earlier* lazy graph reads the wrong data** for a legacy custom op (native ops
   read it correctly). Take the view in the same graph as the call.
4. **`GC_KERNEL_PATH` must be exactly one regular file when the recipe cache is on** (*"Path is not a
   regular file"*), and **the recipe cache key does not cover the kernel library**, so a cache written
   by one build can be handed to another — they move the cache to `<dir>_mxk-<version>/` to defend. Our
   manual has `GC_KERNEL_PATH` material (§3.4.7's `synStatus 26` subsection and §3.4.8) and
   recipe-cache material (§2.10, §5.3), but **not these two restrictions**. Their "must be set before the first HPU
   op" *corroborates* our late-write finding from an independent implementation.

### 2.12 Graph count: bigger is not better past a point

**THEY SAY** (`docs/ports/deepseek-v4-1-flash.md`): *"A whole-model HPU graph never finished compiling
(compile time grows faster than linearly with graph size), so decode replays one graph per 10 layers"*
(`DSV41_LAYER_GROUP=10`). Our §3.5 argues *fewer, larger regions*, and our §0 fact 3 prices the graph
count. This is the counter-pressure: **there is an upper bound on region size set by compile time, and
it is not linear.** We have no measured compile-time-vs-region-size curve — our §4.2.10 has a *launch*
cost that is sublinear in bundle count, which is the opposite direction and a different quantity.

**Follow-up:** the curve. Compile one body at 1/2/4/8/16/32/43 layers' worth of nodes and plot compile
time. **Settles** the practical region-size ceiling for our port — and their `10` is a datapoint.

Related, and also new: **the graph cache keys on the input's *view state*** — *"Passing views where
warm-up captured fresh tensors re-captured every warmed graph in serving, and memory grew until a long
request failed to allocate."* That is a memory-exhaustion failure mode from a *correctness-neutral*
coding choice, and it is exactly the kind of thing our §3.5/§5 material should warn about.

### 2.13 Prefill: where it actually goes, for our model family

Our README lists *"No prefill code example … the evidence base is decode-heavy"* as a known gap. This is
the closest thing to prefill evidence for DSv4F-shaped models that I have seen outside our own work:

**THEY SAY** (`docs/ports/deepseek-v4-1-flash.md`):
- *"the routed MoE is only about **18 %** of a short 4096-token chunk; the per-query key gather and, from
  **32 K**, the indexer's full stable sort dominate"*;
- fixes: banded window attention + the compressed part under one softmax; a **sort-free threshold top-k**
  returning the same sets; the index dot as a bf16 GEMM (bit-identical *because both operands are
  fake-FP4*);
- TTFT 16 K→120 K: 9.93/21.59/48.70/117.61 s → 7.35/14.97/32.56/75.89 s (same-node A/B).

If that 18 % is even roughly right, it re-ranks our prefill priorities: we have been treating the MoE as
the prefill story, and for this architecture the **indexer sort and the key gather** may be the span
worth attacking. **Follow-up:** one prefill profile at 4 K/16 K/32 K on our own arm, decomposing into
routed MoE / key gather / indexer sort / attention. **Settles** whether our prefill work is aimed at the
right 18 %.

### 2.14 Measurement method worth stealing: the multi-rank straggler

`docs/results/moe-decode-step.md` is a *method* page and it is better than ours for multi-card MoE:

1. **Trace every rank** — *"Rank 0 alone hides the straggler."* Their finding: expert kernels were
   **19.5 ms** of a 39 ms step at the slowest EP rank against a **9.6 ms** mean.
2. **Build the critical path per collective** (slowest rank's compute before each all-reduce + the
   collective's latency).
3. **Decide whether the straggler is a hot rank by simulation, not by guessing**: record real routing,
   look up per-route kernel cost, take the max over ranks, **fit placements on the first 60 % of steps
   and evaluate on the last 40 %**; validate the simulator against traces (18.44 simulated vs 17.9–19.5
   traced). Result: **no hot rank** — random placement matches contiguous.
4. Their conclusion — *"Not node-bound. Compute engines are busy 15.4 ms of a 37.4 ms device span; the
   idle time is waiting at all-reduces, not gaps between nodes"* — is an **independent corroboration of
   our occupancy model** (§0 / §2.6): the device is not idle between nodes, it is waiting at
   collectives. Worth citing as such.

Also there: **the per-route fixed cost decides EP×TP** (*"47 µs at full width, 25 at half, 17.7 at
quarter"*), which is a cleaner decision rule than we have.

### 2.15 tpc-clang pitfalls, from someone who fought it

**THEY SAY** (`docs/kernels/mxfp4-moe.md`, "tpc-clang pitfalls met while writing the kernels", 1.24.1 /
tpc-clang 2.7.33). Six of the eight are new to us; two corroborate our §4.4/§4.5:

| pitfall | ours? |
|---|---|
| templates crash the front end | new |
| loops on a constant parameter of an inlined function can be **miscompiled** → stamp variants out with the preprocessor | new (this is *why* the kernels are macro-stamped) |
| VLM 16 KB; static arrays + spills beyond it fail at compile time | new (§2.5) |
| a loop storing a constant zero to local memory becomes a `memset` the backend cannot lower → store a **runtime** zero | new; **READ** the workaround at `mxk8_w2.h:127-128` |
| the backend cannot lower a VLM/SLM **memmove** → use cursors, not shifting copies | new; **READ** `mxk8_w2.h:116` |
| for a 16-bit shuffle the source-half bit must be set in **both** bytes (`0x2020`) | new; refines §4.4 (§2.7) |
| two back-to-back **partial tensor stores** with the same size and offset drop the second | new — silent data loss |
| the glue must **reject unexpected dtypes**, or the graph compiler re-types the output and corrupts it silently | new; complements §4.5's operand-aware gate |

And one from `mxk_deq.c:46-48`: *"Merging two shuffles through the income with disabled pattern lanes does
not work: the group bit `0x20` of the enabled lanes is then ignored."* — a direct, if unexplained,
data point for our §4.4 open question.

---

## 3. Claims we should not repeat without verifying

### 3.1 The device-trace paragraph is internally inconsistent — do not quote it

`docs/kernels/mxfp4-moe.md`, "The bottleneck":

> *"One decode step ran about **124,000 cast calls** (about **8 µs each**, **47.6 ms** of TPC time in a
> 97 ms step), **33,000 small GEMMs (21.6 ms)** and **49,000 DMA copies**."*

**124,000 × 8 µs = 992 ms**, not 47.6 ms — off by ~20×. Only 47.6/124,000 = **0.384 µs** per call makes
the total work. The three figures are consistent only if 47.6 ms is **engine-busy wall time across ~24
TPCs** (124,000 × 8/24 = 41.3 ms), i.e. the sentence mixes a **summed** quantity with a **wall**
quantity — while the GEMM figure (33,000 × 0.65 µs = 21.6 ms) reads as **summed**. And 124,000 cast
nodes for a 43-layer model at batch 4 (≈ 688 routed experts) is implausible on its face unless the
counter is per-chunk rather than per-node.

**No trace artifact is shipped**, so none of it is checkable. The *qualitative* claim survives on other
numbers (0.2–0.5 TB/s packed against a 2.2 TB/s HBM figure) — quote that, not the census.

### 3.2 "5× above the HBM read floor" is 5× *below* it

> *"one local route … takes about 30 µs. That is about **0.44 TB/s** of packed bytes, about **5x above**
> the HBM read floor."*

13 MB / 30 µs = 433 GB/s ✓, and 2.2/0.44 = 5.0 — so 0.44 TB/s is **5× below** the floor they state two
paragraphs earlier (*"HBM can deliver about 2.2 TB/s per card"*). The sign of the sentence is inverted;
it should read "5× below", which is also the only reading that supports their own conclusion
(*"5x above"* would mean the kernel is memory-bound and near the ceiling, i.e. the opposite).

**This is the second internal inconsistency from this author** (our §3.9 item 23 found ~107 % of a stated
BF16 peak in the sibling repo). The per-call and end-to-end *tables* in this repo, by contrast, I checked
and they are self-consistent (T32 stock/kernel ratios, the 1.96 × 1.13 × 1.14 = 2.525 chained-GLM
decomposition, 43.3 → 102.8 = 2.37× for Muse, and the "per card at 32 in flight" column, which is the
aggregate column divided by cards — all check out). So: **the tables are careful; the prose is not.**

### 3.3 "The clamp runs exactly" — it is off by 1/128 at the up-clamp boundary

See §2.1. **INFERRED** arithmetic. Fine in practice; not the word "exactly".

### 3.4 Headline inconsistency: 2.5× / 3× / 3.2×

`README.md` says *"Up to 2.5x faster decode"*; `docs/kernels/mxfp4-moe.md` opens with *"Up to 3x"* and
then quotes *"gpt-oss-120b up to 3.2x per stream"*. All three are defensible against different rows of
their own tables (2.5× is GLM-5.3, 3.2× is gpt-oss TP2+EP at 4 in flight) — but pick one and name the
row, which is exactly the manual's "every figure names its population" rule.

### 3.5 Everything end-to-end

No image, no log, no JSON, no trace is shipped. `bench_moe.py` and `run_parity.sh` *write* JSON and
print `PARITY PASS`, and **not one output file is in the tree**. Every number in §2 that is a
performance figure is **THEY SAY** and nothing stronger. This includes the ones I most want to believe:
the per-call µs tables, the parity rel-L2 table, and TTFT −22 %.

---

## 4. Contradicts our manual

### 4.1 §3.4.7 checklist item 1 — the vendor op is **not** excluded for a clamped model

Our §3.4.7 says, of `bias_mxfp4_fused_weights`/`gpt_swiglu`: *"This op is plain SILU with no clamp, and
the clamped path is unreachable through it. A model that clamps the gate/up is not a candidate — and
**this check alone rejects DSv4F**."*

**The clamp is reachable by constant folding** (§2.1), and the same author ships it for the same model
family. Our rejection was of the *unfolded* use, and we did not consider the fold. **§3.4.7 item 1 must
be rewritten** — not deleted: the *checklist* is still right that an unfolded clamped model is not a
candidate, and the fold has its own constraints (E8M0 headroom, a −1 up-bias, a scaled routing weight,
and the 1/128 boundary shift).

### 4.2 §4.1.4 — "there is no lane id" is too strong

Our §4.1.4: *"there is no wave, no warp, no lockstep width below the register … there is no lane id
either: the compiler's own string table carries `v_lane_id_8/16/32` and the call is rejected."*

A lane id exists as **`read_lane_id_{1,2,4}b_b`** and is used as the *basis* of three working kernels
(§2.6). What was measured was that the **`v_lane_id_*` spelling** is rejected — a narrower fact.

**This matters twice.** (a) It is a **capability we told ourselves we did not have**: per-lane constants,
lane-selective accumulation and shuffle patterns are all built on it, and two of the techniques in §2.3
and §2.7 are *only* expressible with it. (b) It weakens one of the supports for "no lockstep width below
the register" — though that conclusion stands on its own timing experiment (256-lane vs 64-lane, same
skeleton, same time) and should be re-stated on that evidence alone.

### 4.3 §4.2.8 — `WE` *is* requestable, in the only sense that matters

Our §4.2.8 concludes *"`WE` is not requestable"* and that the only lever is the declared access pattern.
The cookbook's glue does it directly and simply:

**READ** `kernels/mxfp4-moe/src/mxk_tpc_lib.cpp:94-106`:
```c
static uint32_t tpcPoints(const HabanaKernelParams* in, uint64_t units) {
    uint32_t npts = in->maxAvailableTpc ? in->maxAvailableTpc : 24;
    if (npts > units) npts = (uint32_t)units;
    return npts;
}
static void setIndexSpace(const HabanaKernelParams* in, HabanaKernelInstantiation* out, uint32_t npts) {
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = npts;
    allRequired(in, out);
}
```

**A 1-D index space of exactly `maxAvailableTpc` points** — which our own closed form (`WE = 24 if
24 | G`) then makes 24, deterministically. Every TPC gets `allRequired = 1` with every dim mapped to
`indexSpaceDim = 0`, i.e. **the whole tensor**, and the kernel splits the work itself with
`get_index_space_offset()` / `get_index_space_size()`. So: **`WE` is not settable by API, but it is
obtainable, and the recipe is "declare one point per TPC and divide the work in the kernel".** That is a
better answer than "influence `a` and check the dump", and it also surfaces a field our manual does not
mention (`maxAvailableTpc`). Worth adding to §4.2.8/§4.2.9 as a *positive* technique.

### 4.4 §1 — the per-launch host cost differs 6× from ours, and the populations differ

Our §2.5 carries **~13 µs marginal per launch** against a 46.6 ms host budget. The cookbook states
**74 µs** per graph launch (322 launches/step) and **~82 µs** (183 launches, ~15 ms). These are
probably **not** the same quantity — ours is a *marginal fit slope* over a synthetic one-node recipe;
theirs is *average host time per launch* in a real engine — but **our manual does not say which it is**,
and the two numbers are close enough to be confused. Flag it; do not resolve it by argument. A real
engine step with a known launch count would settle it.

### 4.5 §3.2.1 — their kernel sits *below* our M=1 GEMV cliff and is still fine

Our §3.2.1 says *"keep a decode weight tile at or above ~16 MB"* because the M=1 GEMV collapses below
that. Their per-route tile is **~13 MB** and runs at **0.44 TB/s** — worse than our 1.14 TB/s at 4.2 MB —
and is *not* a problem, because it is **issue-bound, not memory-bound**. Not a contradiction, but a
boundary worth stating: **our §3.2.1 curve describes a memory-bound GEMV; an issue-bound one can sit 5×
under the floor and still be the right design** (their answer to which is "fewer vector ops per value",
i.e. repack the weights, which they decline to do).

### 4.6 Not contradicted, but strengthened

- **§1.5's ASSUMED VLM row** (80 KB / 16 KB with special functions) gains its first independent
  implementation data point (§2.5).
- **§3.4.7 trap 1** (one out-of-range scale NaNs the layer) is independently guarded against by the fold
  (`vllm_model.py:156`).
- **§3.4.8 / the late `GC_KERNEL_PATH` write** is independently corroborated (§2.11 item 4).
- **Our occupancy model** (§0, §2.6) is independently corroborated by their "not node-bound, waiting at
  all-reduces" (§2.14).
- **Chapter 7 (SRAM): nothing.** They never touch the 48 MiB pool. No corroboration, no contradiction.

---

## 5. The repo's own evidence quality

| question | answer |
|---|---|
| tests? | **Yes, and real.** A CPU layout test (`tests/test_layout_cpu.py`) that re-derives the kernel arithmetic from raw bytes against an fp32 reference; a device parity gate with a **pre-declared pass criterion** (`rel_l2 < 3e-3` **and** not worse than the stock op); a per-variant `--impl sig` gate for the signature-collision hazard; a library gate that checks the stock GUID list is unchanged and in order; an op gate. `fixes/` has 43 patches, 29 of them with their own test file. |
| CI? | **Yes, and it runs** — ruff, yamllint, `mkdocs build --strict`, a CPU pytest run, plus a `fixes-on-release` job against two pinned upstream commits. **But it explicitly skips everything needing torch/vLLM/a device**, so **no kernel gate has ever run in CI**. |
| measurement artifacts? | **None. Zero.** No JSON, no log, no trace, no profiler output, no benchmark result file anywhere in 13 MB. `bench_moe.py --json` exists and writes files; none are checked in. |
| reproducible? | **Partly.** I verified the layout arithmetic on CPU (§2.7) — that part is real and it holds. Everything performance-related needs an HPU plus model code that is **not published** (`docs/ports/index.md`: *"The integration code is not published yet"* — though for DeepSeek-V4.1-Flash and several others it now **is** under `ports/`). |
| issues / history | **23 commits, two days, single author, no issue history.** A history dump, not a development record — so commit messages carry no investigative trail. |

**Verdict on evidence: prose with a good skeleton.** Better engineering hygiene than the sibling repo
by a wide margin (it has gates with pre-declared criteria, which is more than most vendor code), and
still **not a source of measurements**. Treat every number as a hypothesis with a described method —
which is more than nothing, and less than a result.

---

## 6. What I could not determine

| # | UNKNOWN | what would settle it |
|---|---|---|
| 1 | Whether the clamp fold is exact **on device** (my arithmetic says ≤1/128 at the up-clamp boundary; their word is "exactly") | the §2.1 experiment |
| 2 | Whether their `MXK8_*` design is near a ceiling — nobody swept rows-per-pass, and the build lacks `vliw-td` | §2.2 + §2.3 |
| 3 | The `124,000`/`8 µs`/`47.6 ms` census — no trace, and internally inconsistent | a trace of one decode step, or drop the paragraph |
| 4 | Whether the `v_u16_shuffle_b` window is 16 or 32 lanes, and what bit 6 does | our flat-ruler probe, per dtype, non-zero `income` |
| 5 | Whether Gaudi fp32 division is really reciprocal-multiply (and whether our quantisers are exposed) | the one-line `a/b` probe (§2.4) |
| 6 | The real 16 KB VLM boundary and what else consumes it | the `__local__` ladder (§2.5) |
| 7 | Whether "compile time grows faster than linear in graph size" — their reason for 10-layer graphs | a compile-time vs region-size curve (§2.12) |
| 8 | Whether their end-to-end numbers are real | an artifact. There is none. |

---

## 7. If we do one thing

**Run the clamp fold on one card** (§2.1). It is the only item here that changes a decision we have
already made in writing — our §3.4.7 rejects the vendor op for our model family on a ground that turns
out to be foldable — and it is a day's work with a CPU reference we already have.

The cheapest second thing is the **fp32 division probe** (§2.4): one script, no model, and it either
gives us a hardware fact for chapter 1/3.4 or refutes a claim we would otherwise have carried.

---

*Files read in full or substantially: `kernels/mxfp4-moe/src/{mxk8_core.h, mxk8_w13.h, mxk8_w2.h,
mxk_deq.c, mxk_tpc_lib.cpp}`, `kernels/mxfp4-moe/{build.sh, README.md}`, `python/mxfp4_tpc_moe_rt.py`,
`tests/{parity_hpu.py, run_parity.sh, test_layout_cpu.py}`, `bench/bench_moe.py`,
`docs/{kernels/mxfp4-moe.md, results/platform-findings.md, results/moe-decode-step.md,
ports/deepseek-v4-1-flash.md, recipes/deepseek-v4-1-flash-1017.md}`, `.github/workflows/ci.yml`,
`ports/deepseek-v4-1-flash/dsv41_hpu/dsv41_hpu/{vllm_model.py, moe.py, quant.py, mhc.py, numerics.py,
mxk.py}`, `ports/deepseek-v4-1-flash/kernels/dsv41_sinkhorn/dsv41_sinkhorn.c`.*
*Read partially: `csa.py`, `paged.py`, `replay.py`, `runner_patch.py`, `docs/fixes/*` (titles),
`docs/kubernetes/*` (not relevant to chapters 1/3/4/7).*
*The one thing I ran: `/tmp/cookbook-investigator/verify_layout.py` (CPU torch, no device).*
