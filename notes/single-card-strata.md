# Working notes: `gaudi2-single-card-strata`

**Status:** investigation notes for our own team, not a manual chapter.
**Subject:** `src/gaudi2-single-card-strata` — a Gaudi2-native backend (`hpu/`) under the upstream
"Strata" serving stack, targeting Qwen3.8-Flash-Next (512-expert MoE, 48 layers, 256k context) on one
Gaudi2 HL-225.
**Question asked of me:** verify or refute a list of claims a previous pass made from *comments only*.
**Method:** I read the code. Where I say "read from code" I name `file:line`. I ran nothing.

---

## 0. How to read this file — basis tags

Every factual sentence below carries one of:

| tag | meaning |
|---|---|
| **READ** | I read it in the code at the cited `file:line`. A fact about what the code does. |
| **THEY SAY** | The project asserts it, in a comment or a doc. Their claim, not evidence. |
| **INFERRED** | My inference from READ facts, arithmetic, or published numbers. Say so if you rely on it. |
| **MEASURED** | Nobody measured anything for this file. I ran no benchmarks. Where I quote *their* measured number I tag it **THEY SAY (measured)**. |

**The single most important structural fact about this repository:** almost every performance claim in
it is a comment citing a measurement that **is not in the repository**. `hpu/PROGRESS.md` is cited from
`hpu/include/strata/hpu/batch_slots.hpp:29` and `hpu/include/strata/hpu/kv_q4_cell.hpp:14` and
`hpu/src/qsa_state_io.cpp:24` — it does not exist here (not in git, not on disk, not gitignored;
`git ls-files | grep -i progress` returns only `include/strata/core/progress.hpp`). Likewise
`work/DESIGN_PREFILL_STREAMING_20261024.md §18` is cited from `hpu/tools/generate.cpp:554`, and
`work/` does not exist. So **there is no checkable measurement record** — only the surviving comments,
a set of hardware-only probes, and a set of host-only correctness tests. Every "measured" number I
relay below is *their* number with no artifact behind it in this tree. **READ** (absence).

That is also, I think, why the previous comment-only pass went wrong: the comments here are unusually
confident and unusually specific, which reads like evidence and is not.

---

## 1. Executive summary — the six things that matter

1. **The TPC-vs-MME claim is not supported by anything in the repository, and the switch that
   implements it is OFF in the author's own shipped production config.** `STRATA_HPU_GEMV_TPC` gates a
   blanket substitution of every `[1,K]×[K,N]` bf16 GEMM onto a TPC kernel. There is no A/B, no bench,
   no probe and no test of the MME at one row anywhere in the tree. §3 below.
2. **The substitution cannot fire on the shipped hot path**, because that path is a *4-row* DFlash
   verify graph and the gate requires `a.desc.shape[0] == 1`. It can only fire on the single-token
   chain. `hpu/src/recipe.cpp:311` vs `hpu/include/strata/hpu/dflash.hpp:29`. §3.7.
3. **Their own working example of "the MME tiles this badly" is a narrow-*N* argument, not an M=1
   argument** — the HC down/inject projection, `[rows, HC·H] × [HC·H, rank+HC]` with `rank+HC = 324`.
   That switch (`HC_DECODE_TPC`) *is* shipped. The GEMV_TPC comment reuses the intuition with the wrong
   bound. §3.5.
4. **Their read-bandwidth probe is the same instrument as our §7.6 experiment and stops one step
   short of our result:** it sweeps independent accumulators/loads-in-flight over `{1,2,4,8}` and never
   tests 16. Our §7.6 says 16 is the knee. So this is *not* a contradiction of our finding — it is a
   gap in their sweep, and their kernels consequently sit at 5–8 loads in flight. §4.3. This is the
   cheapest experiment we can steal.
5. **The port is not bandwidth-bound and is nowhere near the HBM roofline.** Its own instrument says
   the five-tile expert GEMV runs at **237 GB/s = 10 % of HBM peak** at the decode shape
   (`hpu/tpc/iq4q8_gemv_tile.c:6`). Arithmetic on its published 18.63 ms / 2.9-token verify round and
   the code's own per-expert byte count lands the whole step at 87–350 GB/s effective, i.e. **3.5–14 %
   of peak** (**INFERRED**). 155 tok/s is a small model at a tenth of roofline, not a memory-path
   achievement. §2.4.
6. **The one lever they ship that we should copy outright is *occupancy by K-splitting*, not ILP.**
   A decode token has 11 MoE slots; they split each projection into 2 column parts so that
   `11 × 2 = 22` index-space members fill the 24 TPCs, and they drop back to 1 part for a 4-row
   verify window because 44 slots already fill them. `hpu/src/resident_experts.cpp:194`,
   `hpu/src/resident_experts.cpp:194,199`, `hpu/tpc/route_topk.c:12-17`. §4.6, §9 item 2.

---

## 2. The port, from the code

### 2.1 Model geometry (this is *not* our model, and that changes every comparison)

**READ** `include/strata/core/geometry.hpp:20-72`, refused against a loaded file at
`hpu/src/model.cpp:176-178`:

| | value | source |
|---|---|---|
| `n_embd` (H) | **2560** | `geometry.hpp:27`; `n_embd>2560` refused, `model.cpp:178` |
| `n_layers` | 48 | `geometry.hpp:28` |
| `n_vocab` | **248320** | `geometry.hpp:27` |
| GDN layers | 36 (`qsa_interval=4`) | `geometry.hpp:29,86-87` |
| `ssm_state_size` / `ssm_k_heads` / `ssm_v_heads` / `d_conv` | 128 / 16 / 48 / 4 | `geometry.hpp:34-37` |
| `ssm_conv_channels` / `ssm_value_dim` | 10240 / 6144 | `geometry.hpp:38-39` |
| QSA layers | 12; `n_head=24`, `n_head_kv=2`, `head_dim=256`, `n_rot=64` | `geometry.hpp:42-46` |
| indexer | `idx_q_heads=4`, `idx_key_dim=128`, `idx_block=4`, `idx_top_k=2048` | `geometry.hpp:47-50` |
| hyper-connection | `hc=4` residual streams, `hc_lr=320` | `geometry.hpp:53-54` |
| MoE | 512 experts, **10 routed** + 1 shared, `n_ff=640` | `geometry.hpp:57-60`; `n_expert_used!=10` refused, `model.cpp:178` |

Per-expert resident bytes, **READ** `hpu/src/resident_experts.cpp:24`
(`iq4_resident_bytes(rows,columns) = rows/2*columns + rows*(columns/32)*4`):

- gate/up (`rows=2·ff=1280`, `columns=H=2560`): **2,048,000 B**
- down (`rows=H=2560`, `columns=ff=640`): **1,024,000 B**
- ⇒ **3.07 MB per expert per layer**; × 512 × 48 = **75.5 GB** — matches the README's "~76 GB".
  (**INFERRED** arithmetic on **READ** constants; the README figure is **THEY SAY**.)

**Consequence for our comparisons.** Our manual's §3.2.1 M=1 GEMV curve is `[1,4096] @ [4096,N]` —
**K = 4096, tiles of 4.2 / 16.8 / 67.1 MB**. This model's dense one-row GEMMs are `K = 2560` (or 6144),
`N` from 324 to 248320, i.e. tiles from **1.7 MB to 1.27 GB** and typically 5–70 MB. **Their tiles are
much smaller than ours.** Any bandwidth comparison between the two ports has to carry the tile size
with it, exactly as our §3.2.3 says.

### 2.2 What the shipped configuration actually runs — and the flags the brief names

**READ** `hpu/deploy/strata-hpu-production.json` (identical to `hpu/docker/strata-hpu.json`; the
README says this file "is the combination actually in use", and the README's 2026-10-27 note names
`GDN_STEP_ROWS` and `GEMV_PARTS_GU_WINDOW`, both of which are present — so the config is current).

**This is the most load-bearing table in these notes.** Every flag the brief asks me to investigate,
and whether the author ships it:

| flag | brief's claim | in production? |
|---|---|---|
| `STRATA_HPU_GEMV_TPC` | TPC beats MME at one row | **OFF** |
| `STRATA_HPU_GEMV_PIPE` | load pipelining | **OFF** |
| `STRATA_HPU_GEMV_PF` | software-pipelined prefetch | **OFF** |
| `STRATA_HPU_GEMV_TILE` | one accumulator, 8 loads in flight | **OFF** |
| `STRATA_HPU_MOE_GROUP` | grouped expert reads | **OFF** |
| `STRATA_HPU_MOE_PAIR` | paired grouped reads | **OFF** |
| `STRATA_HPU_MOE_GROUP_BLOCKS` | block-skipping grouped order | **OFF** |
| `STRATA_HPU_DECODE_RECIPE_PER_LAYER` | per-layer recipe instances | **OFF** |
| `STRATA_HPU_TPC_ACCESS` | affine access patterns | **OFF** |
| `STRATA_HPU_TPC_COLPAIR` | lane-per-row FP8 GEMV | **OFF** |
| `STRATA_HPU_TPC_HILO` | FP8 hi/lo TPC GEMV | **OFF** |
| `STRATA_HPU_TPC_EXPERTS` | TPC FP8 expert GEMV | **OFF** |
| `STRATA_HPU_CACHED_GDN_STEP` | GDN step without scalar loads | **ON** |
| `STRATA_HPU_GDN_STEP_PIPE` | 16 state loads ahead | **ON** |
| `STRATA_HPU_HC_DECODE_TPC` | TPC replaces the MME for HC down | **ON** |
| `STRATA_HPU_Q8_PACKED` | packed Q8 activation layout | **ON** |
| `STRATA_HPU_MERGED_GEMM` | one GEMM per layer's projections | **ON** |
| `STRATA_HPU_MOE_MME` (+`_TILE=128`, `_SLICE`) | grouped MME prefill MoE | **ON** |
| `STRATA_HPU_GDN_STEP_ROWS` | 4 recurrent steps in one kernel | **ON** |
| `STRATA_HPU_GEMV_PARTS_GU_WINDOW=1` | parts for a multi-row window | **ON** |

⇒ **Of the ten switches behind claims 1–5, nine are off.** The ones that are on are the ones whose
comments describe a *fusion*, a *layout* or an *occupancy* change — not a "this engine is faster than
that engine" claim. Read that as the author's own verdict: **he built the pipelining/grouping/TPC-GEMV
variants, measured them, and did not ship them.** He does not say why. That is the first question to
ask him, and it is not answerable from this tree. (**READ** for the table; **INFERRED** for the
motive, flagged as such.)

Also **READ**: `--fp8-experts` is on the command line but is *not* the expert compute path here —
`hpu/src/model.cpp:189-193` makes it a *precondition* of `RESIDENT_IQ4` ("requires STRATA_HPU_CHAIN and
--fp8-experts (its router/cache shell)"), and `RESIDENT_IQ4` is what installs the IQ4 expert store and
the IQ4 GEMV. So the FP8 GEMV kernels (`fp8_gemv*.c`, and hence `TPC_COLPAIR`/`TPC_HILO`) are the
*older, non-resident* expert path and are dead in the shipped configuration.

### 2.3 What the kernel technique actually is — sampled broadly

I read ~25 of the ~100 kernels plus `glue.cpp` in full. The technique is consistent and describable:

1. **Everything that can be a vector op is one.** The unit is a 2048-bit vector register:
   `float64` = 64 F32 lanes, `char256`/`uchar256` = 256 int8 lanes, `bfloat128` = 128 bf16,
   `bfloat256`/`minifloat256` = 256 8-bit float. Loads are `v_*_ld_tnsr_b` (a 256-byte burst per
   instruction) and the arithmetic is done with the wide fused ops
   (`v_i8_mac_b`, `v_bf16_mac_acc32_b`, `v_convert_f8_to_bf16_all_b`, `v_u8_shuffle_b`).
2. **Weights are quantized into a layout the vector unit can *decode in place*.** The IQ4 codebook is
   held in one 256-byte register (`lut`), and a code byte `| 0x80` indexes it with one
   `v_u8_shuffle_b` — one shuffle per 256 code values, two shuffles per column pair (low/high nibble).
   Layout is chosen so no bit-fiddling is needed at run time; `hpu/tpc/iq4q8_gemv.c:5-14` documents
   the whole mapping including the shuffle-mask bit semantics.
3. **The reduction is pushed into the MAC.** `v_i8_mac_b` accumulates a 256-wide int8 dot product into
   four int32 lanes; the four lanes are then scaled and summed. The per-row writeback is the expensive
   part, and it is what the "colpair" layout removes entirely (see §7.4).
4. **Small scalar control is loaded as a *vector*, never per-element.** `hpu/tpc/gdn_step_rows4.c:33`
   says it outright: *"vector loads whose lane 0 is the head (scalar global loads wait ~0.5 µs each)"*.
   The same idiom appears in `route_group_order.c`, `moe_mme_gather.c:12` and `q8_quant_packed.c`.
5. **The decode path is a long chain of small fused nodes**, and the *number of nodes* is a first-class
   cost. Production MoE per layer, **READ** `hpu/src/resident_experts.cpp:196-232`:
   `route_topk` → `q8_quant_packed` → gu GEMV ×2 parts → `reduce_sum` → `split` → `sigmoid` →
   `multiply` → `multiply` → `q8_quant_packed` → down GEMV → (reduce) → … i.e. **≥10 nodes per layer's
   FFN, × 48 layers**, before GDN/QSA/HC. `MOE_ACT_FUSED` would fuse the SiLU chain and is **OFF**.
6. **Prefill is a different engine: bucket by expert, expand to BF16, one grouped MME GEMM per tile.**
   `moe_mme_count` → `moe_mme_prefix` → `moe_mme_place`/`scatter` → `moe_mme_tilemap` →
   `iq4_expand_tiles` (dequant IQ4→BF16 into the GEMM's B operand) → `moe_mme_gather` (build A) →
   batched GEMM → `moe_mme_act` → down GEMM → `moe_mme_join`. Tile 128 rows, chunked to bound the
   expanded weights. Nice trick at `hpu/src/moe_mme.cpp:39-46`: the per-expert tile-count prefix sum is
   computed as a GEMM against a `512×512` upper-triangular constant.

### 2.4 How fast is it, really — and what is it bound by

**THEY SAY (measured):** 155.7 tok/s at 2k, DFlash2 block 4, ~2.9 tokens accepted per verify round
(README). **READ:** a 4-row verify window costs 18.63 ms at 2k (`README` and `hpu/tpc/*` agree the
window is 4 rows; `dflash.hpp:29` `block=4`). 2.9 / 0.01863 s = 155.7 tok/s — the numbers are
self-consistent, so I trust the round time.

**INFERRED arithmetic on those numbers and the code's own byte counts:**

- A round reads, at minimum, one copy of the union of the 4 rows' routed experts. If all 4 rows hit the
  same 11 experts: `11 × 3.07 MB × 48 layers = 1.62 GB` in 18.63 ms ⇒ **87 GB/s**.
- If the 4 rows route to 40 distinct experts: `40 × 3.07 MB × 48 = 5.90 GB` ⇒ **317 GB/s**.
- The memory-bound ceiling is `2450 / 1.62 GB` ⇒ **~1510 tok/s**. So the port runs at **~10 % of the
  HBM roofline** in the good case.

This is corroborated by their own instrument: **THEY SAY (measured)** the five-tile expert GEMV is
**237 GB/s at the decode shape = 10 % of HBM peak** (`hpu/tpc/iq4q8_gemv_tile.c:6`), and the whole
point of the `tile`/`pf` kernels is to raise it. So: **this port's decode is latency- and
issue-bound, not bandwidth-bound**, and every one of its decode optimisations is about
instructions-per-weight-byte, not about bytes. That is a very different optimisation problem from the
one our §3.2.1 curve addresses, and it is the correct frame for everything in §4 and §5.

---

## 3. Claim 1 — "TPC beats the MME at one row" — **VERDICT: unsupported, and probably a small-tile effect misdescribed as an M=1 hardware property**

### 3.1 The claim, verbatim

**THEY SAY**, `hpu/tpc/gemv_bf16.c:1-5`:

> `STRATA_HPU_GEMV_TPC`: a one-row BF16 GEMM `out[n] = sum_c x[c] * w[n][c]` (F32 accumulation) on the
> TPC. **At one row the MME streams the weights at 1.3-1.6 TB/s**; a TPC member with eight independent
> weight loads in flight per 128 columns and x held in local memory **reads near the 2.2 TB/s the read
> probe (tests/bw_read.cpp) measures**.

and again at `hpu/include/strata/hpu/performance_modes.hpp:335-337`: *"which streams the weights faster
than the MME at one row"*.

### 3.2 The kernel — read in full (`hpu/tpc/gemv_bf16.c`, 33 lines)

**READ.** Structure:

- `__local__ bfloat128 xl[64]` (line 6) — the activation row, **cached in TPC local memory**, up to
  64 × 128 = **8192 columns**. This is where the `K ≤ 8192` bound comes from, and it is a *kernel*
  limit, not a hardware one.
- One index-space member owns **8 output rows** (glue: `ceil(rows/8)`, `hpu/tpc/glue.cpp:593`).
- Per 128-column chunk: **8 independent `v_bf16_mac_acc32_b`** against 8 rows, all sharing one `xv`
  register (lines 18-19). So **8 weight loads of 256 B and 8 MACs per iteration** — one instruction per
  load, i.e. the loop is at the load-port ceiling by construction (see §3.6).
- Rows past the end repeat the last row and are never stored (lines 13-14, 26) — the standard
  no-tail-branch trick.
- Output via `v_f32_st_tnsr_partial` (F32) or a bf16 convert (line 28-31), because the MME's bf16
  output is round-to-nearest-even and they match it.

**The gate**, **READ** `hpu/src/recipe.cpp:309-318`:

```cpp
if (gemv_tpc_enabled() && transpose_b && a.desc.shape[0] == 1 && a.desc.dtype == DType::BF16 &&
    b.desc.dtype == DType::BF16 && (out.desc.dtype == DType::BF16 || out.desc.dtype == DType::F32) &&
    k % 128 == 0 && k <= 8192 && n <= 65536) {
    ... op("strata_gemv_bf16", ...); return;      // else: fall through to synNodeCreate(..., "gemm", ...)
}
```

Note what this is: **a blanket substitution inside `GraphBuilder::gemm`**. There is no shape-dependent
choice, no measured threshold, no per-call-site opt-in. Any qualifying one-row bf16 GEMM becomes a TPC
node. The MME path it "replaced" is the `else` branch (line 319-325) and is still there.

**The glue check**, **READ** `hpu/tpc/glue.cpp:584-593`: `columns % 128 == 0`, `columns <= 8192`,
`rows > 0` (no upper bound in the glue), index space `ceil(rows/8)`.

### 3.3 Where the three bounds actually come from — none of them is an MME bound

| bound | real reason | basis |
|---|---|---|
| `K % 128 == 0` | the 128-wide `bfloat128` vector the kernel loads x and w in | **READ** `gemv_bf16.c:9,16` |
| `K <= 8192` | `__local__ bfloat128 xl[64]` — 64 × 128 columns of TPC local memory for x | **READ** `gemv_bf16.c:6`, glue `:589` |
| `N <= 65536` | a member/index-space cap — the same `65536` appears as the bandwidth probe's `members` limit (`glue.cpp:699`), so it looks like the TPC index-space dimension maximum. Whatever it is, **in practice it is what excludes the LM head**: `n_vocab = 248320` | **READ** `geometry.hpp:27`, `recipe.cpp:313`, `glue.cpp:699`; that it is the index-space limit is **INFERRED** |

⇒ **The triple `(K%128==0, K≤8192, N≤65536)` is the TPC kernel's own operating envelope.** The header
presents it in the same breath as the MME claim, and it is easy to read as "the region where the MME
loses". It is not. Two of the three bounds are register/local-memory budgets in the TPC kernel, and the
third is the LM-head exclusion.

### 3.4 **There is no measurement of the MME at one row anywhere in the repository**

I searched the whole tree for the claim's provenance:

- `rg -n "TB/s|GB/s"` across `hpu/` returns **five** hits: the comment itself, `iq4q8_gemv_tile.c:6`
  (237 GB/s), `bw_read_probe.c:2,10`, `bw_read.cpp:2-3,121`, `gemv_throughput.cpp:163`. **Every one of
  them is a TPC read or an IQ4 GEMV. Not one is an MME GEMM.**
- `hpu/tests/`: the hardware probes are `bw_read.cpp` (TPC read bandwidth), `gemv_throughput.cpp`
  (resident IQ4 GEMV variants), `gather_probe.cpp`, `gdn_step_probe.cpp`. There is **no**
  MME-vs-TPC comparison program. `gemm.cpp` is a correctness test.
- `bench/micro/` contains exactly one file, `cuda_callback_affinity.cpp` — nothing to do with the HPU
  backend.
- `data/`, `docs/benchmarks/` contain no HPU measurements (`docs/benchmarks/*` are AMD/gfx1100).

⇒ **The 1.3–1.6 TB/s figure has no artifact behind it in this repository.** It is a number in a
comment. (**READ** the absence.)

### 3.5 The author's *other* "MME tiles this badly" claim is about narrow **N**, and it is shipped

This is the useful part of the puzzle, because he makes the same move twice and only one of them
states a mechanism.

**READ** `hpu/tpc/hc_down_bf16.c:1-2`:

> `STRATA_HPU_HC_DECODE_TPC`: a decoded token's HC read down and inject projections on the TPC instead
> of the MME (**which tiles a `[1, HC*H] x [HC*H, rank+HC]` product poorly**)

**READ** `hpu/tpc/hc_down_rows_bf16.c:1-3`: *"a `[rows, HC*H] x [HC*H, rank+HC]` product with **few rows
and a narrow output** that the MME tiles poorly"*.

Now put the geometry in: `HC*H = 4 × 2560 = 10240`, `rank+HC = 320 + 4 = 324`. So the offending GEMM is
`[1,10240] × [10240,324]` — **6.6 MB of weights, N = 324.** That is a *narrow-N* tile, and 6.6 MB is
below our §3.2.1 16 MB threshold. The mechanism named is *"a narrow output"*. **And this switch is ON
in production** (`HC_DECODE_TPC: 1`).

⇒ So the author's working, shipped example of an MME-tiling problem is a **small-tile / narrow-N**
problem, at a tile size our own §3.2.1 already predicts the MME to be bad at. The GEMV_TPC comment
takes the same intuition and attaches it to **M = 1**, which is a different variable and one for which
neither he nor we have a measurement at *his* shapes.

### 3.6 Cross-check against our own §3.2.1 measurement — and the reconciliation

Our manual, `03-porting.md` §3.2.1, **MEASURED** on our box, `[1,4096] @ [4096,N]` bf16:

| N | weight bytes | achieved | % of 2.45 TB/s |
|---|---|---|---|
| 512 | 4.20 MB | **1.14 TB/s** | 47 % |
| 2048 | 16.79 MB | **1.94 TB/s** | 79 % |
| 8192 | 67.13 MB | **2.19 TB/s** | 89 % |

Our §3.2.3 conclusion is *"a bandwidth figure without its size is not a figure"* and our §3.2.1
guidance is *"keep a decode weight tile at or above ~16 MB"*.

**So: is their claim consistent with ours?** At their tile sizes, mostly **yes** — and that is the
whole point. Two shape regimes:

**The tile sizes, computed from `geometry.hpp` and the graph builders** (**INFERRED** arithmetic;
`hpu/src/gdn.cpp:118-124` for the GDN merge, `hpu/src/qsa.cpp:159-171` for the QSA merge):

| GEMM (one token row) | shape | weight bytes | vs our 16 MB line |
|---|---|---|---|
| GDN `ssm_alpha` / `ssm_beta` | `[1,2560]×[2560,48]` | 0.25 MB each | **far below** |
| QSA `index_key` | `[1,2560]×[2560,128]` | 0.65 MB | **far below** |
| QSA `k` / `v` / `index_query` | `[1,2560]×[2560,512]` | 2.6 MB each | **below** |
| HC read down / up | `[1,10240]×[10240,324]` | 6.6 MB | **below** (and K>8192, so GEMV_TPC never sees it) |
| GDN `attn_qkv` | `[1,2560]×[2560,10240]` | 52.4 MB | above |
| GDN `attn_gate` | `[1,2560]×[2560,6144]` | 31.5 MB | above |
| `ssm_out`, `attn_output` | `[1,6144]×[6144,2560]` | 31.5 MB each | above |
| **merged** GDN `qkvzab` | `[1,2560]×[2560,16480]` | **84.4 MB** | above |
| **merged** QSA `qkvi` | `[1,2560]×[2560,13952]` | **71.4 MB** | above |

⇒ **The qualifying shapes straddle our 16 MB line, and the small ones are exactly the ones the author
moves off the MME by hand** (HC down/up has its own TPC kernel — §3.5). So the claim's 1.3–1.6 TB/s is
what our curve predicts for the *small* half of the set, and our curve predicts 1.94–2.19 TB/s for the
*big* half — which is at or above the TPC's own ~2.15 TB/s load-port ceiling (§4.2, §10 item 6), i.e.
a wash at best.

**A speculative history, offered because it is testable (INFERRED):** GEMV_TPC was developed when the
projections were separate GEMMs (the narrow ones above, 0.25–6.6 MB) where the MME really is slow;
`MERGED_GEMM` then merged them into 71–84 MB GEMMs where the MME is fine, and GEMV_TPC stopped paying.
Note the two switches' comments read like different eras: `performance_modes.hpp:252-258` describes
MERGED_GEMM as making the MME *"tile the longer GEMM differently"*. **The size story is not the only
candidate explanation, though — see §3.7.**

### 3.7 The hot decode path is 4 rows, so the gate mostly cannot fire

**READ** `hpu/src/recipe.cpp:311` — the condition includes `a.desc.shape[0] == 1`.

There are two decode graphs, and only one of them is M=1:

- **The single-token chain**, `hpu/src/layer.cpp:94-113`: `residual {4, 2560}` — that is 4
  *hyper-connection streams*, not 4 rows — and `attn_row {1, h}`, i.e. **one token row**. Its GEMMs are
  M=1, so `GEMV_TPC` *would* fire here.
- **The DFlash verify graph**, keyed on the row count: `hpu/src/layer.cpp:2054`
  `f.verify[{step.bucket, window_key(rows)}]`, built by `build_verify_qsa(..., rows)`; its GDN
  projection is `define_gdn_window_rows(..., rows, ...)` → `graph.gemm(x, weight, parts[i], true)` with
  `x = [rows, h]` (`hpu/src/gdn.cpp:187-200, 289`). **READ** `hpu/include/strata/hpu/dflash.hpp:29`:
  `block=4`, and `STRATA_HPU_DFLASH_BLOCK=4` in production.

⇒ **With DFlash verify on — which it is — the throughput path is a 4-row graph and `shape[0] == 1` is
false there.** `GEMV_TPC` can only fire on the single-token fallback/draft path, never on the path that
produces the published tok/s. That is still a strong statement, but it is **not** "unreachable": the
M=1 graph exists and would use it. (I got this wrong on a first pass — I assumed the decode chain was
the 4-row graph. It is the reverse: the *chain* is M=1, the *verify* graph is M=4, and the verify graph
is the hot one.)

### 3.8 Verdict

**Category: workaround for a small-tile/latency effect, presented as an M=1 hardware property. Not
"unresolved" — we have a mechanism — but also not proven, because nobody has measured the MME at their
shapes.** Specifically:

- **Refuted as stated.** "At one row the MME streams the weights at 1.3–1.6 TB/s" is not a property of
  M = 1. On our box an M=1 GEMV reaches **2.19 TB/s at N=8192** (§3.2.1). The MME reads B once
  regardless of M, so M=1 should cost *compute* efficiency, not bandwidth. Nothing in their repo tests
  this.
- **Plausibly true as a small-tile statement**, and our own curve predicts it. The bounds they quote
  are the TPC kernel's envelope, not the MME's.
- **Their shipped narrow-N example** (`HC_DECODE_TPC`, N=324) is the honest version of the same claim,
  and it is about N, not M.
- **It is off in production**, and it can only fire on the single-token chain — not on the 4-row
  DFlash verify graph that produces the published tok/s (§3.7).

**The decisive experiment** (cheap, and it is *the* thing to run): in one process, one graph, the same
`[1,K]×[K,N]` bf16 GEMM, toggle only the node type (`strata_gemv_bf16` vs `strata_gemm`), sweep
`K ∈ {2560, 4096, 6144}` × `N ∈ {324, 512, 1024, 2560, 6144, 13952, 16480, 65536}` inside a timed graph
(not a standalone launch), and report device-ms/launch and TB/s for both. Gate on the graph dump
showing which node was created — the substitution is inside `GraphBuilder::gemm`, so a silent fallback
is otherwise invisible. If the MME loses only below ~16 MB, our §3.2.1 explanation stands and the
"hardware property" framing dies. If it loses at 70 MB too, we have learned something about the MME
in-graph that our standalone-launch measurement missed — which would itself be worth a manual
correction, because §3.2.1's numbers are standalone launches.

---

## 4. Claim 2 — load pipelining

### 4.1 The four variants, and how they differ

All four are the *same kernel* with the same MAC order (so they are bit-identical by construction, and
`hpu/tests/gemv_throughput.cpp:141-159` `memcmp`s them). **READ** in full:

| kernel | accumulator count | weight loads in flight | members |
|---|---|---|---|
| `iq4q8_gemv.c` / `_pipe.c` / `_packed.c` (5-tile) | **5** `int256` | **5** (1.28 KB) | `slots × parts × groups` |
| `iq4q8_gemv_tile.c` | **1** `int256` | **8** (2 KB) | `slots × parts × tiles` |
| `iq4q8_gemv_packed_pf.c` | **5** | **10** (2×5, next pair prefetched) | as 5-tile |
| `iq4q8_gemv_group.c` | **5** | 5 | reordered by expert |

**The "one accumulator and eight weight loads in flight" is `iq4q8_gemv_tile.c`, and it is a
*different direction* from our §7.6 lever — it lowers the accumulator count to buy load slots.**
**READ** `hpu/tpc/iq4q8_gemv_tile.c:3-7`:

> Why: the packed GEMV is **latency bound, not issue bound**. The five-tile member holds five `int256`
> accumulators, the table, both activation halves and five in-flight weight vectors at once, **which is
> the whole TPC register file**, so the next column pair's loads cannot be issued until the previous
> pair's MACs have consumed their vectors (**measured: 237 GB/s at the decode shape = 10 % of HBM
> peak**). One tile per member leaves one accumulator, so the eight loads of a column-pair batch are
> issued together and **2 KB per member is in flight instead of 1.28 KB**.

### 4.2 Is that the same mechanism as our §7.6 lever?

**Yes — same mechanism, opposite knob, and one step short.** Our §7.6 lever #1 is *"loads in flight
8 → 16 per loop iteration, with independent accumulators. The largest single gain. The original
`acc = acc + x` chain was latency-bound; independent accumulators keep many loads outstanding. 32
accumulators did not help — 16 is the knee."* Theirs is the same diagnosis: a TPC issues in order, so
one accumulator chain keeps one load outstanding (**READ** `hpu/tpc/bw_read_probe.c:2-5`), and the fix
is more independent accumulators/loads in flight.

The difference is **what the register file can afford**:

- **Our kernel** is a pure read/accumulate loop: an accumulator costs one register and nothing else.
  Going 8 → 16 is free, and 16 is the knee.
- **Their kernel** has 5 accumulators *plus* a 256-byte LUT *plus* two activation halves *plus* the
  in-flight weight vectors. Their measurement is that at 5 accumulators the register file is **full**
  and the compiler cannot software-pipeline. So they cut accumulators from 5 to 1 to make room for 8
  loads.

**So it is not "better" or "different" — it is the same lever pushed against a register-file wall that
our kernel does not have.** And note the consequence: their *best* configuration is **8 loads in
flight**, and their probe never tests beyond 8 (§4.3). Ours says the knee is 16. **If their int8
weight loads behave like our bf16/f32 loads, every one of their GEMV variants is below the MLP knee,
and the tile kernel bought 8/5 = 1.6× of a possible 16/5 = 3.2×.**

That is a hypothesis, not a finding: int8 `v_u8_ld_tnsr_b` may saturate earlier than a 64-lane f32
load, and their register file is more crowded than ours. But it is *the* cheap experiment (§9 item 1).

### 4.3 The instrument — and its blind spot

**READ** `hpu/tpc/bw_read_probe.c` in full and `hpu/tests/bw_read.cpp` in full. This is a genuinely
good instrument and it is nearly ours:

- Reads exactly `members × vectors × 256` bytes per launch, so the GB/s is a hardware rate, not a
  "logical weight-byte" estimate (`bw_read.cpp:1-3`).
- `chains ∈ {1,2,4,8}` independent register-resident accumulators, each fed by its own load, **all
  loads of a group issued before any of their math** (`bw_read_probe.c:4-6`).
- `ops ∈ {0,7}`: `0` = one add per load (the memory ceiling), `7` = ~8 vector instructions per 256-byte
  load, *"the ratio tpc/iq4q8_gemv_packed.c runs at"* — which separates **in-flight requests** from
  **instruction issue**, "the two things the existing 'logical GB/s' numbers cannot separate".
- It also attributes the fixed ~14 µs per launch to un-warmed work vs per-graph vs per-node cost
  (`bw_read.cpp:120-171`), which is a question our manual handles separately.

**The blind spot, and it is hard-coded:** `chains` is restricted to `{1,2,4,8}` in *both* the kernel
(`bw_read_probe.c:31-73`, an if/else-if chain ending at 8) **and the glue**
(`hpu/tpc/glue.cpp:697-706`: `(a.chains!=1 && a.chains!=2 && a.chains!=4 && a.chains!=8)` is a reject
condition). **16 chains is not merely untested — it is refused.** So their "8 is enough" is a statement
about the top of their sweep, not a measured knee.

⇒ **Their probe is the cheapest possible replication of our §7.6 experiment, and it already disagrees
with us at the top of its range.** Extending it to 16 and 32 is a two-hour job and settles whether our
"16 is the knee" transfers from our pure-read kernel to their int8/bf16-load kernel at `ops=7`.

### 4.4 `GEMV_PIPE` vs `GEMV_PF` — and a stale comment worth knowing about

**READ** `iq4q8_gemv_pipe.c` (and its twin `iq4q8_gemv.c`): the *pipe* variant issues **all seven loads
of one column pair** (2 activation vectors + 5 weight vectors) before decoding any of them, instead of
load-then-consume. So it is intra-pair load batching, not cross-pair prefetch, and the accumulator count
is unchanged at 5.

**READ** `iq4q8_gemv_packed_pf.c`: the *pf* variant adds a **shadow set of five `uchar256`** so the
*next* pair's five weight loads are issued before this pair's MACs consume theirs — i.e. 10 loads in
flight. Its header says it is "the lever at the decode shape, where eleven slots leave one member per
TPC and the rate is latency-bound, not bandwidth-bound" (`:1-5`).

**A code-vs-comment discrepancy worth flagging:** `iq4q8_gemv_packed.c:28-31` still carries the A/B
toggle for this —

```c
// A/B TOGGLE: STRATA_HPU_GEMV_PF=0 keeps the plain (load-then-consume) loop, anything else uses the software
// pipelined one below. Same MAC order in both, so the outputs must be bit-identical.
const int use_pf=0;   // TPC has no false/true keywords: 0/1 only
```

— but `use_pf` is a hardcoded `0`. The comment describes an environment variable that no longer
controls this file; the pipelined branch in `_packed.c` is dead code, and the live A/B is the separate
`_packed_pf.c` kernel selected in `hpu/src/resident_experts.cpp:191-192`. **Don't be misled by the
comment when reading this file.** (**READ**)

### 4.5 `CACHED_GDN_STEP` and `GDN_STEP_PIPE` — both shipped, and the clearest example of the technique

**READ** `hpu/tpc/gdn_step_cached.c` vs `hpu/tpc/gdn_step_pipe.c`. These are the *only* two of the
brief's pipelining flags that are on, and they show the idiom cleanly:

- `gdn_step_cached.c` caches the layer's q/k vectors in `__local__ float64 qcache[4],kcache[4]` and
  broadcasts each element with `v_f32_mov_dual_group_all_b` + `v_f32_shuffle_b` instead of a scalar
  global load per element — **removing scalar loads, not adding parallelism**.
- `gdn_step_pipe.c` then adds the MLP: `LOAD16(BASE)` (`:18-21`) issues **sixteen** state loads into
  sixteen named registers *before* any of them is consumed, and the comment at `:16` is the useful bit:
  *"One coordinate serves the sixteen loads (index registers are few); each load reads it when
  issued."* So they reuse one `int5` coordinate and mutate `lc[2]` between issues — an index-register
  economy trick. `:40` then consumes all sixteen with `C(0)…C(15)`.

**This is the one place in the codebase where the number is 16**, i.e. where they reached our knee.
**INFERRED:** the reason it works *here* and not in the GEMV is that a GDN step has no 256-byte LUT and
no five-way tile split, so sixteen loads fit in the register file. That is consistent with the
register-file-wall explanation in §4.2 and is a useful sanity check on it.

Also **READ** `hpu/tpc/gdn_step_rows4.c` — the shipped 4-row verify-window kernel, and a genuinely
clever piece of work: it fuses the four recurrent steps of a verify window into **one kernel** by
expanding the delta-rule recurrence algebraically so that the window's first state is read twice total
instead of twice per row (`:1-11`): pass 1 accumulates `S_0·k_t` for all four rows at once; the
per-row deltas follow from those plus the `k_u·k_t` Gram values (computed once, `:47-50`); pass 2 walks
`S_0 → S_1 → S_2 → S_3` in registers. That is a **4× reduction in state traffic for the verify window**,
and it is the kind of algebraic fusion I have not seen in our port. Worth a manual section of its own
if we ever fuse a multi-row recurrent step.

### 4.6 The occupancy model — the constraint that explains every "off" switch in §4 and §5

This is the single most useful thing I took out of the codebase, and it is stated twice by the author
in passing and never collected anywhere.

**READ** `hpu/tpc/route_topk.c:12-17`:

> Up to 24 rows (a decoded token, a DFlash verify window) take the one-row loop below, **a row per
> member on its own TPC**: grouping four rows into one member would run them on a quarter of the cores
> (glue: rows <= 24).

**READ** `hpu/src/resident_experts.cpp:194`:

> gate/up: each row's slots share its x row; **the columns split in two parts so `2*slots` members fill
> the TPCs.**

**READ** the shipped numbers, which confirm the model exactly: `STRATA_HPU_GEMV_PARTS_GU=2` (default,
`resident_experts.cpp:199`) and `GEMV_PARTS_GU_WINDOW=1` (production config). A single-token decode has
`slots = per_row × rows = 11 × 1 = 11`, so `11 × 2 = 22` members against **24 TPCs**. A 4-row verify
window has `slots = 44`, so the window path drops to **1** part — `44` members already fills the array.

⇒ **On this hardware the index-space size is the occupancy, and at a decode shape you are short of it.**
Consequences, all of which the code obeys:

1. **You cannot buy throughput with per-member ILP at decode** — you already have fewer members than
   TPCs, so more work per member makes the imbalance worse, not better. This is why
   `route_topk` deliberately *refuses* to fold 4 rows into one member even though it would be more
   efficient per member.
2. **The cheapest way to raise occupancy is to split K across members** (`GEMV_PARTS_*`), and it is
   *not* free: it materialises `gp` copies of the output and adds a `reduce_sum` node
   (`resident_experts.cpp:210-216`), i.e. `2 × slots × 2ff` floats plus a reduction per layer per
   token. So the split is worth exactly as much as filling the array is worth.
3. **Anything that shares work between members costs occupancy** — which is why the row-sharing kernel
   lost (§5.3) and why `MOE_PAIR` (which shares *inside* a member) is the better shape for the same
   idea.
4. **At prefill the constraint vanishes** (hundreds of rows ⇒ hundreds of members), which is why the
   prefill MoE is a completely different engine (§2.3 item 6) and why `GEMV_PARTS_GU_WINDOW=1` is right.

**Why this matters for us:** our own decode kernels are usually discussed in terms of bytes and MLP.
The question *"how many index-space members does this kernel launch, and is that ≥ 24?"* is a different
and cheaper diagnostic, and if the answer is no then no amount of pipelining will help. **It is a
one-line check on any kernel we already have**, and it is the first thing I would look at on our own
decode kernels. (**READ** for the two comments and the shipped constants; **INFERRED** for the
generalisation to our kernels.)

---

## 5. Claim 3 — grouped expert reads

### 5.1 How it works

**READ** `hpu/tpc/route_group_starts.c`, `route_group_order.c`, `route_group_blocks.c`,
`iq4q8_gemv_group.c`, `iq4q8_gemv_pair.c`, and the driver
`hpu/src/resident_experts.cpp:158-195`.

The three-node pre-pass, per window:

1. `route_group_starts` (`experts+1` members): `starts[e] = #{slot : ids[slot] < e}` — a 64-slot-vector
   prefix count, so the runs are `[starts[e], starts[e+1])`. The last member writes `slots`.
2. `route_group_order` (`experts` members): for each expert `e`, scan all slots and write every
   matching slot number at `start[e]+k`, in slot order. Result: **a permutation of `0..slots-1` whose
   runs are one routed id each.** Slot numbers are carried as **F32** because F32 is exact below 2²⁴ and
   `s_f32_st_g` is the scalar store they have.
3. `route_group_blocks` (with `MOE_GROUP_BLOCKS`): same, but skips 64-slot blocks whose per-expert count
   is zero (`u4_route_blocks` supplies the counts) — *"instead of scanning every slot with scalar loads
   for every expert"*.

Then `iq4q8_gemv_group.c` reads `order[position]` for its slot, so **consecutive members read the same
expert's weight rows back to back** and those rows are L2-hot. Its header is precise about what is and
is not shared: *"one read of an expert's weights per window instead of one per slot that routes to it.
The store index is the slot, so the output tensor keeps the routing order and every slot's arithmetic is
the same as the packed kernel's."* (**READ** `:1-5`.)

### 5.2 The reuse is *L2 locality*, not register-level sharing — except in `MOE_PAIR`

This distinction matters and the two flags are easy to conflate:

- **`MOE_GROUP`** → `iq4q8_gemv_group.c`: **no weight load is shared.** Each member still loads its own
  five vectors per column pair. What changes is *which address* it loads — a neighbour's, so the second
  read hits L2. **The saving is HBM traffic only, and only if the run is long enough and the weights
  survive in L2.** The on-die pool is 48 MiB (our ch. 7); one expert's gate/up slice is 2.0 MB, so a run of ≥2 slots can
  plausibly stay hot, but the 48 layers stream 1.6 GB through it per token, so this is a *within-window*
  reuse only. (**READ** the code; **INFERRED** the L2 reasoning.)
- **`MOE_PAIR`** → `iq4q8_gemv_pair.c`: **this is the real register-level reuse.** Two adjacent slots in
  the grouped order with the *same* routed expert share each weight load: `STEP2(A,B)` loads one
  `uchar256 v`, does the two LUT shuffles **once**, and issues **four** `v_i8_mac_b` — two for slot A,
  two for slot B (`:90-98`). So the decode cost (2 shuffles per weight vector) is **amortised over two
  rows**, which is a *compute* saving, not just a bandwidth one. Three output tiles per pass, two passes
  (tiles 0-2, 3-4), because 3 tiles × 2 slots = 6 accumulators is what fits.

⇒ **`MOE_PAIR` is the more interesting kernel, because in an IQ4 codebook GEMV the shuffle is half the
instruction count.** Halving the shuffles per MAC is a ~1.5× instruction-count win on the inner loop,
which is the right lever for a kernel that runs at 10 % of HBM peak (§2.4).

### 5.3 Their measured answer on the row-sharing variant is negative — and that is the useful part

**READ** `hpu/src/resident_experts.cpp:158-162`:

```cpp
// STRATA_HPU_DFLASH_GEMV_ROWS=1: windows use the shared-read kernel (identical outputs; measured slower so far).
static const bool rows_kernel=[] {const char* e=std::getenv("STRATA_HPU_DFLASH_GEMV_ROWS");return e && *e=='1';}();
const bool share_reads=r>1 && slots<=128 && rows_kernel && !q8_packed_enabled();
```

and `iq4q8_gemv_rows.c` / `route_share.c` are that kernel (`share[slot][0] = 2` pairs the slot with
`share[slot][1]`, `1` = alone, `0` = skip). **THEY SAY (measured): it is slower.** Note the extra gate:
`!q8_packed_enabled()` — so it is *incompatible with the shipped packed-Q8 layout* anyway.

**This is a genuinely valuable negative result for us**, and it is the one place where "share a weight
load across rows" was tried and lost. The likely reason (**INFERRED**): pairing halves the number of
independent members, and §4.6's occupancy model says members are what fill 24 TPCs — so at the
decode shape you are trading occupancy for reuse on a machine that does not have enough of either.
`iq4q8_gemv_pair.c` is the better compromise (it keeps one member per pair *and* amortises the shuffle),
and it is also off.

⇒ **The transferable lesson: at a decode shape with ~11 slots on 24 TPCs, sharing work across slots
costs occupancy you cannot afford. Reuse across slots is a *prefill* lever (`MOE_MME`), not a decode
lever.** Note how this sits next to our §7.6 lever: both are about doing more work per byte, but §7.6
buys it with *independent accumulators inside one member* (which costs registers) and this buys it by
*sharing a decoded weight between members* (which costs occupancy). On a decode shape you are short of
occupancy first, so the second loses. That ordering is worth writing into the manual.

---

## 6. Claim 4 — per-layer recipe instances

### 6.1 What the flag does

**READ** `hpu/include/strata/hpu/performance_modes.hpp:127-131` (**THEY SAY**):

> every layer decodes through its own instance of the shared decode recipes, so a launch sees the same
> weight addresses every token (**Synapse patches what changed**) instead of the previous layer's.

**READ** the three use sites, `hpu/src/layer.cpp`:

- `:1411` — GDN layers: `auto& decode = decode_recipe_per_layer() ? f.chain_layers[p.index] : f.chain;`
  A per-layer `HostRecipe` (a `std::vector` indexed by layer) instead of one recipe shared by all 36
  GDN layers.
- `:1446` — QSA device layers: the **recipe-cache key** gains the layer index:
  `f.device_layer_for(step.bucket | (decode_recipe_per_layer() ? uint32_t(p.index)<<24 : 0u), first)`.
  Bits 24+ of the key are the layer.
- `:1472` — QSA decode-select: `f.decode_select_for(step.bucket, ..., decode_recipe_per_layer() ? p.index : 0)`.

So the mechanism is exactly as claimed: **the recipe identity is what varies.** A recipe binds its
weight tensors by address; 36 GDN layers have identical *shapes* but different weight *addresses*. Share
one recipe and every launch must re-patch 30-odd tensor addresses; give each layer its own and the
addresses are compile-time constants in the recipe, so a launch is pure enqueue.

### 6.2 What it costs, and why it is off

- **Graph count × 48.** 36 GDN + 12 QSA recipes instead of 1 + 1, each with its own persistent slots.
  The `QSA_SELECT_KEEP_MIB` machinery (`performance_modes.hpp:165-171`) exists precisely because recipe
  HBM is a real budget here — *"each recipe's persistent slots"* — and that flag is set to 1024 MiB in
  production, i.e. recipe HBM is being actively capped. So per-layer recipes would multiply a budget
  they are already capping. (**READ** the flag; **INFERRED** the interaction.)
- **Compile time.** The README's own known-limitations section says first-touch compilation of a new
  window size costs >10 s; ×48 is a warm-up problem.
- **It is OFF in production.**

### 6.3 What is actually worth taking from this for our graph-count work

**The idea is directly transferable and cheap to test on our side, and it is a *different* axis from
the ones our manual's §1.4.3 counts.** Our §1.4.3 counts `layer × constant × shape` specialisations
where the *shape or a specialised integer* varies. Here the axis is **the weight address**, which
normally does *not* count as a specialisation at all — a recipe is supposed to be reusable across
addresses via rebinding. This code is saying: **rebinding is not free, and on this stack it is a
per-launch cost that you can trade against recipe HBM.**

The measurable quantity is *"what does one external rebind cost per launch"*. That is directly
measurable on our box: take one recipe, launch it N times rebinding one tensor each time vs N times
with no rebind, and diff the device union. If the cost is non-trivial relative to a 12.8 ms/18.6 ms
step, per-layer (or per-address) recipe instances become a lever for us too — and unlike theirs, ours
would be a *graph count* question, which we already have machinery to reason about.

Note also the framing consequence: **if rebinding is expensive, "graph count" is not the right cost
model.** The right model is `graph count × (compile + HBM) + launches × rebind`. Our §1.4.3 does not
carry the second term. Flagging that as a possible manual gap, not a correction — we have not measured
rebind cost.

---

## 7. Claim 5 — the glue, the shape constraints, and the access patterns

### 7.1 `glue.cpp` in outline

**READ** `hpu/tpc/glue.cpp`, 1476 lines, in full. It is a standard SynapseAI TPC glue library:

- **A 112-entry append-only kernel table** (`:132-281`), `static_assert(std::size(kernels)==112)` at
  `:283`. Append-only because `configure()` and `base_kernel()` index it, and the comment records the
  bug that taught them: *"inserting it in the middle would silently renumber the HC window kernels
  (which is exactly what broke hpu_hc_ple_compile once)."*
- `base_kernel(id)` / `fast_twin(id)` (`:285-286`): fast-math twins and "same checks, different ELF"
  variants share the original's shape checks and index space. A nice way to keep a family consistent.
- **Three entry points:** `GetKernelGuids`, `GetShapeInference` (`:1283`), `InstantiateTpcKernel`
  (`:1340`). `InstantiateTpcKernel` prints `glue reject guid=%s status=%d rows=%u` to stderr on
  rejection (`:1350-1352`) — because *"without this the SDK only reports 'Generic failure (status 26)'"*.
  **That is a debugging hint worth stealing**: status 26 from `synGraphCompile` is a glue shape
  rejection, not a compiler failure.
- **Shape checking is by exact equality on every dimension** — `shape()` (`:315-327`) requires
  `minSizes[axis] == maxSizes[axis] == expected` for every axis, and a specific `TensorDataType`. There
  are no dynamic shapes anywhere; every node's geometry is pinned at compile time. This is the same
  "there are no dynamic shapes" fact as our §1.4.1, enforced in the kernel library rather than the
  graph compiler.
- `width(s)` (`:293`): `s > 0 && s <= 256 && s % 64 == 0` — the recurring TPC width rule.

### 7.2 What the enforced constraints reveal about TPC limits

**READ** across `configure()`. The recurring constants are the vector width in each dtype and the
index-space fan-out:

| constraint | appears as | what it is |
|---|---|---|
| `% 64` | `width()`, `GdnConv`, `PleConvRows`, `QsaGather` `width%64` | 64 F32 lanes = one 2048-bit register |
| `% 128` | `gemv_bf16` `columns%128`, `Fp8Gemv` widths, `GdnNormRows` `width==128` | 128 bf16 = 256 B = one load |
| `% 256` | `HcDownRows` `columns%256` | 256 int8 lanes, or 2×128 bf16, per iteration |
| `% 1280` | `Iq4Expand` `rows%1280` | the 5-tile × 256-row expert layout |
| `columns <= 8192` | `gemv_bf16` | `xl[64]` local memory (§3.3) |
| `cells <= 262144` | every QSA gather kernel | the 256k context, hard-capped |
| `heads <= 32` | QSA gather (FP8 cache) | 32 lanes of a bf16/f32 register |
| `rows <= 8192` | `kMaxWindowRows` (`model_abi.hpp:9`) | prefill window cap |
| `slots <= 90112` | `kMaxMoeSlots` (`model_abi.hpp:10`) = 8192×11 | window rows × per_row |
| `indexSpaceRank <= 3` | QSA window kernels set rank 3 | the TPC index space is 5-D but they use ≤3 |
| `top <= 4096`, `block <= 64` | QSA select | selection budget |

**Two of these are worth quoting in the manual as hard facts:** the **2048-bit register** (64 F32 / 128
bf16 / 256 int8 lanes) as the universal width unit, and the fact that **every kernel's geometry is
pinned at compile time and checked for exact equality**, so a shape change is a recompile — the same
conclusion as our §1.4.1, arriving from the kernel library instead of the graph compiler.

One stale message: `hpu/src/resident_experts.cpp:141` throws *"resident MoE needs rows <= the resident
window, slots <=4096"* but `kResidentMoeSlots = abi::kMaxMoeSlots = 90112`. The message is wrong; the
constant is right. Harmless, but do not quote the message.

### 7.3 `TPC_ACCESS` — what it actually changes (and it is **not** what the comment says)

**THEY SAY** (`performance_modes.hpp:417`): *"Affine access patterns for row norm and HC kernels in the
custom TPC glue."* Vague.

**READ** `hpu/tpc/glue.cpp:1363-1415`. The default for every tensor is:

```cpp
out->inputTensorAccessPattern[i].allRequired = 1;   // and noRmwAccess = 1 on outputs
```

With `STRATA_HPU_TPC_ACCESS=1`, specific kernels instead publish an **affine index-space→tensor-dimension
map**: `m.indexSpaceDim = d; m.a = stride; m.start_b = 0; m.end_b = count-1`. That is a
`TensorAccessPattern` ROI mapping — it tells the compiler's **slicer** that a node's tensor access is
affine in the index space, so the node can be *split into slices and pipelined* with its neighbours.

So `TPC_ACCESS` is **not a kernel-side access optimisation at all.** It changes no instruction the
kernel executes. It is a *scheduling* hint that lets Synapse slice the node. It applies to exactly
**8 kernel ids**: `row_norm` (4, 13, 39), `hc_mix` (5, 10, 40), `hc_write` (6, 11). And the comment at
`:1366` states the limit explicitly: *"Routed weight addressing stays opaque"* — i.e. the MoE weight
tensors **cannot** be mapped, because their addressing is data-dependent, so the GEMV/MoE nodes can
never be sliced this way.

`STRATA_HPU_MOE_SLICE` (`:1432-1466`) is the same mechanism for the MoE tile kernels (ids 70, 75, 77,
81, 82, 83), mapping their tile/batch axis so *"the slicer may run the expansion, the gather and the
activation slice by slice with the batch GEMMs through SRAM instead of full HBM temporaries"*. **That
one IS shipped.**

⇒ **The transferable fact: on this stack, whether the compiler can slice-and-pipeline a TPC node
through SRAM is controlled by a glue-side `TensorAccessPattern`, not by the kernel.** Default is
`allRequired = 1`, which forbids slicing. If we ever want a TPC node pipelined against an MME node
through SRAM — which is exactly what our §7.6 and §7.8 item 1 are chasing — this is the lever, and it
is one we have not touched. **That is arguably the single most useful thing in this file.**

### 7.4 `TPC_COLPAIR` and `TPC_HILO` — what the access patterns actually change

These are not access patterns; they are two *different FP8 expert GEMV kernels*, selected in
`hpu/src/model.cpp:175-180` and `:321-322`, and both require `TPC_EXPERTS` + `DIRECT_EXPERTS` +
`--fp8-experts` (i.e. the pre-resident expert path). Both are **OFF** in production and **dead** under
`RESIDENT_IQ4`.

- **`TPC_COLPAIR`** → `hpu/tpc/fp8_gemv_colpair.c`. This is a genuinely different technique and the
  best single idea in the repo after the row-fusion. **READ** `:1-4, 19, 29-47`:
  - The resident codes are re-laid-out `[rows/128][columns/2][128][2]`, with rows interleaved inside each
    128-row block as `0,64,1,65,…,63,127` so the pattern matches the two halves of the BF16 MAC
    accumulator.
  - **One SIMD lane = one output row.** `v_bf16_mac_acc32_b` accumulates per-lane, so **there is no
    horizontal dot-product reduction at all** — the comment says it outright: *"No horizontal
    dot-product reduction is needed."* Each lane's accumulator *is* that row's answer.
  - Four independent accumulators `a0..a3` hide the BF16 MAC latency (`:35`, `:38-39`), and the
    x-vector is broadcast across lanes with `v_bf16_mov_dual_group_all_b` + per-lane `v_bf16_shuffle_b`.
  - Compare the *reduction* layout it replaces, `hpu/tpc/fp8_gemv.c:59-67`: every 4 rows it does
    `v_f32_reduce_add` **and** `v_f32_sel_eq_u32_b` **per row** to write the answer into the right lane.
    That is 2 vector ops per output element on top of the MACs.
  - ⇒ **COLPAIR removes ~2 vector instructions per output element and the whole reduction tree, at the
    cost of a weight layout that has to be built once at load.** For a GEMV where the output is small
    (N=324..2560) and K is large, that is a large win. **The general lesson: transpose the problem so
    the output dim is the lane dim and the reduction disappears.** That is directly applicable to any
    GEMV we write where N ≤ 64·(lanes available).
- **`TPC_HILO`** → `hpu/tpc/fp8_gemv_hilo.c`. Same kernel, but the activation is a **hi/lo FP8 pair**
  (`xl` and `ll`, `:2`, `:15`), with two accumulator sets per row (`a_k` and `d_k`, `:22-25`) and the
  final combination `reduce(a) + reduce(d)/16` (`:62`). Its point is to reproduce *the MME path's*
  dynamically-scaled FP8 activation on the TPC so the two paths agree numerically. **THEY SAY** the
  header's only sentence: *"Same dynamically scaled FP8 hi/lo activation representation as the MME
  path."* Nothing about performance.

---

## 8. Claim 6 — `hpu/PROGRESS.md`

**READ:** it does not exist. Three files cite it (`batch_slots.hpp:29`, `kv_q4_cell.hpp:14`,
`qsa_state_io.cpp:24`), `git ls-files` does not have it, it is not on disk, and it is not gitignored.

The citation in `batch_slots.hpp:31-33` reads: *"The execution is still one slot at a time through this
backend's single Model, so the schedule says so and makes no throughput claim (**declared in
`hpu/PROGRESS.md`** and in the tool's INFO line)."* So the throughput disclaimer the previous pass
quoted is **not** in this repository; the surviving form of it is the INFO line (see §9). The two other
citations are about *named FP-rounding divergences* (`kv_q4_cell.hpp` and `qsa_state_io.cpp`), i.e. the
project keeps a divergence register in a file it did not publish. **That register is the document I
would most like to have** — it is the list of "where this port's arithmetic differs from the reference",
which is exactly the kind of thing our manual's porting chapter should be able to point at.

---

## 9. Claim 7 — BS > 1: **the previous pass was right about the shipped path and wrong to say no path batches rows**

### 9.1 The shipped `--batch` path is time-sliced — confirmed from code

**READ** `hpu/src/protocol.cpp:186-190`: the INFO line literally emits
`" batch_slots=" + slots + " batch_schedule=time-sliced"`, with the comment *"the slots are time-sliced
through one Model, so the figure is a slot COUNT, never a row-batched number"*.
**READ** `hpu/include/strata/hpu/batch_slots.hpp:31-33` (quoted in §8).
**READ** `hpu/tests/batch_protocol.cpp:390` — a test asserts the INFO line contains
`batch_schedule=time-sliced` and fails with *"the INFO line hides that the slots are time-sliced"*.
**READ** `hpu/tools/generate.cpp:144` — the `--help` text says `--batch N / --slots N --serve (2..8
requests in slots; time-sliced through one model)`.

⇒ Confirmed: the shipped batching is one slot at a time. **The disclaimer is real and is in the INFO
line, not in `PROGRESS.md`.**

### 9.2 But a row-batched path *does* exist — as an unwired, host-tested header

**READ** `hpu/include/strata/hpu/multi_slot_window.hpp:1-20` and
`hpu/include/strata/hpu/multi_slot_loop.hpp:1-16`. These describe exactly the thing the previous pass
said does not exist:

> `run_multi_slot_window` decodes **ONE row for each READY slot** … packs up to `max_rows` of them per
> window (ascending slot order), feeds each slot's own last token's embedding …
> `pack_slot_window` → `stage_arena_rows` → `attention + head` → `sample_window_rows`

with `MultiSlotOperations{project, attention, head}` as `[T,hidden]`-shaped callbacks. That is
row-batching into one graph.

**And it is not wired to anything.** `rg -n "multi_slot_loop.hpp|multi_slot_window.hpp"` finds only
`CMakeLists.txt` and the two host tests `hpu/tests/multi_slot_window.cpp` and
`hpu/tests/multi_slot_loop.cpp`. Nothing in `src/` includes either header. So it is a **built, tested,
unwired integration layer** — a prototype.

⇒ **The correct statement is: "the shipped `--batch` path is time-sliced and says so; a row-batched
window/loop exists as a header-only prototype with host tests and is not reachable from the engine."**
That is stronger and more useful than "no path batches rows", because it means the pieces are there and
somebody chose not to ship them.

### 9.3 And multi-row graphs *are* shipped elsewhere

Worth being precise, because it changes the picture: the engine runs multi-row graphs all the time —
just not for `--batch`.

- **DFlash verify**: 2–4 rows in **one launch per layer** (`layer.cpp:1408+`, `dflash.hpp:29`
  `block=4`), including the fused 4-row GDN step of §4.5. So the *decode* path is already a 4-row
  batched graph — it is just batching a *drafter's* rows, not multiple requests.
- **Prefill**: up to 8192 rows per window (`kMaxWindowRows`), with the grouped-MME MoE.
- **`define_resident_moe(rows, ...)`** (`resident_experts.cpp:135-146`) accepts `rows` up to the
  resident window and `slots ≤ 90112`; it is the *same* function for 1 row, 4 rows and 8192 rows. The
  slots/size gating is `(rows>8 && !q8_packed_enabled())` — packed Q8 is required above 8 rows.

⇒ So the machinery to run N request-rows in one graph exists and is exercised by the drafter. What does
not exist is a scheduler that puts N *requests* into it. **The `multi_slot_*` headers are exactly that
scheduler, unwired.** For our own BS>1 work the interesting question is therefore not "can the graph do
N rows" (it can) but "what does the per-bucket step cost do as N grows", and this port has no
measurement of that at all — its only batching numbers are prefill tok/s.

---

## 10. What contradicts, or should be added to, our manual

Cited against the manual's own section numbers.

1. **§3.2.1's framing invites exactly the misreading this port made.** Our curve is indexed by `N` with
   `K=4096` fixed, and the guidance is "keep a decode weight tile at or above ~16 MB". Both are right,
   but a reader can come away thinking the cliff is about `N`. It is about **bytes of weight per
   launch**. **Suggested edit (not a correction):** restate §3.2.1's headline as *"the M=1 GEMV cliff is
   a tile-size cliff, not an N cliff"* and add the port's `[1,10240]×[10240,324]` (6.6 MB, N=324) as a
   second shape that sits below it. That shape is **THEY SAY**-motivated and our own measurement covers
   the size, so it can be cited as consistent evidence without adopting their claim.

2. **§3.2.2's "the cliff is a TPC/GEMV phenomenon" is ambiguous and this port's numbers can sharpen
   it.** Our text means "a property of the GEMV op, not of the DMA engine". A reader can read it as "the
   TPC has the cliff and the MME does not" — and then this port's claim looks like a contradiction of
   us when it is not. **Suggested edit:** say explicitly that the 1.14 TB/s point at N=512 is an *MME*
   GEMV measurement, and that the same size-dependence should be expected of a TPC GEMV at the same
   tile size.

3. **§7.6's "16 is the knee" is neither corroborated nor contradicted — their sweep stops at 8.** Their
   `bw_read_probe` sweeps `chains ∈ {1,2,4,8}` and **refuses 16 in the glue**. So this is not a
   contradiction of our finding, it is a *near-miss replication*, and it is the best independent
   evidence we have that our lever is the right one — and it leaves our knee untested on their path. **Suggested edit:** cite it in §7.6 as independent corroboration of
   the mechanism (single accumulator ⇒ one request in flight ⇒ 365 GB/s) with the caveat that their
   sweep stops at 8 and does not test our knee.

4. **§7.6 should gain the register-file-wall caveat.** Our lever is cheap because a read/accumulate
   kernel's accumulator costs one register. In a *codebook* GEMV (a 256-byte LUT, two activation
   halves, five in-flight weight vectors) five accumulators already fill the register file, and the
   same author who wrote the chains probe cut accumulators from 5 to 1 to buy load slots
   (`iq4q8_gemv_tile.c:3-7`). **The lever's sign can invert when the accumulator is expensive.** That is
   a real limit on §7.6's generality and it is not in our text.

5. **§1.4.3's graph-count model is missing a term.** It counts `layer × constant × shape`. This port
   argues that **weight address** is a specialisation axis too, because rebinding a recipe's tensor
   addresses is a per-launch cost that you can trade against recipe HBM
   (`DECODE_RECIPE_PER_LAYER`, §6). We have not measured rebind cost, so this is a **gap to
   investigate, not an error**. But if rebind is not free, the right cost model is
   `graph count × (compile + HBM) + launches × rebind`, and §1.4.3 should say so or say why not.

6. **A mechanism for the TPC stream envelope, worth adding to §1.6.** `fp8_gemv.c:6` says *"inside the
   four cycles a TPC needs per 2048-bit global load"* — i.e. **256 B / 4 cycles = 64 B per cycle per
   TPC**. Take our §1.6 **MEASURED** envelope of 2,147 GB/s over 24 TPCs, which is **89.5 GB/s per
   core**: 89.5e9 / 64 B = **1.40 GHz**, a plausible TPC clock. So the vendor-side figure in that
   comment *reproduces our measured envelope from a mechanism*, and it does so to within rounding —
   which is much stronger than either number alone. It also gives our §1.6 ceiling a *reason* (the
   load-port issue rate) where we currently state only a measurement. **THEY SAY** for the 4-cycles
   figure; the agreement is the evidence. **Worth checking in the Grimoire** whether the TPC load port
   is documented as 64 B/cycle, because if it is, §1.6 can say *why* the envelope is where it is.

7. **A genuinely new technique our manual has no section for: algebraic row-fusion of a multi-row
   recurrence.** `gdn_step_rows4.c` (§4.5) turns 4 sequential delta-rule steps into 2 passes over the
   state by expanding the recurrence into `S_{t-1}·k_t = A_{t-1}(S_0·k_t) + Σ …`, reading the state
   **twice per window instead of twice per row**. That is a 4× state-traffic reduction for a verify
   window and it is bit-inexact (different summation association) but mathematically identical. If our
   port ever fuses a multi-row recurrent step this is the pattern.

---

## 11. Ranked follow-ups, with the experiment for each

Ranked by expected value to us. Items 1–3 are cheap and each could change a decision.

**1. Extend their read probe to 16 and 32 loads in flight, at `ops=7`. — HIGHEST VALUE.**
Their `bw_read_probe.c` is our §7.6 experiment, on their int8/bf16 load path, with an `ops` knob that
separates MLP from instruction issue — and it stops at 8 because the glue refuses 16. **Experiment:**
port `bw_read_probe.c` semantics (chains ∈ {1,2,4,8,16,32}, ops ∈ {0,7}, 512 MiB, 24 TPCs, 256-byte
vectors) and find the knee for `v_u8_ld_tnsr_b` / `v_bf16_ld_tnsr_b`. **What it decides:** whether our
"16 is the knee" transfers, and therefore whether their GEMV kernels are leaving ~2× on the table
(8/5 → 16/5) or are already at the knee. **Cost:** a day; it is a 100-line kernel plus a harness.

**2. Steal the occupancy-by-K-splitting lever, and test it at our decode shape. — HIGH VALUE, CHEAP.**
They fill 24 TPCs by splitting the K dimension across index-space members (`GEMV_PARTS_GU=2` ⇒
`11 slots × 2 = 22` members), and they drop to 1 part for a 4-row window because 44 slots already fill
them. `route_topk.c:17` states the principle: *"grouping four rows into one member would run them on a
quarter of the cores"*. **Experiment:** on a kernel of ours that is occupancy-limited at decode, sweep
the number of K-parts against measured device time, and confirm the optimum is where
`members ≈ 24` (or a multiple). **What it decides:** whether our decode kernels are under-occupying the
array, which is a *different* failure mode from MLP and would explain a lot. **Cost:** a day if we
already have a shape-sweep harness.

**3. Measure the MME vs the TPC at one row, at *their* shapes and *ours*. — HIGH VALUE, settles §3.**
**Experiment:** §3.8's sweep, inside a timed graph, gated on the graph dump showing which node type was
created. **What it decides:** whether "TPC beats the MME at one row" is dead or whether our §3.2.1
standalone-launch curve understates the MME's in-graph cost. The second outcome would be a *manual
correction*, so this is worth doing even though I expect the first. **Cost:** a day, given the
`strata_gemv_bf16` kernel is in the tree and `tpc-clang` is in the image.

**4. Test the COLPAIR transposition on one of our GEMVs. — MEDIUM-HIGH, MEDIUM COST.**
Lane-per-output-row with a transposed weight layout removes the entire horizontal reduction **and** the
per-row masked select (`fp8_gemv.c:59-67` vs `fp8_gemv_colpair.c:43-47`). **Experiment:** take our
worst-performing GEMV where `N ≤ 64 × lanes`, re-lay the weight out `[rows/128][columns/2][128][2]`, and
A/B it. **What it decides:** whether the ~2 vector ops per output element we spend on reduction are
worth removing. **Caveat:** needs a layout pass at load, so it only pays if the weight is resident
anyway. **Cost:** 2–3 days including the layout builder.

**5. Measure what one external rebind costs per launch. — MEDIUM VALUE, VERY CHEAP.**
One recipe, N launches with one tensor rebound per launch vs N without; diff the device union. **What
it decides:** whether `DECODE_RECIPE_PER_LAYER` is a lever for us, and whether our §1.4.3 graph-count
model needs a rebind term. **Cost:** hours.

**6. Find out whether `TPC_ACCESS`-style `TensorAccessPattern` slicing is a route to SRAM pipelining
for a TPC node. — MEDIUM VALUE, SPECULATIVE.**
`glue.cpp:1363-1415` shows the compiler's slicer is gated on a glue-side affine map and that
`allRequired = 1` (the default) forbids it. Our §7.8 item 1 is stuck on "an MME operand marked
`inSram` adds a staging copy rather than replacing one". **Experiment:** take a TPC node we can slice,
publish an affine access pattern, and see whether the workspace gate (§7.5.2) shows the tensor
materialised or not. **What it decides:** whether the slicer, not `inSram`, is the way to get a
producer/consumer pair into SRAM. **Cost:** a week, and it may be a dead end — but it is a *different*
dead end from the one we have been in, and it is the mechanism the vendor's own kernel library uses.

**7. Ask the author the four questions this tree cannot answer. — HIGH VALUE PER MINUTE.**
(a) Why is `GEMV_TPC` off — did you measure it, and at what shape? (b) Why is `GEMV_TILE` off, given
your own comment says the shipped kernel runs at 237 GB/s = 10 % of peak? (c) What is `hpu/PROGRESS.md`
and can we see it? (d) `MOE_PAIR` amortises the codebook shuffle over two rows — did that pay, and
where? These are four questions, and (b) in particular is a puzzle I cannot resolve from the code: the
author documented the problem, wrote a kernel to fix it, and shipped neither the kernel nor a reason.

---

## 12. Claims we should **not** repeat without verifying

In priority order. The first is the one the brief flagged and it is right to flag it.

1. **"At one row the MME streams the weights at 1.3–1.6 TB/s."** No artifact in the repository; no
   A/B; no probe; the bounds quoted alongside it are the TPC kernel's own envelope; the switch is off
   in production; it cannot fire on the shipped hot path, which is a 4-row graph; and our own §3.2.1
   measures the MME at **2.19 TB/s** for an M=1 GEMV at N=8192. The *defensible* version is *"below
   ~16 MB of weight per launch the MME's M=1 GEMV loses to a TPC read loop, which our §3.2.1 curve
   already predicts"* — and even that we have not measured at their shapes. **Do not repeat the M=1
   framing.**
2. **"237 GB/s at the decode shape = 10 % of HBM peak"** for the five-tile GEMV. **THEY SAY
   (measured)** with no artifact. It is plausible and consistent with the published tok/s (§2.4), but
   we have not reproduced it, and it is load-bearing for the "this port is latency-bound" conclusion.
   Quote it as *their* figure.
3. **"155 tok/s"** as evidence of a well-optimised memory path. Our arithmetic says the port runs at
   ~10 % of the HBM roofline (§2.4). 155 tok/s is a **small model** (76 GB, H=2560, 512 experts × 640
   wide) at a tenth of peak. **Do not use this number as a benchmark of kernel quality.**
4. **"The MME tiles a `[1,HC*H]×[HC*H,rank+HC]` product poorly."** **THEY SAY**; no measurement. The
   *shape* (6.6 MB, N=324) is consistent with our §3.2.1, so the conclusion is probably right, but the
   claim is a comment.
5. **"Grouped expert reads load each expert's weights once per window instead of once per slot."**
   **THEY SAY** (`iq4q8_gemv_group.c:1-5`). What the code does is **reorder the members so the second
   read is L2-hot** — no load is shared and no instruction is saved (§5.2). The saving is HBM traffic
   and it depends entirely on L2 residency, which nobody measured. The **register-level** sharing is
   `MOE_PAIR`, a different kernel, and it is also off.
6. **"`STRATA_HPU_TPC_ACCESS` = affine access patterns for row norm and HC kernels."** Misleading. It
   changes no instruction the kernel executes; it publishes a `TensorAccessPattern` that lets the
   compiler's **slicer** split the node, and it applies to 8 kernel ids (§7.3 above). Say "enables
   slicing", not "affine access".
7. **"Measured slower so far"** for the row-sharing kernel (`resident_experts.cpp:160`). A real negative
   result, but it is a comment with no number, and the kernel is gated off `q8_packed` anyway. Quote it
   as a directional signal, not a measurement.
8. **Anything sourced from `hpu/PROGRESS.md` or `work/DESIGN_*.md`.** Neither file is in the
   repository (§0, §8). If a claim's only basis is one of those citations, the basis does not exist.
9. **"The port has ~89 performance switches."** `performance_modes.hpp` has **82** environment reads
   covering **70** distinct variables (I counted). Some are numeric rather than boolean, so "89
   switches" is at best approximate. Minor, but it is a number someone will quote. (The file itself is
   worth reading for its own sake: every switch carries a comment saying what it changes *and* whether
   it changes the arithmetic, which is better discipline than most of our flag documentation.)

---

## 13. What is genuinely good here, and worth copying

Not everything is a comment problem. Four things are better than what we have:

1. **Host-side emulation of TPC kernels for correctness testing.** `hpu/tests/moe_group.cpp:1-16` runs
   the *actual* `tpc/*.c` sources under scalar/vector shims and checks the contract ("no ISA execution,
   no device, no performance claim"). `hpu/tests/tpc_emulator.hpp` and the `*_semantics.cpp` tests do
   the same. **This lets them test kernel logic in CI without a card**, which is exactly the thing our
   own iteration loop is worst at (our §6). The shims are ~200 lines.
2. **Bit-exactness as a testable contract between kernel variants.** Every variant kernel's header
   states *why* it is bit-identical to the one it replaces ("same MAC order", "only the load order
   moves", "the four int32 accumulator vectors are rows 64(K-1)..64K-1 in order"), and
   `hpu/tests/gemv_throughput.cpp:141-159` `memcmp`s the outputs. So a performance variant is a
   *provable* drop-in, not a new arithmetic mode. That is a discipline our manual should state as a rule.
3. **A per-flag "arithmetic divergence" declaration.** Every switch whose output differs in the last
   bits says so in its own comment and names the disk-state marker that records it — e.g.
   `performance_modes.hpp:65-66` "the parts' F32 sums are added afterwards, so other values change the
   summation order of the projection (its own arithmetic)". Compare `hpu/src/model.cpp:175-330`, which
   is a wall of `throw std::invalid_argument` preconditions tying each switch to the others it needs.
   **A switch that cannot be enabled in an invalid combination is worth more than a switch with a
   comment**, and this is the pattern to copy.
4. **The `ops` knob on the bandwidth probe.** `ops ∈ {0,7}` separates "limited by requests in flight"
   from "limited by instruction issue" — the two things, as their comment says, that a GB/s number
   cannot separate. Our §7.6 instrument does not have this, and adding it would tell us whether our
   SRAM arm is memory-bound or array-bound, which is exactly the ambiguity our §7.6 flags as unresolved.

---

## 14. Appendix — what I read

**Kernels, in full** (`hpu/tpc/`): `gemv_bf16.c`, `iq4q8_gemv.c`, `iq4q8_gemv_pipe.c`,
`iq4q8_gemv_packed.c`, `iq4q8_gemv_packed_pf.c`, `iq4q8_gemv_tile.c`, `iq4q8_gemv_group.c`,
`iq4q8_gemv_pair.c`, `iq4q8_gemv_rows.c`, `route_group_starts.c`, `route_group_order.c`,
`route_group_blocks.c`, `route_topk.c` (head), `q8_quant_packed.c`, `gdn_step_cached.c`,
`gdn_step_pipe.c`, `gdn_step_rows4.c`, `fp8_gemv.c`, `fp8_gemv_colpair.c`, `fp8_gemv_hilo.c`,
`bw_read_probe.c`, `hc_down_bf16.c` (head), `hc_down_rows_bf16.c` (head), `moe_mme_gather.c` (head),
`iq4_expand_tiles.c` (head).

**`hpu/tpc/glue.cpp`** — all 1476 lines.

**Engine** (`hpu/src/`): `recipe.cpp` (`gemm` + `GraphBuilder`), `resident_experts.cpp`
(`define_resident_moe` + the variant selection), `model.cpp` (the switch-precondition wall),
`layer.cpp` (`decode_chain`, the per-layer recipe sites), `gdn.cpp` (`project_merged*`),
`model_head.cpp`, `moe_mme.cpp` (head), `batch_slots.cpp`, `protocol.cpp` (INFO line).

**Headers**: `performance_modes.hpp` (all 446 lines), `batch_slots.hpp`, `multi_slot_loop.hpp`,
`multi_slot_window.hpp` (head), `layer.hpp`, `model.hpp` (head), `geometry.hpp`, `model_abi.hpp`,
`dflash.hpp`, `resident_experts.hpp`, `u4_resident.hpp` (head).

**Probes and tests**: `hpu/tests/bw_read.cpp`, `gemv_throughput.cpp`, `moe_group.cpp` (head),
`batch_protocol.cpp` (the INFO-line assertion).

**Config and docs**: `hpu/deploy/strata-hpu-production.json`, `hpu/docker/strata-hpu.json`,
`hpu/deploy/run-production.sh`, `README.md`, `hpu/README.md`, `hpu/tools/generate.cpp` (the measured
prefill-window comment), `hpu/tests/` directory listing, `docs/` directory listing.

**Not read, and it matters:** `packed_matmul.c` (36 KB — the LM head's quantized matmul; it is a
separate technique from everything above and deserves its own pass), `qsa_select_mask.c` (11 KB),
`gdn_prepare_rows.c`, `qsa_*.c` beyond the heads, and the ~40 remaining kernels. Also not read:
`src/qsa*.cpp`, `src/dflash.cpp`, `src/expert_*.cpp`, `src/kv_*.cpp` — the attention and residency
machinery is a second investigation and none of the brief's claims live there.
