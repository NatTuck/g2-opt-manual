# 1Cat-vLLM-Gaudi on 4×Gaudi2 — install, measurement, and what the tests showed

Status: install complete and measured. Prefill matches the repo's recorded baseline; decode is
1.15× off it. The batch path is unqualified and does not work at our configuration. Everything
below is measured on this box, in this install; nothing is carried over from another tree.

- Install: `cases/1cat/` — clean build of `NatTuck/1Cat-vLLM-Gaudi` @ `e2266e0` from public
  sources, on a stock Habana 1.24.2 container, TP4×PP1, four HL-225.
- Replication: `cases/1cat/REPLICATE.md`. Install detail: `cases/1cat/INSTALL.md`.
- Evidence: `cases/1cat/measurements/*/FINDINGS.md` and the per-cell JSON next to them.

## 1. What is installed

| item | value |
|---|---|
| engine | 1Cat vLLM fork, `e2266e0`, vLLM `0.28.1rc1.dev392+ge47aa780b` |
| plugin | `vllm_gaudi` from the same tree, installed editable into a workspace venv |
| native runtime | patched Synapse / HCL / bridge built from their pinned public commits; TP2 fused AR/norm extension; V4.1 TPC kernels + host gather |
| weights | base `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a` prepared to 48 rank shards + 12 sidecars |
| serving config | TP4×PP1, 512,288 context, prefix caching on, block 128, greedy or official sampling, chunk 16,384 |
| deviations from the shipped defaults | `WOA_OUTPUT_ROUNDTRIP=0`, `PREFILL_OUTPUT_PROJECTION=0`, `MHC_RRMS_POST=0` in the arm that produced every number here. A conformant alternative exists (`WOA=1`, `PREFILL=1`, the C1 decode handoff off) and is **not** faster — see §3.5 |

Two build fixes are **not** in the repo and are required on this container: a CMake-4 fix to the
bundled `lemon` policy (`CMP0048 OLD` was removed in CMake 4, and the version-floor variable does
not restore it), and a `-pie` restriction in the bridge's `Targets.cmake` (it applied `-pie` to
every target, and GNU ld makes `-shared`/`-pie` last-one-wins, so every shared library silently
became a PIE executable link — 105 undefined references). Both are in `INSTALL.md` with the
exact file and edit.

## 2. The headline numbers

The only cell that answers the question, and the one every earlier number is superseded by
(`measurements/conformance2/FINDINGS.md`): 16,384-token prompt **exactly**, one 16,384 chunk,
official sampling (T=1.0, top_p=0.95, seed=42), never-used slice per round, `cached_tokens=[0,0,0]`,
natural EOS in two of three rounds, one lifetime.

| quantity | ours | repo's recorded baseline | ratio |
|---|---|---|---|
| prefill | **8,813 / 8,824 tok/s** (round 0: 8,544) | 8,029–8,915 tok/s | **inside the band** |
| decode | **9.478 / 9.509 ms/token** (ITL median 9.365) | 8.239 ms/token | **1.15×** |

Round 1 of that cell produced a 66-token answer at 11.262 ms/token; it is not comparable (per-step
fixed cost dominates a 66-interval output) and is excluded rather than averaged.

§3.3 quotes a slightly higher prefill figure (9,035 tok/s) for the same 16,384-token, one-chunk
shape. It is a **different cell** — `measurements/prefillcold/`, its own lifetime and slice seed,
reported as the median of rounds 1–2 with round 0 (which carries the shape compile) excluded —
against this table's per-round values from `conformance2/`. The two are consistent within the
repo's recorded band; neither supersedes the other, and the per-round `conformance2` numbers are
the ones to quote for the headline because they come from the same lifetime as the decode figure.

## 3. What the tests showed

### 3.1 A real TP4 correctness defect, found and fixed

Prompts strictly between 4,096 and 16,384 tokens crashed the engine at TP4:

```
ValueError: Replicated decoder tail must fit the final TP4 token shard
```

`replicate_owned_tail` collected the decoder halo from the **last token owner only** and asserted
`retained <= local.shape[0]`, with `retained` fixed at 4096 and the shard at `T/4` — true only at
`T >= 16384`, while the branch is entered at `T > 4096`. The rows the suffix needs are the global
interval `[T-retained, T)`, which straddles owners at 8,192 and 12,000.

Fixed by reassembling the transaction with the module's existing `gather_tokens` when
`retained > local.shape[0]`; the `T >= 16384` fast path is byte-identical. CPU test red before
(10 failures at T=8192 and T=12000), green after (19/19). Verified on device: 8,192 and 12,288 now
run clean where 8,192 previously returned HTTP 500 and killed the engine.
Details: `notes/halo-fix.md`.

### 3.2 Decode does not scale with context

| context | ITL (greedy) |
|---|---|
| 4,096 | 12.7 ms |
| 8,192 | 12.8 ms |
| 12,288 | 13.0 ms |
| 16,384 | 12.8 ms |

A 4× context increase costs nothing per token. Prefix-cache state also does not affect decode
ITL: 9.348 ms warm and 9.348 ms cold, to three decimals — an assumption every earlier A/B leaned
on, now measured.

### 3.3 The chunk is the prefill variable, and only at full-chunk length

| prompt | chunk | cold prefill |
|---|---|---|
| 16,384 | **16,384** | **9,035 tok/s** |
| 16,384 | 8,192 | 5,175 tok/s |
| 11,000 | 16,384 | 2,204 tok/s |
| 11,000 | 8,192 | 2,147 tok/s |

At a full-chunk length the 16,384 chunk is 1.75× better; at 11,000 the two are indistinguishable,
because a prompt that does not fill a chunk costs nearly a full chunk. **Only full-chunk
transactions are reliable**; a non-chunk-aligned prompt measures the padding, not the engine.
The anomaly is not explained by chunk fill, bucket edges or 512-alignment — three hypotheses
tested and refuted, including 10,240 tokens measuring 874 tok/s, the worst number in the campaign.
Recorded as an open observation in `measurements/prefillbucket/`.

### 3.4 The greedy penalty is parameter-driven

Same lifetime, same prompt, same 16K context, 256 tokens:

| arm | parameters | ITL |
|---|---|---|
| greedy | T=0.0 | **13.544 ms** |
| nucleus | T=1.0, top_p=0.95 | 9.348 ms |
| top_k=1 | T=1.0, top_k=1 | 9.348 ms |

Two arms seeing identical tokens differ by 45%, so this is the sampling branch, not content.
It is **not** the full-vocabulary fallback: the runner's audit reads `bounded_steps: 255` with
`fallbacks: 0`, so the device sampler covers 255 of 256 steps. `top_k=1` is fast but is **not** a
greedy substitute — ids agree for 25 tokens and then diverge, because it samples from a bounded
candidate packet rather than the global argmax.

The repo's own records show greedy slower than nucleus too (ratio 1.211); ours is 1.449, so we
own about 20% excess greedy penalty — but on the nucleus arm we actually serve, the gap is the
1.15× in §2.

### 3.5 The fused path: one switch, a broken sub-kernel, and a split that buys no speed

`WOA_OUTPUT_ROUNDTRIP` and `PREFILL_OUTPUT_PROJECTION` are not independent: `PREFILL=1` is
refused at construction without the roundtrip, and `WOA=1` alone kills the boot with a
non-finite hidden state in `_initialize_kv_caches`. The producers themselves are bit-exact at
every shape (T=1/512/4096/8192/16384, eager and compiled, within one bf16 ULP of the unfused
producer), so the defect is in the composition.

**It is now located, and it is not the composed MLA op.** The flag reroutes *two* C1 decode
chains. The composed MLA op
(`custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2`) is **bit-exact against the
production unfused chain in all twelve geometries tested** — 0 bf16 mismatches, `max_abs 0.0`,
at layers 2 and 20, positions 0 and 512, three cache states — and is exonerated. The chain that
actually runs first at boot is the **ratio-0 layer handoff**
(`custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2`, `ops/deepseek_v41_paged_attention.py:1637`),
and its output is **all 5120 lanes NaN**. The first non-finite stage inside it is
`custom_deepseek_v41_woa_scale_dense_quant_gaudi2`, whose `tokens == 1` branch iterates the
product's 1024-lane channel axis instead of the `groups*8` tile owners, leaving the emitted tile
uninitialised and its store running ~64x past the end of the 2048-element output row. Contract,
first non-finite stage, verdict and the exoneration: `notes/composed-graph.md`.

**The split, and why it is not the default.** Gating that handoff on a new
`VLLM_HPU_DSV41_WOA_DECODE_HANDOFF` (default `0`) lets the repo's own shipped configuration run:
`WOA=1` + `PREFILL=1` + handoff `0` boots and serves, 0 compiles after startup, 0 non-finite
mentions, `max_model_len` 524,288. Measured in one cold lifetime
(`measurements/woasplit/FINDINGS.md`):

| | `WOA=0` (the arm every number here uses) | split (`WOA=1`, handoff `0`) |
|---|---|---|
| decode ITL median, official | 9.365 ms | **9.277 ms** |
| decode, greedy 16K | 13.03 ms | **13.03 / 13.01 ms** |
| prefill, rounds 1+ | 8,818 tok/s | **8,630 tok/s (0.979x)** |
| 32-token greedy ids, same prompt | reference | **differ, first at index 7** |

**Two things are therefore settled and one is withdrawn.** (i) The C1 decode handoff is worth
~0 ms at T=1, so the split costs no decode speed. (ii) The fused wo_a producer does **not**
make prefill faster end to end: the isolated 1.49-1.84x at 4,096-16,384 rows is a single-recipe
union saving that does not translate into a step-wall saving, and the "~9 % of a 16K prefill"
figure that circulated earlier was a microbench extrapolation and is **not confirmed** — the
measured effect is 0.979x. (iii) The split is **not fidelity-neutral**: `WOA=1` also swaps the
decode wo_a *producer* (`woa_fp8` -> `woa_fp8_roundtrip`, the group-32 BF16 codec), which
changes ~94 % of that operand's lanes by about one bf16 ULP and flips near-tie argmaxes after
40 layers. The sequences resynchronise (14 consecutive ids identical from index 11/13), so this
is quantisation noise rather than a structural break, but it is a real change in what the model
says.

**So the split is kept as a documented, safe, conformant alternative, not as the default.** Its
value is that it runs the shipped configuration at all and removes an out-of-bounds kernel from
the serving path; it is not a speed win. If id fidelity is wanted, pinning the decode producer
too (one line, `project_output`) makes decode byte-identical — not implemented, not measured,
and it cannot be justified on speed either.

**The remaining 1.15x decode gap is not this and not configuration.** A device-free conformance
audit (`scripts/flag-conformance.py`) compares our launch against the entrypoint's own fastpath
tables for TP4xPP1 / DSpark off / V2 on, against `envs.py`'s declared defaults, and against
`docs/configuration/env_variables.md`: **our launch differs from the entrypoint's intent in
nothing** (108 effective keys, 103 on, 5 off; the only keys we set that the entrypoint does not
manage are `MHC_RRMS_POST=0`, whose operator is in the unbuilt additive bundle, and
`PREFILL_COMPUTE_TOKENS=16384`, which TP4 derives anyway). Configuration therefore cannot
explain the gap, and the search moves to the sampler branch (§3.4), the batch-path stall (§3.6)
and the unattributed device union of a decode step.

### 3.6 Batch does not work, and the docs say it is not qualified

Widths above 1 collapse: aggregate pinned near 11 tok/s from width 2 to 16 while per-stream ITL
climbs ~77 ms per added row (186 ms at width 2, 1,259 ms at width 16), with zero compiles, no
preemption, and device utilization unchanged between width 1 and width 2. A flat per-row quantum
is a stall, not computation. Unresolved.

**Configuration of that measurement, stated because it is not the headline config:** chunk
**8,192** and `--max-num-seqs 16` (`measurements/run3/probe/`, a `run-phase9.sh` boot with only the
two fusion flags overridden). The width ladder has not been repeated at chunk 16,384; the table
above is therefore a statement about the batch path at 8,192, not at the headline chunk.

Two documented facts bound how much this matters: the repo's qualified contract for this model is
**TP4×PP1, one request, greedy** (`docs/features/deepseek_v41.md:5`), and the V2 adapter
"supports only ordinary C1 decode". Any width>1 figure also carries the repo's own caveat that
full-model outputs at larger concurrencies still differ from the saved reference.

### 3.7 The additive bundle is not worth building for this

`csrc/deepseek_v41_unique/pytorch/setup.py` builds host-only, but it is only the registration
half: the kernel library and its build manifest have no from-scratch builder in the tree. What it
unlocks is `MHC_RRMS_POST` (0.29 µs/token) and default-off DSpark fusions. The `mhc_rrms_post` op
and its TPC kernel are already in the base extension; the only blocker is an unconditional
`load_unique_expert_operators()` at `entrypoints/deepseek_v41.py:791-795`.
Details: `notes/additive-bundle.md`.

## 4. What is open

| # | open item | what would settle it |
|---|---|---|
| 1 | 1.15× decode gap | **not** the composed MLA op (bit-exact, exonerated) and **not** configuration (our launch matches the entrypoint's intent in nothing differing — `scripts/flag-conformance.py`). Next: the sampler branch (§3.4) and the unattributed device union of a decode step, priced as a `merged` budget |
| 2 | batch > 1 collapses at ~77 ms/row | a per-step host CPU/fault split at width 2, and a check of which runner the batch step takes |
| 3 | DSpark never enabled | needs the additive bundle; then a BS=1 cell with greedy id-equality against the no-draft arm |
| 4 | image input never exercised | the model has a vision tower and the checkpoint an `image_token_id`; the entrypoint's media path is untested here |
| 5 | non-16k prefill lengths unexplained | a sweep of aligned lengths with three repeats each, reporting round 0 separately |

## 5. Honest limitations of this report

- Every number is from this install on this box. No figure from any other tree is used as a
  baseline or a target; the only external numbers are the repo's own recorded baseline, labelled
  as such.
- Decode and prefill are measured with a client whose definition is fixed in
  `measurements/run2/method.md`; the prefill convention is `prompt_tokens / TTFT` and the decode
  convention is `(t_last − t_first) / (completion_tokens − 1)` over the whole output to EOS.
- The three deviations in §1 (`WOA=0`, `PREFILL=0`, `MHC_RRMS_POST=0`) are **correctness
  workarounds on this build**, not tuning choices. The first two are now understood: the flag
  routes into an out-of-bounds kernel (`notes/composed-graph.md`), and a split that keeps the
  fused producer while gating that kernel off boots and serves but is **not faster** and moves
  the token ids (`measurements/woasplit/FINDINGS.md`). `MHC_RRMS_POST=0` remains a build
  deviation, not a choice.
- The `WOA=1` split is measured in one cold lifetime and one seed set; its prefill figure
  (0.979x, rounds 1+ of a 3-round cell) is inside the repo's own recorded band, so "flat" is
  the claim, not "2 % slower".
- The halo patch is applied to the working tree, not upstreamed.
