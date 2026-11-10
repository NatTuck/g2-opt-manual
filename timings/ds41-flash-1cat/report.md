# DeepSeek-V4.1-Flash on 4×Gaudi2: the 1Cat vLLM fork, measured

**What this is.** A measurement report on one serving stack — the `1Cat-vLLM-Gaudi` fork of
vLLM running DeepSeek-V4.1-Flash on four Gaudi2 cards — together with the findings that fell out
of getting it to run and measuring it. It is a report of what one box did, not a performance
claim about Gaudi2 in general.

**Provenance.** One host, four HL-225 cards, TP4×PP1, inside a standard Habana 1.24.2 container.
Engine tree pinned at `NatTuck/1Cat-vLLM-Gaudi` `e2266e0`; weights `deepseek-ai/DeepSeek-V4.1-Flash`
@ `dba1be0a`, prepared to 48 rank shards and 12 sidecars. Measured 2026-11-09/10.

**Basis tags.** As elsewhere in this manual: **MEASURED** (an instrument on real hardware, named),
**INFERRED** (derived by reasoning from measurements, derivation stated), **source read** (read out
of a named file, not executed), **UNKNOWN**. A number without its conditions is not a number, so
every rate below carries the shape and configuration it was taken at.

**Evidence.** The campaign's raw cells, logs and per-run findings live in an unpublished working
tree; the figures below are reproduced here in tables so that no claim rests only on a pointer to
it. Where a table's source is named, that name is provenance, not a link you can open.

## 1. What was installed

| item | value | basis |
|---|---|---|
| engine | 1Cat vLLM fork, `e2266e0`; vLLM `0.28.1rc1.dev392+ge47aa780b` | source read |
| plugin | `vllm_gaudi` from the same tree, editable into a workspace venv | source read |
| native runtime | patched Synapse / HCL / bridge from their pinned public commits; TP2 fused AR/norm extension; V4.1 TPC kernels + host gather | source read |
| weights | base `DeepSeek-V4.1-Flash` @ `dba1be0a`, prepared to 48 shards + 12 sidecars | source read |
| serving config | TP4×PP1, 524,288-token window, prefix caching on, block 128, chunk 16,384 | source read |
| deviations from the shipped defaults | `WOA_OUTPUT_ROUNDTRIP=0`, `PREFILL_OUTPUT_PROJECTION=0`, `MHC_RRMS_POST=0` in the arm that produced every number here — correctness workarounds, not tuning (§3.5) | MEASURED |

**Tested with** (required vs incidental is not the same thing):

| component | version tested | required? |
|---|---|---|
| driver | 1.24.2-52a555d | the driver generation the container expects |
| Habana userspace | 1.24.2.481 (`habana_frameworks`, `habana-torch-plugin`) | required — the native runtime is built against it |
| torch | 2.13.0a0+git2469d18 | required — the bridge's ABI is built against it |
| `tpc-clang` | 2.7.33 | required — the kernels are compiled with it |
| cmake | 4.4.0 | **≥ 4.0 required** by the bridge's CMakeLists |
| python | 3.12 | required by the pinned wheels |

Two build fixes are **not** in the repo and are required with this container generation: a CMake-4
fix to the bundled `lemon` policy (`CMP0048 OLD` was removed in CMake 4, and the
version-floor variable does not restore it), and a `-pie` restriction in the bridge's
`Targets.cmake` (it applied `-pie` to every target, and GNU ld makes `-shared`/`-pie`
last-one-wins, so every shared library silently became a PIE executable link — 105 undefined
references). Both are in the replication document with the exact file and edit.

## 2. Headline numbers

**MEASURED**, one cold lifetime: 16,384-token prompt **exactly**, one 16,384 chunk, official
sampling (T=1.0, top_p=0.95, seed=42), a never-used slice per round, `cached_tokens = [0,0,0]`,
natural EOS in two of three rounds.

| quantity | measured | repo's recorded baseline | ratio |
|---|---|---|---|
| prefill | **8,813 / 8,824 tok/s** (round 0: 8,544) | 8,029–8,915 tok/s | **inside the band** |
| decode | **9.478 / 9.509 ms/token** (ITL median 9.365, p90 10.40, n=1,701 intervals) | 8.239 ms/token | **1.15×** |

The third round of that cell produced a 66-token answer at 11.262 ms/token. It is excluded rather
than averaged: per-step fixed cost dominates a 65-interval output, which is why short outputs
mislead everywhere in this campaign.

A second, independent lifetime measured the same prefill shape as a steady median of rounds 1–2 at
**9,035 tok/s** and the same decode at **9.364 ms ITL**. The two prefill figures are different
cells with different slice seeds; both sit inside the recorded band, and the per-round values above
are the ones to quote because they share a lifetime with the decode figure.

The "recorded baseline" column is the 1Cat project's own archived measurement on the same model and
card count, quoted here as a claim about *their* run — we did not reproduce their environment.

## 3. What the tests showed

### 3.1 A real TP4 correctness defect, found and fixed — MEASURED

Prompts strictly between 4,096 and 16,384 tokens crashed the engine at TP4:

```
ValueError: Replicated decoder tail must fit the final TP4 token shard
```

`replicate_owned_tail` collected the decoder halo from the **last token owner only** and asserted
`retained <= local.shape[0]`, with `retained` fixed at 4096 and the shard at `T/4` — true only at
`T >= 16384`, while the branch is entered at `T > 4096`. The rows the suffix needs are the global
interval `[T-retained, T)`, which straddles owners at 8,192 and 12,000.

Fixed by reassembling the transaction with the module's existing `gather_tokens` when
`retained > local.shape[0]`; the `T >= 16384` fast path is byte-identical. A CPU test fails on the
unfixed code (10 failures at T=8192 and T=12000) and passes after (19/19). On device, 8,192 and
12,288 now run clean where 8,192 previously returned HTTP 500 and killed the engine.

### 3.2 Decode does not scale with context — MEASURED

| context | ITL (greedy) |
|---|---|
| 4,096 | 12.7 ms |
| 8,192 | 12.8 ms |
| 12,288 | 13.0 ms |
| 16,384 | 12.8 ms |

A 4× context increase costs nothing per token. Prefix-cache state does not affect decode ITL
either: 9.348 ms warm and 9.348 ms cold, to three decimals — an assumption every earlier A/B in the
campaign leaned on, now measured.

### 3.3 The chunk is the prefill variable, and only at full-chunk length — MEASURED

| prompt | chunk | cold prefill |
|---|---|---|
| 16,384 | **16,384** | **9,035 tok/s** |
| 16,384 | 8,192 | 5,175 tok/s |
| 11,000 | 16,384 | 2,204 tok/s |
| 11,000 | 8,192 | 2,147 tok/s |

At a full-chunk length the 16,384 chunk is 1.75× better; at 11,000 the two are indistinguishable,
because a prompt that does not fill a chunk costs nearly a full chunk. **Only full-chunk
transactions are reliable**; a non-chunk-aligned prompt measures the padding, not the engine.

The behaviour of non-aligned lengths is otherwise **UNKNOWN**. Three explanations were tested and
refuted: chunk fill (16,384 in one chunk is the cheapest per token), bucket edges (12,288 is not a
bucket and runs at 5,128 tok/s), and 512-alignment (10,240 is aligned and was the worst number
measured, 874 tok/s). 11,000 moved 26% between lifetimes. What would settle it: a sweep of aligned
lengths with three repeats each, round 0 reported separately.

### 3.4 The greedy penalty is parameter-driven — MEASURED, with one INFERRED step

Same lifetime, same prompt, same 16K context, 256 tokens:

| arm | parameters | ITL |
|---|---|---|
| greedy | T=0.0 | **13.544 ms** |
| nucleus | T=1.0, top_p=0.95 | 9.348 ms |
| top_k=1 | T=1.0, top_k=1 | 9.348 ms |

Two arms seeing identical tokens differ by 45%, so this is the sampling branch, not content
(MEASURED: the prompts are byte-identical by recorded slice ids). It is **not** the
full-vocabulary fallback: the runner's audit reads `bounded_steps: 255` with `fallbacks: 0`, so
the device sampler covers 255 of 256 steps.

`top_k=1` is as fast as nucleus but is **not** a greedy substitute: the ids agree for 25 tokens and
then diverge, because it samples from a bounded candidate packet rather than the global argmax.

The 1Cat project's own records show greedy slower than nucleus too (ratio 1.211); ours is 1.449,
so this build carries about 20% more greedy penalty than theirs — INFERRED from those two ratios.

### 3.5 The fused path: one switch, a broken sub-kernel, and a split that buys no speed — MEASURED

`WOA_OUTPUT_ROUNDTRIP` and `PREFILL_OUTPUT_PROJECTION` are not independent: `PREFILL=1` is refused
at construction without the roundtrip, and `WOA=1` alone kills the boot with a non-finite hidden
state in `_initialize_kv_caches`.

**The composed MLA op is innocent.** The flag reroutes *two* C1 decode chains. The composed MLA op
(`custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2`) is bit-exact against the
production unfused chain in all twelve geometries tested — 0 bf16 mismatches, `max_abs 0.0`, at
layers 2 and 20, positions 0 and 512, three cache states.

**The chain that actually runs first at boot is the ratio-0 layer handoff**
(`custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2`), and its output is **all 5,120 lanes
NaN**. The first non-finite stage inside it is
`custom_deepseek_v41_woa_scale_dense_quant_gaudi2`, whose `tokens == 1` branch iterates the
product's 1,024-lane channel axis instead of the `groups*8` tile owners, leaving the emitted tile
uninitialised and its store running roughly 64× past the end of the 2,048-element output row. The
evidence is the signature of an out-of-bounds read/write rather than arithmetic: the zero-input
invariant holds on layer 0 and is violated on layer 20, and the same call on the same operands
returned absmax 128.0 in a different process.

**A split, and why it is not the default.** Gating that handoff on a new
`VLLM_HPU_DSV41_WOA_DECODE_HANDOFF` (default `0`) lets the shipped configuration run: `WOA=1`,
`PREFILL=1`, handoff `0` boots and serves, 0 compiles after startup, 0 non-finite mentions, window
524,288. Measured in one cold lifetime:

| | `WOA=0` (the arm every number here uses) | split (`WOA=1`, handoff `0`) |
|---|---|---|
| decode ITL median, official | 9.365 ms | **9.277 ms** |
| decode, greedy 16K | 13.03 ms | **13.03 / 13.01 ms** |
| prefill, rounds 1+ | 8,818 tok/s | **8,630 tok/s (0.979×)** |
| 32-token greedy ids, same prompt | reference | **differ, first at index 7** |

Three things are therefore settled. (i) The C1 decode handoff is worth ~0 ms at T=1, so the split
costs no decode speed. (ii) The fused wo_a producer does **not** make prefill faster end to end:
the isolated 1.49–1.84× at 4,096–16,384 rows is a single-recipe saving that does not translate into
a step-wall saving, and the "~9% of a 16K prefill" figure that circulated earlier was a microbench
extrapolation — the measured effect is 0.979×. (iii) The split is **not** fidelity-neutral:
`WOA=1` also swaps the decode wo_a *producer* (`woa_fp8` → `woa_fp8_roundtrip`, the group-32 BF16
codec), which changes ~94% of that operand's lanes by about one bf16 ULP and flips near-tie
argmaxes. The sequences resynchronise (14 consecutive ids identical from index 11/13), so this is
quantisation noise rather than a structural break — but it is a real change in what the model says.

So the split is kept as a documented, safe, conformant alternative, not as the default. Its value
is that it runs the shipped configuration at all and removes an out-of-bounds kernel from the
serving path; it is not a speed win.

**The remaining 1.15× decode gap is not this, and not configuration** — INFERRED from a device-free
audit that compared the effective environment (108 keys, 103 on) against the entrypoint's own
fastpath tables, `envs.py`'s declared defaults, and the configuration documentation: our launch
differs from the entrypoint's intent in nothing. Configuration therefore cannot explain the gap,
and the search moves to the sampler branch (§3.4), the batch-path stall (§3.6) and the unattributed
device union of a decode step.

### 3.6 Batch does not work at this configuration — MEASURED

Widths above 1 collapse: aggregate pinned near 11 tok/s from width 2 to 16 while per-stream ITL
climbs ~77 ms per added row (186 ms at width 2, 1,259 ms at width 16), with zero compiles, no
preemption, and device utilization unchanged between width 1 and width 2.

| width | aggregate tok/s | per-stream tok/s | ITL median |
|---|---|---|---|
| 1 | 57.78 | 74.6–74.8 | **12.90 ms** |
| 2 | 10.23 | 5.2–5.4 | **186.10 ms** |
| 4 | 11.83 | 3.0–3.1 | **329.09 ms** |
| 16 | 11.46 | 0.8 | **1,259.12 ms** |

A flat per-row quantum is a stall, not computation — INFERRED; the mechanism is UNKNOWN. The
measurement was taken at chunk 8,192 and `max_num_seqs` 16, not at the headline chunk; the ladder
has not been repeated at 16,384.

Two documented facts bound how much this matters: the fork's qualified contract for this model is
**TP4×PP1, one request, greedy**, and its V2 adapter "supports only ordinary C1 decode". The fork
also states that full-model outputs at larger concurrencies still differ from its saved reference.

### 3.7 The additive bundle is not worth building for this — MEASURED

The `csrc/deepseek_v41_unique` addon builds host-only, but only its registration half: the kernel
library and its build manifest have no from-scratch builder in the tree. What it unlocks is
`MHC_RRMS_POST` (0.29 µs/token by the project's own figure) and default-off DSpark fusions. The
`mhc_rrms_post` op and its TPC kernel are already in the base extension; the only blocker is an
unconditional `load_unique_expert_operators()` call in the entrypoint.

## 4. What is open

| # | open item | what would settle it |
|---|---|---|
| 1 | 1.15× decode gap | not the composed MLA op (bit-exact) and not configuration (effective environment matches the entrypoint's intent); next: the sampler branch and a `merged` device-union budget for one decode step |
| 2 | batch > 1 collapses at ~77 ms/row | a per-step host CPU/fault split at width 2, and which runner the batch step takes |
| 3 | DSpark never enabled | needs the additive bundle; then a BS=1 cell with greedy id-equality against the no-draft arm |
| 4 | image input never exercised | the model has a vision tower and the checkpoint an `image_token_id`; the entrypoint's media path is untested |
| 5 | non-16k prefill lengths | a sweep of aligned lengths, three repeats each, round 0 separate |

## 5. Limits of this report

- Every number is from this install on this box. The only external figures are the 1Cat project's
  own recorded baseline, labelled as a claim about their run.
- Prefill is `prompt_tokens / TTFT`; decode is `(t_last − t_first) / (completion_tokens − 1)` over
  the whole output to natural EOS. Both are client-side, from streamed arrival timestamps.
- The three deviations in §1 are correctness workarounds on this build. The first two are now
  understood: the flag routes into an out-of-bounds kernel, and the split that avoids it boots and
  serves but is not faster and moves the token ids. `MHC_RRMS_POST=0` remains a build deviation.
- The split is measured in one cold lifetime and one seed set; its prefill figure (0.979×, rounds
  1+ of a three-round cell) is inside the recorded band, so "flat" is the claim, not "2% slower".
- The halo patch is applied to the working tree, not upstreamed.

## 6. What would differ on another box

- **Rates.** Everything in §2 and §3.3 is one host's result at TP4×PP1 on HL-225. Expect the
  absolute numbers to move; the *ratios* (chunk 16,384 vs 8,192, nucleus vs greedy, decode's
  flatness across context) are the transferable part.
- **Device identity.** Module ids are a property of the host's card population. Pin by module id,
  never by the index any monitoring tool prints.
- **Container generation.** The native runtime is built against a specific torch/ABI. A different
  userspace generation is a rebuild of the bridge, Synapse and HCL, not a re-link.
- **The two build fixes** are properties of CMake 4 plus this bridge pin; a different pin may not
  need them.
