#!/usr/bin/env python3
"""Example: a minimal, annotated MoE decode stage sketch.

**Status: annotated sketch. Not executed as part of the manual. No HPU was used to produce it.**
It is written to show the *shape* of the routed-expert path, to mark the six stages, and to show
where the measurement hooks go. It is deliberately small — real shapes come from the model config.
Read chapter 3 §3.3 and §3.4 before using it.

WHAT THIS SKETCH IS FOR
-----------------------
The single most common accounting error in this path is counting **four** materialising stages
when there are **six**. The two that get forgotten — the **scale gather** and the
**`repeat_interleave` expand** — are among the largest op-count items in the whole step.

    1. gather packed fp4 weights      (by routing index)
    2. gather the scales              (by the same index)          <-- often forgotten
    3. build the exponent             (a bitshift on the scale)     <-- "kexp"
    4. expand / broadcast it          (repeat_interleave)           <-- often forgotten
    5. dequantize fp4 -> fp8          (one fused kernel pass)
    6. concatenate the two halves     (the kernel's two outputs)

    then: fp8_gemm_v2, which is 2-D ONLY, so it runs once per expert.

THE OPERAND CONTRACT (chapter 3 §3.3.4)
---------------------------------------
`hpu::fp8_gemm_v2(A, trans_A, B, trans_B, D, out_dtype, A_scale_inv, B_scale_inv, bias, accumulate)`
requires, per expert and per weight:

  * dtype `float8_e4m3fn`, consumed with the device's fp8 code semantics (ceiling 240);
  * shape `[N, K]`, 2-D, row-major contiguous, consumed with `trans_B=True` — so the LAST dim is
    the contraction dim;
  * `B_scale_inv=None` — the operand must be SCALE-FREE (the per-32-group e8m0 scale already
    folded into the fp8 exponent). The op has no 2-D block-scale input;
  * one tensor per expert.

**The K-axis order is FREE**: because the contraction is a sum over K, any permutation of K applied
to BOTH operands leaves the result unchanged. The reference implementation exploits this — it lets
the dequant kernel emit its native 64-byte-chunk interleave and permutes the much smaller
ACTIVATION to match, instead of un-permuting the large weight. MEASURED.

TAGS: MEASURED / ASSUMED / INFERRED / UNKNOWN, per the manual's convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Shapes. All values below are DeepSeek V4 Flash specific [DSv4F] and MEASURED.
# Substitute your own model's config.
# ---------------------------------------------------------------------------
H = 4096              # hidden size
INTER = 2048          # moe_intermediate_size  (expert w1/w3 output width)
N_LOCAL = 64          # local experts per rank  (= n_routed_experts / TP)
TOPK = 6              # num_experts_per_tok
N_LAYERS = 43

FP4_BLK = 64          # fp8 bytes per interleave chunk (32 packed fp4 bytes -> 64 fp8 bytes)

# The device's fp8 activation ceiling. MEASURED: 240 on Gaudi 2 (not 448).
DEVICE_FP8_MAX = 240.0


@dataclass
class StageCost:
    """What to record for each stage. Report ops AND merged device time, never sums alone."""
    name: str
    ops_per_step: int = 0
    merged_ms_per_step: float = 0.0


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def block_permutation(n_blocks: int, device) -> torch.Tensor:
    """The 64-element block permutation matching the dequant kernel's native output order.

    The kernel writes the EVEN 64-byte chunks into `o0` and the ODD chunks into `o1` (to avoid a
    stack/transpose). So the concatenated row is in the order
    `[0, 2, 4, ..., 62, 1, 3, ..., 63]` rather than `[0..63]`.

    The GEMM sum is unchanged if BOTH contraction operands are permuted identically — so we
    permute the (much smaller) activation rather than un-permuting the (much larger) weight.

    Cache this: it is a pure function of (n_blocks, device) and rebuilding it inside a traced
    region emits an arange/cat/to(device) per call.
    """
    return torch.cat([torch.arange(0, n_blocks, 2),
                      torch.arange(1, n_blocks, 2)]).to(device)


def permute_blocks(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply the block permutation along the last dim.

    Prefer `index_select` on the block dim over advanced indexing `[..., perm, :]`: on HPU the
    latter lowers to a full broadcast/tile/transpose + gather sequence (MEASURED at 155 device
    ops/iter vs 29 for `index_select`), and it dominates the per-step memcpy traffic.
    """
    d = x.shape[-1]
    return (x.reshape(*x.shape[:-1], d // FP4_BLK, FP4_BLK)
             .index_select(-2, perm)
             .reshape(*x.shape[:-1], d))


def build_kexp(scales: torch.Tensor, group_bytes: int) -> torch.Tensor:
    """Stage 3+4: turn per-group scales into a per-BYTE int8 exponent, broadcast over the group.

    The e8m0 scale is an exact power of two, so the dequant kernel can fold it by ADDING the
    exponent to the fp8 exponent field. That requires the exponent pre-decoded and pre-broadcast
    to the byte layout of `blocks` — which is what this function produces, and what makes it two
    of the six stages rather than a detail.

    `scales` may arrive as:
      * raw `uint8` e8m0 BYTES  -> decode as 2 ** (byte - 127), i.e. exponent = byte - 127;
      * `float8_e8m0fnu`        -> already the scale; extract the exponent;
      * fp32                    -> the loader converted it; extract the exponent by bit shift.

    NOTE (chapter 3 §3.4.4): a loader that converts e8m0 -> fp32 at load time is what forces the
    bitshift path here. Keeping the bytes avoids both the conversion and this stage's bitshift.
    """
    if scales.dtype == torch.uint8:
        kexp = scales.to(torch.int16) - 127
    elif scales.dtype == torch.float8_e8m0fnu:
        kexp = ((scales.view(torch.uint8).to(torch.int16)) - 127)
    else:
        # fp32 (or bf16): the exponent is the biased exponent field.
        kexp = ((scales.view(torch.int32) >> 23) & 0xFF) - 127
    kexp = kexp.to(torch.int8)
    # Stage 4: broadcast each per-group exponent across the group's bytes.
    return kexp.repeat_interleave(group_bytes, dim=1)


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def routed_expert_stage_sketch(
    flat: torch.Tensor,              # [T, H] activations for this layer
    packed_w: torch.Tensor,          # [N_LOCAL, ROWS, H//2] uint8  packed fp4
    packed_scale: torch.Tensor,      # [N_LOCAL, ROWS, H//32] scale (dtype varies)
    gather_ids: torch.Tensor,        # [G] int64, sorted, the routed experts on this rank
    fp4_dequant_op,                  # the fused TPC dequant op (blocks, kexp) -> (o0, o1)
    costs: list[StageCost] | None = None,
) -> torch.Tensor:
    """One expert weight's worth of the path, with the six stages marked.

    Returns the fp8 operand in the layout `fp8_gemm_v2` wants, viewed as `[G, ROWS, COLS]`.
    """
    costs = costs if costs is not None else []
    G = gather_ids.shape[0]
    rows, packed_cols = packed_w.shape[1], packed_w.shape[2]
    group_bytes = packed_cols // packed_scale.shape[-1]   # 16 for a 32-value e8m0 group

    # ---- Stage 1: gather packed fp4 weights ------------------------------------------------
    # MEASURED [DSv4F]: 3,096 ops/step, 3.85 ms merged. This is a real op with real cost; it is
    # NOT free just because it is "just indexing". If your op reads the ROUTED experts only, the
    # traffic is ~12.9x less than reading all local experts -- check this first (chapter 3 §3.4.5).
    p = packed_w[gather_ids]
    costs.append(StageCost("1 gather packed fp4", ops_per_step=G * rows))

    # ---- Stage 2: gather the scales --------------------------------------------------------
    # The commonly-forgotten stage. MEASURED [DSv4F]: 3,096 ops/step, 1.89 ms merged.
    s = packed_scale[gather_ids]
    costs.append(StageCost("2 gather scales", ops_per_step=G * rows))

    # ---- Stages 3+4: build and broadcast the exponent --------------------------------------
    # MEASURED [DSv4F]: the bitshift alone is 9,288 ops/step (1:1 with the dequant kernel);
    # the broadcast is 45,365 ops/step and 10.78 ms merged -- one of the largest op-count items
    # in the whole step, and it is easy to dismiss as "just a broadcast".
    kexp = build_kexp(s.reshape(-1, s.shape[-1]), group_bytes)
    costs.append(StageCost("3+4 kexp build + expand", ops_per_step=G * rows * 2))

    # ---- Stage 5: dequantize fp4 -> fp8, scale folded in -----------------------------------
    # MEASURED [DSv4F]: 9,288 ops/step, 22.63 ms merged, and ALU-bound: 42.38 of its 58.28 us
    # per event is ALU, and it is already at its memory wall (0.95x its own bound).
    # => the optimisation target here is the ALU, NOT the bandwidth.
    o0, o1 = fp4_dequant_op(p.reshape(-1, packed_cols), kexp)
    costs.append(StageCost("5 fp4 dequant kernel", ops_per_step=G * rows * 3))

    # ---- Stage 6: concatenate the two halves ----------------------------------------------
    # MEASURED [DSv4F]: 4,365 ops/step but only 0.22 ms merged.
    #
    # !! DEAD END (chapter 3 §3.3.5) !!  "Remove the cat" is NOT an actionable lever: the graph
    # compiler inserts its own `concat_logical` + `memcpy_dma` for GEMM operand staging
    # regardless, so a GEMM-containing recipe contains DMA layout ops whether or not this cat
    # exists. It is kept here because it is the correct thing to WRITE, not because removing it
    # is a win.
    wq = torch.cat([o0, o1], dim=-1)
    costs.append(StageCost("6 cat", ops_per_step=G * rows))

    # View as [G, rows, cols]. NOTE: the row order is whatever `gather_ids` gave; the column
    # (contraction) order is the kernel's native chunk interleave, which the consumer must match.
    return wq.view(G, rows, packed_cols * 2)


def expert_gemm_sketch(x: torch.Tensor, wq: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """The consumer. `fp8_gemm_v2` is 2-D ONLY, so this is a loop over experts.

    MEASURED [DSv4F]: at K = 6 experts and 43 layers x 3 weights that is 774 GEMM calls per
    decode step. A TPC kernel cannot replace this GEMM; the MME is what you want.
    """
    G = x.shape[0]
    outs = []

    # The activation is permuted with the SAME block permutation as the weight's contraction
    # axis, so the GEMM sum is unchanged and the large weight needs no un-permutation.
    x = permute_blocks(x, perm)

    for g in range(G):
        # Per-expert dynamic activation quantization. Use the DEVICE's fp8 ceiling (240), not
        # the checkpoint format's (448) -- using 448 clips the largest element of every expert
        # and is a measured 1.8x worse max-rel-err against a bf16 class.
        amax = x[g].abs().amax().clamp(min=1e-12)
        scale = (amax / DEVICE_FP8_MAX).to(torch.float32)
        xq = torch.ops.hpu.cast_to_fp8_v2(
            x[g], (1.0 / scale).to(torch.float32), False, False, torch.float8_e4m3fn)[0]

        outs.append(torch.ops.hpu.fp8_gemm_v2(
            A=xq,
            trans_A=False,
            B=wq[g],              # [N, K] fp8, contiguous; contraction is the LAST dim
            trans_B=True,
            D=None,
            out_dtype=torch.bfloat16,
            A_scale_inv=scale,
            B_scale_inv=None,     # SCALE-FREE: the e8m0 scale is already in the exponent
            bias=None,
            accumulate=False,
        ))
    return torch.stack(outs, 0).float()


# ---------------------------------------------------------------------------
# Measurement hooks — this is the part to copy into a real port.
# ---------------------------------------------------------------------------

MEASUREMENT_CHECKLIST = """
For each stage above, record ALL of these, before and after any change:

  * ops/step            -- and remember a device event is a REPLICATION count, not a launch:
                           one event per (launch x TPC/DMA engine). Kernel events / recipe
                           launches measured 544 / 569 / 971 across three traces of ONE model.
  * merged ms/step      -- the union of the stage's intervals, NOT the sum of durations.
                           merged is device OCCUPANCY; only the un-overlapped part counts.
  * bytes read/written  -- from a SHAPE x CALL-COUNT MODEL, because the profiler trace carries
                           NO BYTE COUNTS. Label it a model, and cross-check it against the
                           measured DMA merged time rather than trusting it.
  * us/op               -- so a fixed handshake cost is visible as such.

Then price the change in `merged`, never in summed ms and never in op count:
    wall = 21.7 + 0.997 * merged      (R^2 0.9999 over one 12-run set -- quote it with its set)

CHECKS BEFORE STARTING ANY OPTIMISATION HERE
  1. Does the op read the ROUTED experts or ALL LOCAL experts?  (~12.9x vs ~1.21x traffic win)
  2. Is the path already ONE multi-engine recipe with the dequant and GEMM in one submission?
     (If yes, "the overlap is missing" is not your problem.)
  3. Is the dequant ALU-bound or bandwidth-bound?  (Measured: ALU-bound, and already at its
     memory wall.)
  4. Is the stage you are about to optimise the LARGEST term?  (Attention beat the routed
     experts in the floor for our workload.)
"""

if __name__ == "__main__":
    print(__doc__.split("TAGS")[0])
    print("This sketch needs an HPU to execute. It is provided for structure and for the")
    print("measurement hooks.\n")
    print(MEASUREMENT_CHECKLIST)
