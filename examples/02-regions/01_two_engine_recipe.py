#!/usr/bin/env python
"""Example 1 — the minimal two-engine recipe that overlaps.

WHAT THIS DEMONSTRATES
----------------------
Chapter 2 section 2.2:  `CROSS > 0` iff ONE Synapse submission AND cross-engine
AND no dataflow edge AND comparable op durations.

This file is the *smallest thing that overlaps*: one `torch.compile` region holding
an independent TPC op (`torch.mul`) and an MME op (`torch.matmul`).  Nothing else.

WHAT TO LOOK FOR (printed at the end)
-------------------------------------
  Launch/step          == 1.00       -> one submission            (condition 1)
  both engines present               -> cross-engine              (condition 2)
  CROSS                >  0          -> the families intersect
  union(all)           == union(TPC) -> the WHOLE MME family is hidden under the
                                        TPC, not a fixed slice of it
  precondition         OK            -> ops@H_end >= 2 AND device_busy >= 0.80

If `CROSS` comes back 0.0000, do NOT conclude "Gaudi 2 does not overlap".  Check the
precondition line first.  An arm whose queue is empty at the moment the host stops
issuing is INCONCLUSIVE, never a negative result (chapter 2, section 2.8).

WHY THE SIZES ARE WHAT THEY ARE  (chapter 2, section 2.2, condition 4)
---------------------------------------------------------------------
The two ops are duration-matched (~0.27 ms TPC vs ~0.20 ms MME when calibrated on
the machine this was developed on).  A 1:8 ratio pair hides only the small op and
reports a small-but-nonzero CROSS in BOTH the independent and the dependent variant
-- i.e. it cannot tell them apart.  Match your groups before you believe a result.

RUNNING IT
----------
  source activate.sh
  ./venv/bin/python 01_two_engine_recipe.py

Environment notes that each cost a run:
  * do NOT set HABANA_VISIBLE_DEVICES in a container that already sets it -- inside the container
    it is inert (INFERRED: nothing in the image reads it), so it neither helps nor isolates;
  * GC_KERNEL_PATH must never be unset (an empty value fails every compile);
  * collect ONLY ProfilerActivity.HPU.  Adding CPU activity inflates the trace's
    host span ~3x and manufactures a false empty-queue verdict (section 2.9, rule 6).
"""
from __future__ import annotations

import collections

import torch
import habana_frameworks.torch  # noqa: F401  (must be imported for the HPU device)

BACKEND = "hpu_backend"

# Sizes.  Pick them so the two ops have comparable SOLO durations on YOUR machine.
TPC_ELEMS = 100_663_296          # 192 MB bf16  -> ~0.27 ms on a TPC
MME_N = 3072                     # 3072^2 bf16  -> ~0.20 ms on the MME

STEPS = 20
WARM = 5


# ---------------------------------------------------------------------------
# the body.  A real module-level function (not a lambda): dynamo's
# (filename, lineno) recompile accounting sees each body distinctly, which is how
# you keep arms from contaminating each other's cache entries.
# ---------------------------------------------------------------------------
def body_tpc_mme(x, a, b):
    y = torch.mul(x, 2.0)          # TPC  -- reads x only
    z = torch.matmul(a, b)         # MME  -- reads a, b only.  NO edge to y.
    return y, z


# ---------------------------------------------------------------------------
# trace reduction -- section 2.9, rule 1: durations from UNIONS, never sums.
# ---------------------------------------------------------------------------
def merge(intervals):
    """|merge(intervals)| in us.  Intervals are (start, end) tuples."""
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e:                      # overlaps or touches
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def family_of(name):
    """Bucket a device kernel name into an engine family.

    NOTE (section 2.9, rule 3): this is op TAXONOMY, not engine identity.  It is
    good enough to compute CROSS.  When you need to know which engine a node is
    on, read the compiler's own per-node `engine` field from the graph dump --
    see example 3.
    """
    n = (name or "").lower()
    if "gemm" in n or "mme" in n:
        return "MME"
    if "dma" in n or "transpose" in n:
        return "DMA"
    return "TPC"


def measure(fn, args, steps=STEPS):
    """Run `fn` `steps` times under an HPU-only profiler; return (events, steps)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(WARM):
        fn(*args)
    torch.hpu.synchronize()

    with profile(activities=[ProfilerActivity.HPU]) as prof:   # HPU ONLY
        for _ in range(steps):
            fn(*args)
        torch.hpu.synchronize()
    return prof.events(), steps


def reduce_trace(events, steps, ops_per_step):
    """-> the section-2.2 quantities, in ms/step."""
    fam_iv = collections.defaultdict(list)
    all_iv = []
    launches = 0
    launch_ts = []

    for ev in events:
        name = ev.name or ""
        dur = float(getattr(ev, "device_time", 0.0) or 0.0)
        tr = getattr(ev, "time_range", None)
        ts = float(tr.start) if tr is not None else 0.0
        if dur > 0:
            fam_iv[family_of(name)].append((ts, ts + dur))
            all_iv.append((ts, ts + dur))
        if name in ("Launch", "LaunchRecipe", "enqueueWithExternalEventsExt"):
            launches += 1
            launch_ts.append(ts)

    if not all_iv:
        return None
    dev_span = max(e for _, e in all_iv) - min(s for s, _ in all_iv)
    fam_u = {f: merge(iv) for f, iv in fam_iv.items()}
    union_all = merge(all_iv)

    # `per_op` and the E5 backlog metric: already-issued, unexecuted device work
    # in declared-op units.
    per_op = dev_span / (steps * ops_per_step)
    kernel_end = max(e for _, e in all_iv)
    h_end = max(launch_ts) if launch_ts else 0.0
    S = 1.0 / steps / 1000.0            # us -> ms/step
    return {
        "fam_union_ms": {f: v * S for f, v in fam_u.items()},
        "union_all_ms": union_all * S,
        "CROSS_ms": (sum(fam_u.values()) - union_all) * S,
        "launch_per_step": launches / steps,
        "ops_at_h_end": (kernel_end - h_end) / per_op if per_op else float("nan"),
        "device_busy": union_all / dev_span if dev_span else 0.0,
    }


def main():
    dev = "hpu"
    x = torch.randn(TPC_ELEMS, dtype=torch.bfloat16, device=dev)
    a = torch.randn(MME_N, MME_N, dtype=torch.bfloat16, device=dev)
    b = torch.randn(MME_N, MME_N, dtype=torch.bfloat16, device=dev)

    # ONE region, one submission.  `fullgraph=True` is what makes it one region: a
    # graph break gives you two regions and two submissions, and then CROSS = 0.
    compiled = torch.compile(body_tpc_mme, backend=BACKEND, fullgraph=True, dynamic=False)

    events, steps = measure(compiled, (x, a, b))
    r = reduce_trace(events, steps, ops_per_step=2)
    if r is None:
        print("no device events captured -- is this running on an HPU?")
        return 1

    print("\n" + "=" * 68)
    print("  minimal two-engine recipe  (TPC mul + MME matmul, independent)")
    print("=" * 68)
    for f in sorted(r["fam_union_ms"]):
        print("  union(%-4s)         %8.4f ms/step" % (f, r["fam_union_ms"][f]))
    print("  union(all)          %8.4f ms/step" % r["union_all_ms"])
    print("  CROSS               %8.4f ms/step   <- >0 means real overlap" % r["CROSS_ms"])
    print("  Launch/step         %8.2f          <- want exactly 1.00" % r["launch_per_step"])
    print("  ops available @H_end%8.2f          <- want >= 2" % r["ops_at_h_end"])
    print("  device busy         %8.1f %%        <- want >= 80 %%" % (r["device_busy"] * 100))

    ok = r["ops_at_h_end"] >= 2 and r["device_busy"] >= 0.80
    print("\n  precondition: %s" % ("OK" if ok else "FAILED -> INCONCLUSIVE"))
    if not ok:
        print("  The queue was empty when the host stopped issuing.  Make the ops")
        print("  bigger or add depth until the precondition passes; the overlap")
        print("  question is unanswerable in this arm.  Do NOT report a negative.")
        return 2

    tpc = r["fam_union_ms"].get("TPC", 0.0)
    if tpc > 0 and abs(r["union_all_ms"] - tpc) < 1e-6:
        print("\n  union(all) == union(TPC) exactly -> the WHOLE MME family is hidden,")
        print("  not a fixed slice.  Confirm the nesting too: MME block k should sit")
        print("  inside TPC block k with zero straddles (a shift test on the trace).")
    if r["CROSS_ms"] <= 0.017:
        print("\n  CROSS is at/below the noise floor.  Before believing it, check that")
        print("  the ops are duration-matched and that Launch really is 1.00.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
