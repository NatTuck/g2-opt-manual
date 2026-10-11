#!/usr/bin/env python
"""Example 2 — two regions that differ by ONE dataflow edge.

WHAT THIS DEMONSTRATES
----------------------
Chapter 2 section 2.2, condition 3: "no dataflow edge between the groups".  The
chapter's cleanest evidence is a matched pair: identical shapes, identical ops,
identical submission count -- and the ONLY difference is whether the MME's first
operand is the TPC's output (an edge) or an independent tensor (no edge).

    independent :  y = mul(x2d, 2) ; z = mm(x2d, b)     <- both read x2d.  No edge.
    dependent   :  y = mul(x2d, 2) ; z = mm(y,   b)     <- z consumes y.  Edge.

    independent -> CROSS > 0        (the TPC hides ENTIRELY under the MME)
    dependent   -> CROSS == 0.0000  (union(all) == sum of the family unions, exactly)

WHY THE SINGLE-SHOT HARNESS
---------------------------
`union(all) == union(one family)` is ALSO consistent with cross-step pipelining:
step k's MME hiding under step k+1's TPC.  So this file calls
`torch.hpu.synchronize()` before EVERY call, which guarantees exactly one recipe is
in flight and removes pipelining as an explanation.  If you only ever measure the
back-to-back loop, you cannot tell those two stories apart.

SIZES MATTER (condition 4).  At a 1:8 TPC:MME ratio this pair is NOT discriminating
-- a small TPC is partly hidden in BOTH variants.  The sizes below give ~0.27 ms
TPC against ~0.41 ms MME, i.e. about 1:1.5.

RUNNING IT
----------
  source activate.sh
  ./venv/bin/python 02_region_that_overlaps_and_one_that_does_not.py
"""
from __future__ import annotations

import collections

import torch
import habana_frameworks.torch  # noqa: F401

BACKEND = "hpu_backend"

TPC_ELEMS = 100_663_296      # 192 MB bf16 -> ~0.27 ms on a TPC
MME_N = 4096                 # 4096^2 bf16 -> ~0.41 ms on the MME
CALLS = 20
WARM = 5


# ---------------------------------------------------------------------------
# The two bodies.  They differ on exactly one line -- which operand the matmul
# reads.  Keep them as real module-level functions so dynamo's recompile
# accounting can tell them apart.
# ---------------------------------------------------------------------------
def body_indep(x2d, b):
    y = torch.mul(x2d, 2.0)        # TPC  -- reads x2d
    z = torch.matmul(x2d, b)       # MME  -- reads x2d and b.  NO edge to y.
    return y, z


def body_dep(x2d, b):
    y = torch.mul(x2d, 2.0)        # TPC
    z = torch.matmul(y, b)         # MME  -- consumes y.  THE EDGE.
    return z


# ---------------------------------------------------------------------------
# reduction (section 2.9, rule 1: unions, never sums)
# ---------------------------------------------------------------------------
def merge(intervals):
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    s0, e0 = iv[0]
    for s, e in iv[1:]:
        if s <= e0:
            e0 = max(e0, e)
        else:
            total += e0 - s0
            s0, e0 = s, e
    return total + (e0 - s0)


def family_of(name):
    n = (name or "").lower()
    if "gemm" in n or "mme" in n:
        return "MME"
    if "dma" in n or "transpose" in n:
        return "DMA"
    return "TPC"


def single_shot(fn, args, calls=CALLS):
    """One recipe in flight per call -- pipelining is impossible by construction."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(WARM):
        torch.hpu.synchronize()
        fn(*args)
    torch.hpu.synchronize()

    with profile(activities=[ProfilerActivity.HPU]) as prof:
        for _ in range(calls):
            torch.hpu.synchronize()        # <- the single-shot guarantee
            fn(*args)
        torch.hpu.synchronize()

    fam_iv = collections.defaultdict(list)
    all_iv = []
    launches = 0
    for ev in prof.events():
        dur = float(getattr(ev, "device_time", 0.0) or 0.0)
        tr = getattr(ev, "time_range", None)
        ts = float(tr.start) if tr is not None else 0.0
        if dur > 0:
            fam_iv[family_of(ev.name)].append((ts, ts + dur))
            all_iv.append((ts, ts + dur))
        if ev.name in ("Launch", "LaunchRecipe", "enqueueWithExternalEventsExt"):
            launches += 1

    fam_u = {f: merge(iv) for f, iv in fam_iv.items()}
    union_all = merge(all_iv)
    S = 1.0 / calls / 1000.0               # us -> ms/call
    return {
        "fam_union_ms": {f: v * S for f, v in fam_u.items()},
        "union_all_ms": union_all * S,
        "sum_fam_ms": sum(fam_u.values()) * S,
        "CROSS_ms": (sum(fam_u.values()) - union_all) * S,
        "launch_per_call": launches / calls,
    }


def main():
    dev = "hpu"
    x = torch.randn(TPC_ELEMS, dtype=torch.bfloat16, device=dev)
    b = torch.randn(MME_N, MME_N, dtype=torch.bfloat16, device=dev)

    results = {}
    for tag, body in (("independent", body_indep), ("dependent", body_dep)):
        f = torch.compile(body, backend=BACKEND, fullgraph=True, dynamic=False)
        results[tag] = single_shot(f, (x, b))

    print("\n" + "=" * 74)
    print("  one dataflow edge decides it   (single-shot: one recipe in flight)")
    print("=" * 74)
    hdr = "  %-12s %9s %9s %9s %9s %8s  %s"
    print(hdr % ("body", "union(TPC)", "union(MME)", "union(all)", "sum(fam)", "CROSS", "verdict"))
    for tag in ("independent", "dependent"):
        r = results[tag]
        cross = r["CROSS_ms"]
        verdict = "OVERLAP" if cross > 0.017 else "serialised"
        print(hdr % (tag,
                     "%.4f" % r["fam_union_ms"].get("TPC", 0.0),
                     "%.4f" % r["fam_union_ms"].get("MME", 0.0),
                     "%.4f" % r["union_all_ms"],
                     "%.4f" % r["sum_fam_ms"],
                     "%.4f" % cross,
                     verdict))

    ind, dep = results["independent"], results["dependent"]
    print("\n  What to check, and why:")
    if abs(ind["union_all_ms"] - ind["fam_union_ms"].get("MME", -1)) < 1e-6:
        print("   * independent: union(all) == union(MME) exactly")
        print("     -> the TPC is ENTIRELY hidden under the MME, not partly.")
    else:
        print("   * independent: union(all) != union(MME).  Either the pair is not")
        print("     size-matched (condition 4) or the two ops are not in one")
        print("     submission -- check Launch/call: %.2f" % ind["launch_per_call"])
    if abs(dep["CROSS_ms"]) < 1e-9:
        print("   * dependent:   CROSS == 0.0000 exactly, and union(all) == sum(fam).")
        print("     One edge, and the whole saving is gone.  Same ops, same shapes,")
        print("     same submission count -- the ONLY difference is the operand.")
    else:
        print("   * dependent:   CROSS = %.4f, not 0.  Something else is going on:"
              % dep["CROSS_ms"])
        print("     the two ops may not be in one submission, or the pair may be")
        print("     unequally sized so a small TPC is partly hidden either way.")
    print("\n  Do NOT read the 4096^2 sizes as evidence on their own: at a 1:8")
    print("  TPC:MME ratio this pair is non-discriminating.  Size-match first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
