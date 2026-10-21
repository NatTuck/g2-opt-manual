#!/usr/bin/env python3
"""gap_histogram.py -- decompose the gap between step wall and device union, by hole width.

WHY THIS EXISTS
---------------
`wall - merged` is the part of a step in which NO engine is executing. It is the quantity an
`a + b*merged` fit calls `a`, and it is where a reader's first instinct -- "the difference must
be communication" -- is usually wrong. The manual's own open question is what that gap is made
of; this script answers it from a trace you already have, with no device and no profiler.

METHOD (three steps, all cheap)
-------------------------------
  1. Build `merged` = |union(kernel U hpu_op Dma*)| -- the manual's occupancy figure.
  2. Take the COMPLEMENT of `merged` inside the step window. That is the set of holes.
  3. Histogram the holes BY WIDTH. The width distribution names the cause:

       < 50 us, tens of thousands of them   inter-descriptor turnaround. Tracks OP COUNT.
                                            This is launch granularity, not communication --
                                            and it is the part of the gap you can shrink by
                                            fusing ops, not by touching HCCL.
       50 us - 1 ms, hundreds               host-side stalls between submissions.
       > 1 ms, a handful                    dependency / host / collective stalls. THE ONLY
                                            bucket in which "communication" can hide, and only
                                            if the holes are ALIGNED ACROSS RANKS.

  Pass two or more traces (one per rank) and the script tests the > 200 us holes for rank
  alignment -- in START and in WIDTH, across the ranks that take part. A per-layer collective
  shows ~one aligned hole per layer (40+ per step); a few unaligned multi-ms holes are host or
  dependency stalls. That distinction is the whole point: an unaligned multi-ms hole is not a
  collective no matter how much it looks like one.

  Every alignment rate is printed beside a NULL -- the same statistic computed on holes drawn
  uniformly at the same density. This is not decoration: with ~20,000 holes per step a +/-50 us
  tolerance matches almost anything by chance, so a rate without its null is not a measurement.
  `--selftest` is the control that proves the test can see alignment when it is there.

WHAT A TRACE CANNOT TELL YOU
----------------------------
* "No engine in flight" is NOT "waiting on the network". If the capture carries no NIC/HCCL
  engine rows, communication is invisible to it and the gap cannot be attributed either way.
* Marker durations are not execution windows (see trace_engines.py, and the manual's ch.1 §1.8).
  This script uses the runtime's own `HW event name` field, which is the defensible label.
* `merged` is an OCCUPANCY, not summed work. A family's union is not what removing it frees --
  that is its leave-one-out (manual §3.6.4).

USAGE
-----
    python gap_histogram.py <trace.json.gz>                      # one rank
    python gap_histogram.py <rank0.gz> <rank1.gz> <rank2.gz> ... # alignment check
    python gap_histogram.py <trace-dir>                          # largest .gz in the dir

Step count follows the manual's convention: the number of `python_function` events whose name
ends in `_execute_model_generic`, or 1 if the trace carries none. That is what makes the
per-step figures here comparable with the manual's own.
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import sys

# Synapse's own per-event engine labels -- the authoritative field. Same map as trace_engines.py;
# kept here so this example runs standalone.
MARKER_TO_ENGINE = {
    "TPC_SPU_START_TO_SPU_HALT": "TPC",
    "MMEH_WB1_MON_TS_BIT1": "MME",
    "MMEH_WB0_MON_TS_BIT1": "MME",
    "DBG_DMA_TRC_WR_DATA_LAST": "DMA",
    "DBG_DMA_TRC_WR_FRST_ADDR_PUSH": "DMA",
    "DBG_DMA_TRC_RD_FRST_ADDR_PUSH": "DMA",
}

WIDTHS = (("<50us", 0, 50), ("50-200us", 50, 200), ("200us-1ms", 200, 1000), (">1ms", 1000, float("inf")))
ALIGN_TOL_US = 50.0          # two holes on different ranks are "aligned" within this
ALIGN_MIN_US = 200.0         # only holes at least this wide are tested for alignment


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path) as fh:
        return json.load(fh)["traceEvents"]


def pick(path):
    """Accept a file or a directory.

    A directory yields ONE trace (the largest .gz) and says so -- a directory is a convenience for
    the single-rank case, not a way to pick rank 0. For a multi-rank census, pass the files.
    """
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.gz")), key=os.path.getsize)
        if not cands:
            raise SystemExit(f"no .gz trace under {path}")
        print(f"  [dir] {len(cands)} traces present; using the largest only -- pass files for all")
        return cands[-1]
    return path


def engine_of(event):
    marker = (event.get("args") or {}).get("HW event name")
    return MARKER_TO_ENGINE.get(marker)


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def complement(merged, lo, hi):
    """The holes: maximal intervals inside [lo, hi] that `merged` does not cover."""
    holes = []
    cursor = lo
    for s, e in merged:
        if s > cursor:
            holes.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < hi:
        holes.append((cursor, hi))
    return holes


def census(holes):
    rows = []
    for name, lo, hi in WIDTHS:
        sel = [e - s for s, e in holes if lo <= (e - s) < hi]
        rows.append((name, len(sel), sum(sel) / 1000.0, (max(sel) / 1000.0 if sel else 0.0)))
    return rows


def analyse(path):
    ev = load(path)
    steps = sum(1 for e in ev
                if e.get("cat") == "python_function" and e.get("ph") == "X"
                and str(e.get("name", "")).endswith("_execute_model_generic")) or 1

    feed = [e for e in ev
            if e.get("ph") == "X" and e.get("dur")
            and (e.get("cat") == "kernel"
                 or (e.get("cat") == "hpu_op" and str(e.get("name", "")).startswith("Dma")))]
    if not feed:
        raise SystemExit(f"{path}: no kernel/hpu_op-Dma events -- is this a device trace?")

    merged = merge([(e["ts"], e["ts"] + e["dur"]) for e in feed])
    lo, hi = merged[0][0], merged[-1][1]
    union_us = sum(e - s for s, e in merged)

    per_engine = collections.defaultdict(list)
    for e in feed:
        eng = engine_of(e)
        if eng:
            per_engine[eng].append((e["ts"], e["ts"] + e["dur"]))

    holes = complement(merged, lo, hi)
    print(f"===== {os.path.basename(path)}")
    print(f"  steps (python_function *_execute_model_generic) : {steps}")
    print(f"  window                        {((hi - lo) / 1000):9.2f} ms   ({((hi - lo) / 1000) / steps:.2f} ms/step)")
    print(f"  merged (union)                {union_us / 1000:9.2f} ms   ({union_us / 1000 / steps:.4f} ms/step)")
    print(f"  gap (window - merged)         {(hi - lo - union_us) / 1000:9.2f} ms   "
          f"({(hi - lo - union_us) / 1000 / steps:.4f} ms/step)")
    for eng in ("TPC", "MME", "DMA"):
        u = sum(e - s for s, e in merge(per_engine[eng]))
        print(f"    {eng:3s} union                   {u / 1000:9.2f} ms   ({u / 1000 / steps:8.4f} ms/step, "
              f"{u / union_us * 100:5.1f}% of merged)")
    print(f"  holes: {len(holes)} runs, {(hi - lo - union_us) / 1000:.2f} ms total")
    for name, n, tot, mx in census(holes):
        print(f"    {name:11s} {n:8d} runs {tot:9.2f} ms   max {mx:8.2f} ms")
    return holes, lo


def _matched(base_holes, others, tol_us):
    """How many base holes have a same-width partner on EVERY other rank.

    Inputs are (start_us, width_us) pairs -- NOT (start, end).
    """
    hit = 0
    for s, w in base_holes:
        ok = True
        for starts in others.values():
            if not any(abs(s2 - s) <= tol_us and abs(w2 - w) <= max(tol_us, 0.10 * w)
                       for s2, w2 in starts):
                ok = False
                break
        if ok:
            hit += 1
    return hit


def _match_rate(base_holes, others, tol_us):
    return _matched(base_holes, others, tol_us) / len(base_holes) if base_holes else 0.0


def alignment(holes_by_rank, trials=200, seed=0):
    """How many of the wide holes are rank-aligned -- against a NULL, which is the point.

    With ~20k holes per step, a +/-50 us tolerance matches almost anything by chance. So the
    observed rate is meaningless on its own: it is reported against the rate obtained from
    holes drawn uniformly at the same density. Only a rate far above the null is evidence of
    alignment, and only alignment supports the collective reading.
    """
    import random

    wide = {r: [(s, e - s) for s, e in h if (e - s) >= ALIGN_MIN_US]
            for r, h in holes_by_rank.items()}
    ranks = sorted(wide)
    if len(ranks) < 2:
        return None
    base = ranks[0]
    base_holes = wide[base]
    if not base_holes:
        print(f"\n  no holes >= {ALIGN_MIN_US:.0f} us on {base} -- nothing to align")
        return None
    others = {r: wide[r] for r in ranks[1:]}
    n_obs = _matched(base_holes, others, ALIGN_TOL_US)
    obs = n_obs / len(base_holes)

    lo = min(s for h in holes_by_rank.values() for s, _ in h)
    hi = max(e for h in holes_by_rank.values() for _, e in h)
    rng = random.Random(seed)

    def null_rate(keys, trials_n):
        vals = []
        for _ in range(trials_n):
            fake = {r: [(rng.uniform(lo, hi), rng.uniform(ALIGN_MIN_US, 1000.0))
                        for _ in others[r]] for r in keys}
            vals.append(_match_rate(base_holes, fake, ALIGN_TOL_US))
        return sum(vals) / len(vals)

    null = null_rate(list(others), trials)

    print(f"\n  rank alignment of holes >= {ALIGN_MIN_US:.0f} us "
          f"(start tol {ALIGN_TOL_US:.0f} us, width tol max({ALIGN_TOL_US:.0f} us, 10%))")
    print(f"    base rank {base}: {len(base_holes)} wide holes")
    print(f"    pairwise, base hole matched in ONE other rank      observed / null")
    pw_obs, pw_null = [], []
    for r in ranks[1:]:
        one = _match_rate(base_holes, {r: others[r]}, ALIGN_TOL_US)
        one_null = null_rate([r], trials)
        pw_obs.append(one)
        pw_null.append(one_null)
        print(f"      vs {r[:22]:22s}  {one * 100:5.1f}% / {one_null * 100:5.1f}%")
    mean_obs = sum(pw_obs) / len(pw_obs)
    mean_null = sum(pw_null) / len(pw_null)
    print(f"    aligned on ALL {len(ranks)} ranks: {n_obs} of {len(base_holes)} holes "
          f"({obs * 100:5.1f}%), null {null * 100:5.1f}%")
    if obs > 0.10 and obs > 5 * max(null, 1e-9):
        print("    READ: aligned across ALL ranks -> genuinely rank-aligned. Check the width")
        print("          pattern: ~one aligned hole per LAYER (40+ per step) is a per-layer")
        print("          collective or barrier; a handful of aligned multi-ms holes is a global stall.")
    elif mean_obs > 5 * max(mean_null, 1e-9) and mean_obs > 0.05:
        print(f"    READ: PARTIALLY aligned -- pairwise {mean_obs * 100:.1f}% against a "
              f"{mean_null * 100:.1f}% null, but {obs * 100:.1f}% across all {len(ranks)} ranks.")
        print("          Some ranks share these holes and all four never do, so this is NOT a clean")
        print("          all-rank barrier. Look at WHICH pairs align before calling it communication:")
        print("          a real collective on the critical path is aligned on every rank it involves.")
    else:
        print("    READ: at or near the null -> NOT evidence of alignment, and therefore NOT")
        print("          evidence of communication. A few unaligned multi-ms holes are host or")
        print("          dependency stalls.")
    return {"n_base": len(base_holes), "n_aligned": n_obs, "rate": obs, "null": null}


def selftest():
    """A negative result is only worth something if the test can produce a positive one.

    Two synthetic populations at roughly the real trace's density (about 150 wide holes over
    a 450 ms window, on each of 4 ranks):
      A) 12 genuinely aligned 400 us holes + independent noise -> must FIND the 12
      B) independent noise only                                -> must find ~0
    If (A) does not find them, the test has no power and a 0% reading from it means nothing.
    This self-test earned its place: the first version of this script double-subtracted the
    start from the width and reported 0% aligned on BOTH populations.
    """
    import contextlib
    import io
    import random

    rng = random.Random(7)
    window = 450_000.0

    def noise(n):
        out = []
        for _ in range(n):
            s = rng.uniform(0, window)
            out.append((s, s + rng.uniform(50.0, 5_000.0)))   # (start, END)
        return out

    def population(with_signal):
        holes = {}
        aligned = [(50_000.0 + 3_000.0 * i, 400.0) for i in range(12)]   # (start, width)
        for r in range(4):
            h = noise(150)
            if with_signal:
                h += [(s + rng.uniform(-3, 3), s + rng.uniform(-3, 3) + w) for s, w in aligned]
            holes[f"rank{r}"] = h
        return holes

    def stats(holes):
        with contextlib.redirect_stdout(io.StringIO()):
            return alignment(holes, trials=40)

    a = stats(population(True))
    b = stats(population(False))
    print(f"selftest: 12 aligned holes present -> found {a['n_aligned']} of {a['n_base']} "
          f"({a['rate'] * 100:.1f}%, want >= 11 of 12)")
    print(f"selftest: noise only              -> found {b['n_aligned']} of {b['n_base']} "
          f"({b['rate'] * 100:.1f}%, want <= 2)")
    ok = a["n_aligned"] >= 11 and b["n_aligned"] <= 2
    print("selftest: " + ("PASS -- the test can see alignment when it is there, so a 0% "
                         "reading is informative" if ok else
                         "FAIL -- do not trust a 0% reading from this test"))
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv[1:]:
        return selftest()
    paths = [a for a in argv[1:] if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2
    holes_by_rank = {}
    for p in paths:
        resolved = pick(p)
        holes, _ = analyse(resolved)
        holes_by_rank[os.path.basename(resolved)] = holes
        print()
    alignment(holes_by_rank)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
