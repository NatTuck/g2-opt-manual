#!/usr/bin/env python3
"""The correctness gate for a span that reads state — manual §6.6.2.

Runs on CPU. No HPU and no device required: dynamo is a host-side tracer, so every
result here reproduces on any machine with torch installed.

WHAT THIS DEMONSTRATES
----------------------
A Python-level container consulted *inside* a traced region is a source of CONSTANTS,
not of values. A dict entry written on one call and read on a later call inside a
torch.compile'd region is baked into the graph. The symptom is not a crash — it is a
plausible wrong number.

The case that matters is NOT "two calls, different data". It is **two uses of the same
cached helper inside ONE traced region**: the second use reads what the first wrote,
during the same trace. That is the case a per-region cache clear does not fix.

THE THREE VARIANTS
------------------
  none   no clear at all            -> input ignored ENTIRELY, across calls
  outer  clear at top of the REGION -> cross-call staleness fixed, IN-TRACE reuse survives
  inner  clear at top of each USE   -> correct

Run it:

    ./venv/bin/python examples/06-workflow/traced_state_gate.py            # report
    ./venv/bin/python examples/06-workflow/traced_state_gate.py --selftest # the gate

`--selftest` is the point of the file. It asserts that the gate DETECTS the buggy
constructs. A gate that passes on the buggy code is not a gate — it is a gate that has
never been shown to fail, which is indistinguishable from one that works.

WHY THIS IS A REQUIRED CASE CLASS
---------------------------------
A bit-exact gate on self-consistent inputs cannot find this. Both sides of the
comparison are self-consistent: the reference and the rewrite agree, and both are
wrong. The only case that finds it is SAME SHAPE, DIFFERENT DATA — compile once, call
twice with different values, compare each against a reference that cannot bake.
"""

import argparse
import sys

import torch


# --------------------------------------------------------------------------------------
# the reference: no cache, no compile. This is the only thing in the file that is right
# by construction, so every variant is judged against it.
# --------------------------------------------------------------------------------------
def reference(x):
    """(x+1)*2 + (x+1000+1)*2, computed with no state at all."""
    return (x + 1) * 2 + (x + 1000.0 + 1) * 2


def build(clear_where, dynamic=True):
    """A compiled region calling one cached helper TWICE.

    Both calls are inside a single traced region, so the second is an in-trace hit.
    """
    CACHE = {}

    def inner(x):
        if clear_where == "inner":
            CACHE.clear()          # per-USE clear
        hit = CACHE.get("k")
        if hit is not None:
            return hit * 2         # <- on a HIT this is a baked constant
        CACHE["k"] = x + 1
        return CACHE["k"] * 2

    def outer(x):
        if clear_where == "outer":
            CACHE.clear()          # per-REGION clear
        return inner(x) + inner(x + 1000.0)

    return torch.compile(outer, backend="eager", dynamic=dynamic, fullgraph=True)


def probe(clear_where):
    """Call the compiled region twice with different data, same shape."""
    fn = build(clear_where)
    return [fn(torch.full((3,), v))[0].item() for v in (10.0, 20.0)]


def single_call_hit():
    """The minimal form: a hit across two calls bakes the value and ignores the input."""
    CACHE = {}

    def cached(x):
        hit = CACHE.get("k")
        if hit is not None:
            return hit * 2
        CACHE["k"] = x + 1
        return CACHE["k"] * 2

    fn = torch.compile(cached, backend="eager", dynamic=True, fullgraph=True)
    fn(torch.zeros(3))                       # trace / populate
    fn(torch.zeros(3))                       # hit
    return fn(torch.full((3,), 100.0))[0].item(), (100.0 + 1) * 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="assert the gate DETECTS the buggy constructs (the control)")
    args = ap.parse_args()

    expect = [reference(10.0), reference(20.0)]
    got = {w: probe(w) for w in ("none", "outer", "inner")}

    print(f"reference                : {expect}")
    for w in ("none", "outer", "inner"):
        verdict = "CORRECT" if got[w] == expect else "WRONG  "
        print(f"clear at {w:6}          : {got[w]}  {verdict}")

    baked, should_be = single_call_hit()
    print(f"\nsingle-call cache hit    : {baked}   (input 100.0 wants {should_be})")

    if not args.selftest:
        print("\n(use --selftest to run this as a gate)")
        return 0

    # ---------------------------------------------------------------- the gate
    failures = []

    # 1. The CORRECT variant must match the reference. If this fails the harness is broken.
    if got["inner"] != expect:
        failures.append(f"correct variant 'inner' gave {got['inner']}, expected {expect}")

    # 2. The gate must DETECT each buggy variant. This is the control that proves the
    #    gate can fail — without it, a green run means nothing.
    if got["none"] == expect:
        failures.append("gate failed to detect 'none' — it returned the reference value")
    if got["outer"] == expect:
        failures.append("gate failed to detect 'outer' — a per-region clear looked safe")

    # 3. The baked hit must be observably wrong, not merely different.
    if baked == should_be:
        failures.append(f"cache hit was not baked (got {baked}) — cannot reproduce the trap")

    # 4. And the whole point: the bug must survive 'more self-consistent cases'. The
    #    same shape at a different VALUE is what exposes it; repeating one value does not.
    if got["none"][0] == got["none"][1]:
        print("\nnote: the buggy arm returned the same value for two different inputs —")
        print("      that is the signature, and it is why a bigger matrix of")
        print("      self-consistent inputs cannot substitute for this case.")

    if failures:
        print("\nSELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nSELFTEST PASSED — the gate detects both buggy constructs and the trap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
