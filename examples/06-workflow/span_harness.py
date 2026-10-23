#!/usr/bin/env python3
"""span_harness.py -- the template for a span harness (chapter 6, step 3b onward).

The pattern, in the order the chapter insists on:

  1. load CAPTURED BYTES (never a seed) for the span's inputs, outputs and state;
  2. run the REAL span function -- imported from the model, not reimplemented here;
  3. run the correctness gate FIRST and refuse to report a time if it fails;
  4. time it, and report ops / launches / device-busy alongside the time, because a win that comes
     from fewer launches does not transfer while a win that removes device work does;
  5. print what the proxy check needs, so the harness can be compared against the in-situ profile.

This file is a TEMPLATE. Point `--span` at the model's own function and `--capture` at a file your
instrumented run wrote; it refuses to guess either.

Usage:
    span_harness.py --capture cap/span-0.pt --span my_model.moe:patched_fused_moe_forward
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time

# Set by --require-state; read by load_capture. See chapter 6 §6.5.1.
REQUIRE_STATE = False


def load_span(spec):
    """'module.path:function' -> the model's OWN function. Never a copy."""
    if ":" not in spec:
        raise SystemExit(f"--span must be 'module:function', got {spec!r}")
    mod_name, fn_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise SystemExit(f"{mod_name} has no attribute {fn_name}")
    return fn


def load_capture(path):
    """Captured bytes: inputs, expected outputs, and any state the span reads.

    If the capture is missing, stop -- a harness that invents its own inputs is measuring a
    different computation (chapter 6 §6.5).
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"no capture at {path}. Run the model with SPAN_CAPTURE=1 first (chapter 6 §6.5); "
            "a harness with synthetic inputs is not a proxy for the span.")
    import torch
    blob = torch.load(path, map_location="hpu", weights_only=False)
    for key in ("inputs", "outputs"):
        if key not in blob:
            raise SystemExit(f"capture {path} has no {key!r}")
    if REQUIRE_STATE and "state" not in blob:
        # A stateful span captured without its state is self-consistent and wrong: the timing is
        # unchanged (buffers are allocated at full size whatever they hold) and the gate passes
        # (reference and rewrite both see the same empty buffer). Chapter 6 §6.5.1.
        raise SystemExit(
            f"capture {path} has no 'state', but --require-state was given. Audit the capture "
            "against the recipe's own tensor table: the persistent, section-backed buffers are "
            "the ones you are missing (chapter 6 §6.5.1).")
    return blob


# --- the correctness gate ----------------------------------------------------------------------
def ulp_key(t):
    """Ordered-integer key for ulp distance.

    One key per FORMAT -- an ulp of bf16 is not an ulp of fp8, and fp8's encoding differs per
    flavour. Extend this for the dtypes your span uses rather than falling back to abs().
    """
    import torch
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16).to(torch.int32)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    raise SystemExit(f"no ulp key for dtype {t.dtype}; add one instead of using a tolerance")


def gate(actual, expected, tier, bound):
    """Returns (ok, report). A failing gate means: do not report a time."""
    import torch
    if len(actual) != len(expected):
        return False, {"error": f"{len(actual)} outputs vs {len(expected)} expected"}
    if tier == "bit-exact":
        bad = [(i, int((a != e).sum())) for i, (a, e) in enumerate(zip(actual, expected))
               if not torch.equal(a, e)]
        return (not bad), {"tier": "bit-exact", "mismatched_outputs": bad}
    if tier in ("ulp", "ulp-relaxed"):
        worst, hist = 0, []
        for a, e in zip(actual, expected):
            d = (ulp_key(a) - ulp_key(e)).abs()
            worst = max(worst, int(d.max()))
            hist.append({"max_ulp": int(d.max()), "mean_ulp": float(d.float().mean())})
        return (worst <= bound), {"tier": tier, "bound_ulp": bound, "worst_ulp": worst,
                                  "per_output": hist}
    raise SystemExit(f"unknown tier {tier!r}; declare bit-exact / ulp / ulp-relaxed")


# --- timing ------------------------------------------------------------------------------------
def run(fn, blob, iters, warmup):
    import torch
    args, state = blob["inputs"], blob.get("state")

    def call():
        return fn(*args) if state is None else fn(*args, state=state)

    for _ in range(warmup):
        call()
    torch.hpu.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = call()
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return out, times


def device_counters():
    """Compile/cache counters, cheap enough to read every iteration.

    Read the backend's own metrics rather than running a profiler: this loop is meant to be fast.
    Launch/submission counts are the ones that matter -- a win from fewer launches may not transfer
    (chapter 0), a win that removes device work does.
    """
    out = {}
    try:
        from habana_frameworks.torch.hpu.metrics import metric_global
        for name in ("graph_compilation", "recipe_cache"):
            m = metric_global(name)
            if m is not None:
                st = m.stats()
                out[name] = dict(st) if not isinstance(st, dict) else st
    except Exception as exc:  # metrics are optional; say so rather than pretend
        out["metrics"] = f"unavailable: {type(exc).__name__}"
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--span", required=True, help="'module:function' of the MODEL's own code")
    ap.add_argument("--tier", default="bit-exact", choices=["bit-exact", "ulp", "ulp-relaxed"])
    ap.add_argument("--bound", type=int, default=0, help="ulp bound; declare it BEFORE running")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--require-state", action="store_true",
                    help="fail if the capture has no 'state' (use for any span that reads a KV "
                         "cache, attention buffer, counter or router table)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv[1:])

    global REQUIRE_STATE
    REQUIRE_STATE = args.require_state
    fn = load_span(args.span)
    blob = load_capture(args.capture)

    actual, times = run(fn, blob, args.iters, args.warmup)

    ok, report = gate(actual, blob["outputs"], args.tier, args.bound)
    print(f"gate   {args.tier} bound={args.bound}  ->  {'PASS' if ok else 'FAIL'}")
    print(f"       {json.dumps(report)[:400]}")
    if not ok:
        print("REFUSING TO REPORT A TIME -- the harness is not computing the span.")
        return 1

    med = statistics.median(times)
    print(f"time   median {med:.3f} ms   min {min(times):.3f}   "
          f"p90 {sorted(times)[int(len(times) * 0.9)]:.3f}   (n={len(times)})")
    print(f"counters {json.dumps(device_counters())[:300]}")
    print("\nNext: compare recipe count / node mix / op counts against the in-situ dump "
          "(chapter 6 §6.7). A harness that does not match is measuring itself.")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"span": args.span, "median_ms": med, "times_ms": times,
                       "gate": report, "counters": device_counters()}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
