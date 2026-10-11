#!/usr/bin/env python3
"""trace_engines.py -- attribute device events to engines, and check your assumptions.

Two things this does that a name-based classification does NOT:

  1. Uses `args["HW event name"]` -- the RUNTIME'S OWN per-event engine label. It is present on
     ~98.85% of device events and agrees with the tid windows on 100% of device work. A name
     heuristic ("anything starting with Dma is DMA") is an OP TAXONOMY, not an engine
     classifier: it abstains on exactly the ops you are experimenting with.

  2. Answers the question "did my copy actually go to the DMA engine?" -- see TRAP below.

TRAP: `torch.copy_` IS NOT THE DMA ENGINE
-----------------------------------------
In EAGER mode, `y.copy_(x)` lowers to a TPC kernel (`memcpy_u8`), not to the DMA engine.
Measured: 768 TPC kernel events, ZERO DMA-engine events. Under `torch.compile` the same op DOES
go to the engine.

This matters because a row reading "D2D copy (DMA engine) | 1014/1704/1888 GB/s" sat in two of
our notes on the strength of its label. Those are TPC numbers. Any "DMA engine" bandwidth figure
taken from `torch.copy_` in eager mode is TPC data, and the error is invisible from the label.

Run `check_copy_routing()` below to see which engine your copy actually used.

USAGE
-----
    python trace_engines.py /path/to/trace.pt.trace.json.gz
    python trace_engines.py --selftest        # run the copy-routing check on the device
"""

import collections
import gzip
import json
import sys

# Synapse's own per-event engine labels. This is the authoritative field.
MARKER_TO_ENGINE = {
    "TPC_SPU_START_TO_SPU_HALT":       "TPC",
    "MMEH_WB1_MON_TS_BIT1":            "MME",
    "MMEH_WB0_MON_TS_BIT1":            "MME",
    "DBG_DMA_TRC_WR_DATA_LAST":        "DMA",
    "DBG_DMA_TRC_WR_FRST_ADDR_PUSH":   "DMA",
    "DBG_DMA_TRC_RD_FRST_ADDR_PUSH":   "DMA",
}

# Tid windows: a CORROBORATING check, not the ground truth. They are stable across every trace
# we have, but the DMA window in particular is right by accident -- its upper bound truncates a
# stride-10 channel series one channel before tids that are already in use.
TID_WINDOWS = {
    "TPC": range(7020, 7047),
    "MME": range(6050, 6081),
    "DMA": range(4050, 4076),
}


def engine_of(event):
    """Primary: the runtime's own marker. Fallback: the tid window."""
    args = event.get("args", {}) or {}
    marker = args.get("HW event name")
    if marker in MARKER_TO_ENGINE:
        return MARKER_TO_ENGINE[marker], "marker"
    tid = event.get("tid")
    for engine, window in TID_WINDOWS.items():
        if tid in window:
            return engine, "tid"
    return None, "unattributed"


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path) as fh:
        return json.load(fh)["traceEvents"]


def report(path):
    events = [e for e in load_events(path) if e.get("cat") in ("kernel", "hpu_op")]
    by_engine = collections.Counter()
    by_source = collections.Counter()
    marker_coverage = 0
    host = 0
    for e in events:
        eng, src = engine_of(e)
        by_source[src] += 1
        if eng is None:
            # The only events with no marker at all are the host spans
            # (synchronizeEvent, enqueueWithExternalEventsExt, compileGraph, ...).
            # "no engine" is the CORRECT answer for them.
            host += 1
            continue
        by_engine[eng] += 1
        if src == "marker":
            marker_coverage += 1

    total = len(events)
    print(f"{path}\n  device events: {total}")
    print(f"  engine attribution: {dict(by_engine)}")
    print(f"  attribution source: {dict(by_source)}")
    if total:
        print(f"  marker coverage: {marker_coverage / total * 100:.2f}% "
              f"(the remainder are host spans -> correctly 'no engine')")
    print(f"  host spans (no engine): {host}")
    print("""
  HOW TO READ THIS
  ----------------
  * TPC / MME / DMA counts here are EVENT counts. A TPC kernel launch records one event PER
    TPC, so a device event is a REPLICATION count, not a launch count. Do not call it "launches".
  * If you expected DMA traffic and see none, check the TRAP at the top of this file: your
    copy probably lowered to TPC `memcpy_u8`.
  * Build durations from UNIONS, never sums. Summed device durations are inflated by the
    per-TPC replication factor (measured 18-24x for TPC kernels).""")


def check_copy_routing():
    """Did `torch.copy_` go to the DMA engine? Run this once and you will never assume again."""
    import torch
    import habana_frameworks.torch  # noqa: F401

    n = 64 * 1024 * 1024 // 4          # 64 MB of fp32
    x = torch.randn(n, device="hpu")
    y = torch.empty_like(x)
    for _ in range(3):                  # warm/compile before profiling
        y.copy_(x)
    torch.hpu.synchronize()

    import collections as _c
    import glob as _g
    import gzip as _gz
    import json as _j
    import os as _os
    import shutil as _sh
    from torch.profiler import (ProfilerActivity, profile,   # noqa: E402
                                tensorboard_trace_handler)

    out = "/tmp/copy-routing-trace"
    _sh.rmtree(out, ignore_errors=True)
    _os.makedirs(out)
    # HPU-only: including CPU activity inflates the host span and makes a device-bound arm
    # look host-bound.
    with profile(activities=[ProfilerActivity.HPU],
                 on_trace_ready=tensorboard_trace_handler(out, use_gzip=True)):
        for _ in range(5):
            y.copy_(x)
        torch.hpu.synchronize()

    for p in _g.glob(_os.path.join(out, "*.gz")):
        with _gz.open(p) as fh:
            evs = _j.load(fh)["traceEvents"]
        c = _c.Counter()
        for e in evs:
            if e.get("cat") in ("kernel", "hpu_op"):
                eng, _ = engine_of(e)
                c[eng or "host"] += 1
        print(f"\neager y.copy_(x) -> engines: {dict(c)}")
        if c.get("DMA"):
            print("  -> went to the DMA engine (this happens under torch.compile)")
        else:
            print("  -> did NOT use the DMA engine. In EAGER mode copy_ lowers to a TPC kernel.")
            print("     Any 'DMA engine bandwidth' figure measured this way is TPC data.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        check_copy_routing()
    elif len(sys.argv) > 1:
        report(sys.argv[1])
    else:
        print(__doc__)
        sys.exit(2)
