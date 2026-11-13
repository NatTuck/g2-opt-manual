"""Check that the kernel GUID becomes a device event, and read the split off it.

The question this answers is §9.4's: does Synapse resolve the torch op to the
TPC kernel we shipped, and how would you know?  The answer is that the device
event carries the GUID as its name, so it is readable in a trace.

Also prints the per-TPC census of one launch -- how many TPCs the node got --
which 04-execution-model.md §4.2.8 says to read rather than assume.

Usage:  GC_KERNEL_PATH=<...>/libadd1_kernels.so PT_HPU_LAZY_MODE=0 \
            python3 check_guid_in_trace.py
"""
import json
import os

import torch
import habana_frameworks.torch.core  # noqa: F401

from custom_add1 import CustomAdd1

GUID = "add1_fwd_f32_gaudi2"
HERE = os.path.dirname(os.path.realpath(__file__))
TRACE = os.path.join(HERE, "build", "add1_trace.json")
LAUNCHES = 5
ELEMENTS = 1024 * 1024


def main():
    os.makedirs(os.path.dirname(TRACE), exist_ok=True)

    op = CustomAdd1()
    x = torch.full((1024, 1024), 5.0, dtype=torch.float32).to("hpu")
    op(x)
    torch.hpu.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.HPU,
        ],
    ) as prof:
        for _ in range(LAUNCHES):
            op(x)
        torch.hpu.synchronize()

    # On this stack torch.profiler's public export is monkey-patched by the HPU
    # plugin with a signature that no longer matches, so go through the inner
    # object.  Measured: the public call raises TypeError.
    prof.profiler.export_chrome_trace(TRACE)

    with open(TRACE) as f:
        events = json.load(f)["traceEvents"]
    hits = [e for e in events if e.get("name") == GUID]
    print(f"trace events: {len(events)}   events named {GUID!r}: {len(hits)}")
    if not hits:
        raise SystemExit("FAIL: the GUID does not appear in the trace")

    e = hits[0]
    print(f"  event name      {e['name']}")
    print(f"  category        {e['cat']}")
    print(f"  EventName       {e['args']['EventName']}")
    print(f"  HW event name   {e['args']['HW event name']}")
    print(f"  dataType        {e['args']['dataType']}")
    print(f"  Device          {e['args']['Device']}")

    tids = sorted({e["tid"] for e in hits})
    print(f"  TPCs (distinct tids) {len(tids)}  over {LAUNCHES} launches "
          f"= {len(hits)} events")

    by_recipe = {}
    for e in hits:
        by_recipe.setdefault(e["args"]["recipeId"], []).append(e)
    print(f"  distinct recipeId per launch: {len(by_recipe)}")

    spans = []
    for rid, es in by_recipe.items():
        start = min(e["ts"] for e in es)
        end = max(e["ts"] + e["dur"] for e in es)
        spans.append(end - start)
    mean_span = sum(spans) / len(spans)
    print(f"  span per launch (us): {[round(s, 3) for s in spans]}")
    print(f"  mean span: {mean_span:.3f} us")

    traffic = ELEMENTS * 4 * 2  # in + out, float32
    print(f"  traffic: {traffic / 1e6:.2f} MB over {mean_span:.3f} us "
          f"= {traffic / (mean_span * 1e-6) / 1e12:.2f} TB/s")


if __name__ == "__main__":
    main()
