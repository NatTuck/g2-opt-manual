#!/usr/bin/env python3
"""devprobe.py -- PRE-FLIGHT every device trace before you trust it.

WHY THIS EXISTS
---------------
Tracing device events on Gaudi needs a host-profiler shim, loaded from the path in
`HABANA_PLUGINS_LIB_PATH`. If that path is stale -- it points somewhere the image no longer
ships the plugin -- then `synProfilerStart` FAILS SILENTLY. You still get a trace. It still
parses. It contains `cpu_op` and `privateuse1_runtime` events and **no `kernel` and no `hpu_op`
events at all**.

The failure does not look like a failure. It looks like "the device did nothing", which is a
conclusion someone will believe and act on. This script is the 20-line guard against that: it
runs a known small matmul under the profiler and prints a histogram of event categories. If you
do not see `kernel` and `hpu_op`, the shim is wrong and nothing downstream is worth reading.

USAGE
-----
    # point at the path the image ACTUALLY ships (commonly /opt/habanalabs/habana_plugins)
    export HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins
    python devprobe.py

EXPECTED OUTPUT (the thing to check)
------------------------------------
    trace.json.gz [('kernel', 15), ('hpu_op', 12), ('cpu_op', ...), ('python_function', ...)]

`kernel` and `hpu_op` MUST both be present. If they are absent -> the plugin path is wrong.
"""

import collections
import glob
import gzip
import json
import os
import shutil

import torch
import habana_frameworks.torch  # noqa: F401  (must be imported before touching hpu tensors)

OUT = "/tmp/devprobe-trace"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

# A tiny, known workload. Compile/warm it first so the profiled window is steady state --
# otherwise the trace's first events are compilation, not execution.
a = torch.randn(1024, 1024, device="hpu")
b = torch.randn(1024, 1024, device="hpu")
for _ in range(3):
    a @ b
torch.hpu.synchronize()

from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler  # noqa: E402

# NOTE: CPU activity is deliberately included here so the histogram shows what a "no device
# events" trace looks like. When you profile a DEVICE question, profile HPU-only -- collecting
# CPU activity alongside HPU inflates the trace's host span by ~3x, which makes a device-bound
# arm look host-bound.
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.HPU],
             on_trace_ready=tensorboard_trace_handler(OUT, use_gzip=True)):
    for _ in range(5):
        a @ b
    torch.hpu.synchronize()

ok = False
for path in glob.glob(os.path.join(OUT, "*.gz")):
    with gzip.open(path) as fh:
        events = json.load(fh)["traceEvents"]
    cats = collections.Counter(e.get("cat") for e in events)
    print(os.path.basename(path), cats.most_common(6))
    if cats.get("kernel") and cats.get("hpu_op"):
        ok = True

if ok:
    print("\nPASS -- device events present. The shim is loaded; the trace is trustworthy.")
else:
    print("\nFAIL -- NO kernel/hpu_op events. The profiler shim did not load.")
    print("  check HABANA_PLUGINS_LIB_PATH points at the path the image actually ships")
    print("  (commonly /opt/habanalabs/habana_plugins, also set in /etc/profile.d/habanalabs.sh).")
    print("  A stale path such as /usr/lib/habanatools/habana_plugins fails SILENTLY.")
    raise SystemExit(1)
