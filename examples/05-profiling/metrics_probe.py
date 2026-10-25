#!/usr/bin/env python3
"""Does the GC compile metric carry a key that ties a RECIPE to a dynamo COMPILE ID?

If it does, the region->recipe leg becomes a key join at compile time and the runtime launch
pairing (which fails on a real model) is unnecessary.
"""
import os, json
os.environ["PT_HPU_METRICS_GC_DETAILS"] = "1"
os.environ["PT_HPU_ENABLE_CACHE_METRICS"] = "1"
os.environ["PT_HPU_METRICS_FILE"] = "/tmp/recipe-chain/metrics.json"
os.environ["TORCH_TRACE"] = "/tmp/recipe-chain/spec"

import torch, habana_frameworks.torch  # noqa
from habana_frameworks.torch.hpu.metrics import metric_global

@torch.compile(backend="hpu_backend", dynamic=False, fullgraph=True)
def blk_a(x, w): return torch.matmul(torch.matmul(x, w), w)

@torch.compile(backend="hpu_backend", dynamic=False, fullgraph=True)
def blk_b(x):
    y = torch.relu(x); y = torch.exp(y) * 0.5; return torch.add(y, 1.0)

x = torch.randn(512, 1024, dtype=torch.bfloat16, device="hpu")
w = torch.randn(1024, 1024, dtype=torch.bfloat16, device="hpu")
for _ in range(3):
    blk_a(x, w); blk_b(x)
torch.hpu.synchronize()

for name in ("graph_compilation", "recipe_cache"):
    m = metric_global(name)
    print(f"\n=== {name}  ({type(m).__name__})")
    if m is None:
        print("   metric is None (unavailable)")
        continue
    st = m.stats()
    # stats() returns a LIST of (key, value) pairs, not a dict -- handling it as a dict was a
    # TypeError in the first version of this probe.
    st = dict(st) if not isinstance(st, dict) else st
    print("   stats keys:", sorted(st))
    for k, v in st.items():
        if isinstance(v, list):
            print(f"   {k}: {len(v)} entries")
            for item in v[:6]:
                print("      ", item)
        else:
            print(f"   {k}: {v}")
