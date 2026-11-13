#!/usr/bin/env python3
"""Rung 1 — one compiled region, one tensor size, one add.

The question this rung answers is the smallest version of the whole problem:
**a torch.compile region over a fixed-shape 1-D tensor should become exactly one
recipe.** Everything else in the stack (dynamo tracing, the bridge's placement
decisions, Synapse's recipe key) is visible in what this produces.

What to look at in the output:
  * the FX graph dynamo produced — that is "the region";
  * the recipe count in the cache directory — 1 is the expectation;
  * the post-graph dumps — node names and their placement (`hpu_cluster`/`eager`);
  * first-call vs second-call wall time — tracing+compile vs dispatch.

Run it with experiments/run.sh, which sets the cache contract and reports counts.
"""
import time

import torch
import torch_hpu

N = 1024  # the only tensor size at this rung


def add_one(t):
    return t + 1


def main() -> None:
    x = torch.arange(N, dtype=torch.float32, device="hpu")

    # The HPU backend is registered by the plugin but is not the default; without
    # this argument torch.compile picks inductor and dies with KeyError: 'hpu'
    # (measured). Name it explicitly.
    compiled = torch.compile(add_one, backend="hpu_backend")

    t0 = time.perf_counter()
    y = compiled(x)
    torch.hpu.synchronize()
    t1 = time.perf_counter()

    y2 = compiled(x)
    torch.hpu.synchronize()
    t2 = time.perf_counter()

    print(f"  N={N}  y[0]={y[0].item()}  y[-1]={y[-1].item()}  equal={bool((y == y2).all())}")
    print(f"  first call  {1000 * (t1 - t0):8.1f} ms   (trace + bridge + compile)")
    print(f"  second call {1000 * (t2 - t1):8.1f} ms   (dispatch only)")

    # What dynamo thinks the region is.
    # `explain().graphs` holds GraphModules, so each FX graph is gm.graph.
    explanation = torch._dynamo.explain(add_one)(x)
    print(f"  dynamo: graph_count={explanation.graph_count} "
          f"break_count={explanation.graph_break_count} "
          f"op_count={explanation.op_count}")
    for i, gm in enumerate(explanation.graphs):
        print(f"  FX graph {i}:")
        print("\n".join("    " + line for line in str(gm.graph).splitlines()))


if __name__ == "__main__":
    main()
