#!/usr/bin/env python3
"""recipe_chain_demo.py -- a deliberately artificial program with >= 3 recipes on all three
engines, built so that a profile of it can be walked back to specific source lines.

Why it is shaped this way
-------------------------
The attribution question is "which code produced this device time?". To answer it *verifiably*
rather than plausibly, each block below is:

  * its own ``torch.compile`` region  -> its own compile id, and therefore its own
    ``file:line:function`` identity through ``TORCH_TRACE``;
  * deliberately single-engine      -> MME, TPC and DMA each get a block whose expected node
    types are known in advance, so the chain can be checked against an independent expectation
    instead of being believed;
  * called from an UNCOMPILED driver -> the three regions stay separate. Compiling the driver
    would fuse them into one graph and destroy the experiment.

Identity knobs are set before ``torch`` is imported, because two of them are read once, early:
``TORCH_TRACE`` (dynamo writes the compile-id -> ``co_filename:co_firstlineno:co_name`` map as a
side effect of the run) and ``DUMP_POST_GRAPHS`` (the GC reads it once per process). A run that
omits either produces a trace you can time but cannot attribute.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.environ.get("CHAIN_ROOT", "/tmp/recipe-chain")
# "hpu"   -> HPU activity only. Correct for device timing; DROPS the host-side
#            `Torch-Compiled Region: <compile_id>` annotations, so the trace cannot be joined to
#            code. Measured: 0 region spans in that mode.
# "both"  -> CPU + HPU. Keeps the region annotations (and therefore the compile ids) but inflates
#            the host span ~3x, so do not quote timings from it.
ACTIVITIES = os.environ.get("CHAIN_ACTIVITIES", "both").lower()
SPEC = os.path.join(ROOT, "spec")        # TORCH_TRACE  -> compile id -> source line
DUMPS = os.path.join(ROOT, "dumps")      # DUMP_POST_GRAPHS -> per-recipe node tables
CACHE = os.path.join(ROOT, "recipes")    # recipe cache; a fresh dir => cold => graphs get built
PROF = os.path.join(ROOT, "prof")        # torch.profiler output

# --- identity knobs: BEFORE torch import -------------------------------------------------------
for d in (SPEC, DUMPS, CACHE, PROF):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

os.environ["TORCH_TRACE"] = SPEC
os.environ["DUMP_POST_GRAPHS"] = DUMPS
# <path>,<use_cache>,<size_mb>,<...>: caching off + a fresh dir, so every recipe cold-compiles and
# therefore gets dumped. A warm cache would load a blob and write no graph at all.
os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = f"{CACHE},false,8192,false"

import torch                                              # noqa: E402
import habana_frameworks.torch                            # noqa: F401,E402

DEV = "hpu"
DT = torch.bfloat16
M, K = 512, 1024


# --- the three blocks: one engine each ---------------------------------------------------------
def mme_block(x, w):
    """MME: two chained GEMMs, nothing elementwise."""
    return torch.matmul(torch.matmul(x, w), w)


def tpc_block(x):
    """TPC: an elementwise chain with no matrix multiply and no copy."""
    y = torch.relu(x)
    y = torch.exp(y) * 0.5
    y = torch.add(y, 1.0)
    return y


def dma_block(src, dst, scratch, row):
    """DMA: a device-to-device copy, a strided transpose materialisation, and a broadcast write."""
    dst.copy_(src)                      # memcpy_dma
    t = src.t().contiguous()            # transpose_dma
    scratch.copy_(t.t())                # memcpy_dma back
    dst.add_(row)                       # broadcast of a [K] row over [M, K]
    return dst


# backend="hpu_backend" is REQUIRED: the default backend is inductor, which does not know the
# 'hpu' device and fails with KeyError: 'hpu'. fullgraph=True keeps each block to exactly one
# region, which is what makes the region->code mapping unambiguous.
COMPILE = dict(backend="hpu_backend", dynamic=False, fullgraph=True)
mme_block = torch.compile(mme_block, **COMPILE)
tpc_block = torch.compile(tpc_block, **COMPILE)
dma_block = torch.compile(dma_block, **COMPILE)


def step(x, w, dst, scratch, row):
    """The driver is deliberately NOT compiled: three separate regions, not one fused graph."""
    a = mme_block(x, w)
    b = tpc_block(a)
    return dma_block(b, dst, scratch, row)


def main(iterations: int = 20, warmup: int = 5) -> int:
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=DT, device=DEV)
    w = torch.randn(K, K, dtype=DT, device=DEV)
    dst = torch.empty(M, K, dtype=DT, device=DEV)
    scratch = torch.empty(M, K, dtype=DT, device=DEV)
    row = torch.randn(K, dtype=DT, device=DEV)

    for _ in range(warmup):
        step(x, w, dst, scratch, row)
    torch.hpu.synchronize()

    import gzip

    from torch.profiler import ProfilerActivity, profile

    # NOTE: `tensorboard_trace_handler` does NOT work on this build -- torch 2.13 calls
    # export_chrome_trace(path, metadata) while the Habana shim defines (self, path), so the
    # handler raises TypeError at exit. Export directly and gzip ourselves.
    acts = [ProfilerActivity.HPU]
    if ACTIVITIES == "both":
        acts.append(ProfilerActivity.CPU)
    prof = profile(activities=acts, record_shapes=False)
    prof.start()
    for _ in range(iterations):
        step(x, w, dst, scratch, row)
    torch.hpu.synchronize()
    prof.stop()

    raw = os.path.join(PROF, "trace.json")
    # TWO workarounds, both required on this build (torch 2.13 / Habana 1.24.2):
    #  1. torch's own export passes a metadata argument the Habana shim does not accept, so call
    #     the wrapped profiler's method directly;
    #  2. the HPU side appends into the file and fails with `output_json.cpp: Failed to open`
    #     when it does not exist, so pre-create a VALID (empty) chrome trace first. Without this
    #     you get a trace with cpu_op events and NO device events at all.
    with open(raw, "w") as fh:
        fh.write('{"traceEvents": []}')
    prof.profiler.export_chrome_trace(raw)
    with open(raw, "rb") as src, gzip.open(raw + ".gz", "wb") as dst_gz:
        shutil.copyfileobj(src, dst_gz)
    os.unlink(raw)

    summary = {
        "root": ROOT,
        "shape": {"M": M, "K": K, "dtype": str(DT)},
        "iterations": iterations,
        "warmup": warmup,
        "activities": ACTIVITIES,
        "spec_files": sorted(os.listdir(SPEC)),
        "dump_files": sorted(f for f in os.listdir(DUMPS) if f.endswith(".json")),
        "trace_files": sorted(os.listdir(PROF)),
    }
    with open(os.path.join(ROOT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} files")
                      for k, v in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(*[int(a) for a in sys.argv[1:3]]))
