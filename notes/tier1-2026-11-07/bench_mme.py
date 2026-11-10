#!/usr/bin/env python3
"""Tier-1 verification bench: MME cycle law + M=1 GEMV rate + per-node floor.

Verifies, on this box, claims transcribed from an external cost-model sheet:
  cycles = ceil(N/1024)*K + (K/64)*ROIs*(2**ceil(log2 M) - 1)   [claimed exact on fp8]
  M=1 fp8 GEMM 5120x1280 ~ 10320 cycles ~ 5.2-5.5 us
  per-node floor 1.2-1.7 us

Instrument: torch.profiler device spans (kernel events), never host Event timing,
because a launch costs ~11-15 us and would swamp a 5 us node.
Pinned to one module (HLS_MODULE_ID). Cold-ish compile, warm launches, n=30.
"""
import json
import math
import os
import statistics
import sys

os.environ.setdefault("HLS_MODULE_ID", "0")
# the image's plugin path is stale; the profiler needs the real one
os.environ["HABANA_PLUGINS_LIB_PATH"] = "/opt/habanalabs/habana_plugins"

import torch
import torch_hpu  # noqa: F401
from torch.profiler import ProfilerActivity, profile

DEV = "hpu"
N_WARM = 5
N_REP = 30
OUT = "/tmp/vivid-hippopotamus-root/bench"
os.makedirs(OUT, exist_ok=True)


def make_fn(kind):
    if kind == "matmul":
        return lambda x, y: x @ y
    if kind == "relu":
        return lambda x: torch.relu(x)
    raise ValueError(kind)


def run_arm(tag, kind, shapes, dtype):
    torch.manual_seed(0)
    tensors = [torch.randn(*s, dtype=dtype, device=DEV) for s in shapes]
    fn = torch.compile(make_fn(kind), fullgraph=True, dynamic=False,
                       backend="hpu_backend")
    out = fn(*tensors)
    torch.hpu.synchronize()
    for _ in range(N_WARM):
        fn(*tensors)
    torch.hpu.synchronize()

    path = os.path.join(OUT, f"{tag}.json")
    with profile(activities=[ProfilerActivity.HPU, ProfilerActivity.CPU],
                 record_shapes=False) as prof:
        for _ in range(N_REP):
            fn(*tensors)
        torch.hpu.synchronize()
    prof.profiler.export_chrome_trace(path)

    with open(path) as f:
        tr = json.load(f)
    ev = tr.get("traceEvents", tr if isinstance(tr, list) else [])
    spans = {}
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        dur = e.get("dur", 0)
        if cat not in ("kernel", "Kernel", "gpu_op", "hpu_op"):
            continue
        if dur <= 0:
            continue
        spans.setdefault(name, []).append(dur)
    print(f"\n### {tag}  kind={kind} shapes={shapes} dtype={dtype}")
    if not spans:
        print("  NO DEVICE EVENTS -- trace unusable")
        return None
    for name, durs in sorted(spans.items(), key=lambda kv: -statistics.median(kv[1])):
        if len(durs) < 3:
            continue
        print(f"  {name:52s} n={len(durs):3d} min={min(durs):9.3f} "
              f"med={statistics.median(durs):9.3f} mean={statistics.mean(durs):9.3f} us")
    return spans


def cycles_law(M, K, N, rois=1):
    return math.ceil(N / 1024) * K + (K / 64) * rois * (2 ** math.ceil(math.log2(M)) - 1)


def report_law(M, K, N, spans, key_sub="gemm"):
    """Compare the largest matching span against the law at the two candidate clocks."""
    if not spans:
        return
    cand = [(k, v) for k, v in spans.items() if key_sub in k.lower()]
    if not cand:
        print("  (no GEMM-named span; keys were: %s)" % list(spans)[:6])
        return
    name, durs = max(cand, key=lambda kv: statistics.median(kv[1]))
    med = statistics.median(durs)
    pred = cycles_law(M, K, N)
    print(f"  law: cycles={pred:.0f}  -> @1800MHz {pred/1800*1e-3:.3f} us | "
          f"@1650MHz {pred/1650*1e-3:.3f} us | @2000MHz {pred/2000*1e-3:.3f} us")
    print(f"  measured {med:.3f} us ({name})  => implied clock {pred/med/1e3:.0f} MHz, "
          f"measured/pred@1800 = {med/(pred/1800*1e-3):.3f}")


def main():
    print("torch", torch.__version__, "| device_count", torch.hpu.device_count())
    # ---- A. the calibration point from the sheet: M=1, K=5120, N=1280
    for dt, dtn in ((torch.bfloat16, "bf16"),):
        s = run_arm(f"mme_m1_k5120_n1280_{dtn}", "matmul",
                    [(1, 5120), (5120, 1280)], dt)
        report_law(1, 5120, 1280, s)
    # ---- B. N dependence at M=1 (law predicts steps at N/1024)
    for N in (1024, 2048, 4096):
        s = run_arm(f"mme_m1_k5120_n{N}_bf16", "matmul",
                    [(1, 5120), (5120, N)], torch.bfloat16)
        report_law(1, 5120, N, s)
    # ---- C. M dependence at fixed N,K (law predicts (2^ceil(log2 M)-1) term)
    for M in (2, 4, 8):
        s = run_arm(f"mme_m{M}_k5120_n1024_bf16", "matmul",
                    [(M, 5120), (5120, 1024)], torch.bfloat16)
        report_law(M, 5120, 1024, s)
    # ---- D. per-node floor: a trivial single-node region
    s = run_arm("floor_relu_8x8_bf16", "relu", [(8, 8)], torch.bfloat16)
    if s:
        for k, v in s.items():
            print(f"  floor span {k}: med={statistics.median(v):.3f} us")


if __name__ == "__main__":
    main()
