#!/usr/bin/env python3
"""fp8 vs bf16 MME arms at the sheet's calibration shape; merged-union occupancy."""
import json
import os
import statistics

os.environ.setdefault("HLS_MODULE_ID", "0")
os.environ["HABANA_PLUGINS_LIB_PATH"] = "/opt/habanalabs/habana_plugins"

import torch
import torch_hpu  # noqa: F401
from torch.profiler import ProfilerActivity, profile

DEV = "hpu"
OUT = "/tmp/vivid-hippopotamus-root/bench"
os.makedirs(OUT, exist_ok=True)


def fp8_gemm(a, b, sc):
    return torch.ops.hpu.fp8_gemm_v2(a, False, b, True, None, torch.bfloat16, sc, None, None, False)


def bf16_gemm(a, b):
    return a @ b


def merged_union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return sum(e - s for s, e in out)


def arm(tag, fn, args, n=30, warm=5):
    o = fn(*args)
    torch.hpu.synchronize()
    for _ in range(warm):
        fn(*args)
    torch.hpu.synchronize()
    p = os.path.join(OUT, f"{tag}.json")
    with profile(activities=[ProfilerActivity.HPU, ProfilerActivity.CPU]) as prof:
        for _ in range(n):
            fn(*args)
        torch.hpu.synchronize()
    prof.profiler.export_chrome_trace(p)
    ev = json.load(open(p))["traceEvents"]
    g = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"
         and "GEMM" in e.get("name", "").upper()]
    if not g:
        print(f"{tag:34s} NO GEMM EVENTS")
        return None
    u = merged_union([(e["ts"], e["ts"] + e["dur"]) for e in g])
    per_event = statistics.median([e["dur"] for e in g])
    print(f"{tag:34s} events={len(g):4d} per-event med={per_event:7.3f}us  "
          f"union/launch={u/n:7.3f}us  tids={sorted({e.get('tid') for e in g})}")
    return u / n


def main():
    print("torch", torch.__version__)
    for (M, K, N) in ((1, 5120, 1280), (1, 5120, 1024), (1, 5120, 2048)):
        a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
        af = a.to(torch.float8_e4m3fn)
        bf = b.to(torch.float8_e4m3fn)
        sc = torch.tensor(1.0, device=DEV, dtype=torch.float32)
        f8 = torch.compile(fp8_gemm, fullgraph=True, dynamic=False, backend="hpu_backend")
        fb = torch.compile(bf16_gemm, fullgraph=True, dynamic=False, backend="hpu_backend")
        try:
            arm(f"fp8_m{M}_k{K}_n{N}", f8, (af, bf, sc))
        except Exception as e:  # noqa: BLE001
            print(f"fp8 M{M} K{K} N{N} FAILED: {type(e).__name__}: {str(e)[:200]}")
        try:
            arm(f"bf16_m{M}_k{K}_n{N}", fb, (a, b.t().contiguous().t()))
        except Exception as e:  # noqa: BLE001
            print(f"bf16 M{M} K{K} N{N} FAILED: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
