# Examples — Chapter 5 (Profiling and the identity chain)

The three programs that produced chapter 5's measurements. Unlike most examples in this manual,
**these were run on a real Gaudi 2 on 2026-10-21** and their outputs are archived
(`Archive/by-date/2026-10-21/recipe-chain/`).

| file | what it does | needs a device? |
|---|---|---|
| [`recipe_chain_demo.py`](recipe_chain_demo.py) | three blocks, one engine each, each its own `torch.compile` region; sets `TORCH_TRACE`, `DUMP_POST_GRAPHS` and a cold recipe cache **before importing torch** | yes |
| [`attribute_chain.py`](attribute_chain.py) | walks a trace back to source lines: code → compile id → recipe → dump → nodes → engines, **and verifies the result** | **no** — parses an existing trace, CPU-only |
| [`metrics_probe.py`](metrics_probe.py) | reads `graph_compilation` / `recipe_cache` from the backend metrics (graph name + compile duration, per-recipe hit/miss) | yes |

## The order to read them in

1. **`recipe_chain_demo.py`** — the identity knobs are the point. `TORCH_TRACE` and
   `DUMP_POST_GRAPHS` are read early and cannot be set after the fact; a run without them produces a
   trace you can time but cannot attribute (chapter 5 §5.1).
2. **`attribute_chain.py`** — the chain, with its three verdicts spelled out in the output:
   `key join, 1:1 verified`, `ORDER-BASED …`, or `AMBIGUOUS — no pairing available`. It **refuses**
   on a real model, because the region→recipe leg is a pairing that only holds when the launch and
   enqueue streams are 1:1.
3. **`metrics_probe.py`** — the two extra keys the backend metrics expose, for when you want to know
   *which recipe is expensive to compile*.

## Traps these examples encode

| trap | where it bites |
|---|---|
| `tensorboard_trace_handler` raises `TypeError` on this build | torch passes `(path, metadata)`, the Habana shim takes `(self, path)` — call `prof.profiler.export_chrome_trace(path)`. Reported upstream as issue 23 |
| `torch.compile` needs `backend="hpu_backend"` | the default backend is inductor, which dies with `KeyError: 'hpu'` |
| the dumps land in `<dir>/.graph_dumps/` | a non-recursive `listdir` reports 0; `glob("**")` skips hidden dirs — use `os.walk` |
| a missing dump directory is treated as a **file** | the GC writes one JSON to that path, and the next run's `makedirs` fails |
| HPU-only collection drops every compile id | `Torch-Compiled Region` is a CPU annotation; include `ProfilerActivity.CPU` when you need attribution |
| summed per-engine durations invert the answer | 400 DMA events vs 160 MME events made a matmul block look DMA-bound; use **unions** |
