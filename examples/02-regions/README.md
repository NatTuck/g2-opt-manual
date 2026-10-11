# Chapter 2 examples — small programs and regions

Companion code for [chapter 2](../../02-regions.md). Each file is self-contained, annotated with
**what to look for**, and prints its own verdict.

| file | runs on | what it shows |
|---|---|---|
| [`01_two_engine_recipe.py`](01_two_engine_recipe.py) | HPU | The minimal thing that overlaps: an independent TPC op and MME op in **one** `torch.compile` region. Computes `CROSS`, `Launch`/step and the precondition, and tells you when the arm is INCONCLUSIVE. |
| [`02_region_that_overlaps_and_one_that_does_not.py`](02_region_that_overlaps_and_one_that_does_not.py) | HPU | The matched pair from §2.2: the same two ops, twice, differing by **one dataflow edge**. Uses a single-shot harness (sync before every call) so cross-step pipelining cannot explain the result. |
| [`03_inspect_graph_dumps.py`](03_inspect_graph_dumps.py) | **CPU only** | Reading a `DUMP_POST_GRAPHS` directory: per-node `engine`, per-graph engine composition, the DMA/MME cost models, and a filter cross-tab. `--demo` generates a synthetic dump so you can run it anywhere. |

Start with `03 --demo` (no HPU needed), then read `01`'s docstring, then run `01` and `02`.

## Running them

```bash
source activate.sh
./venv/bin/python 03_inspect_graph_dumps.py --demo          # CPU, always works
./venv/bin/python 01_two_engine_recipe.py                   # needs one free AIP
./venv/bin/python 02_region_that_overlaps_and_one_that_does_not.py
```

## Environment rules that each cost a run

1. **Do not set `HABANA_VISIBLE_DEVICES`** in a container that already sets it. Disagreeing views
   give silent mis-pinning.
2. **Never unset `GC_KERNEL_PATH`.** An empty value fails every compile.
3. **`torch.hpu.set_device(i)` must be the first HPU call** if you pin. There is no working
   self-report API — read `os.environ["HLS_MODULE_ID"]`.
4. **Collect `ProfilerActivity.HPU` only.** Adding CPU activity inflates the trace's host span ~3×
   and manufactures a false empty-queue verdict (§2.9, rule 6).
5. **Stagger concurrent launches by ~2 s** if you run several arms at once; simultaneous starts lost
   processes to `synStatus=8` in 3 of 5 attempts.

## Why these three, and not a bigger harness

The source project's full harness had 13 arms. Three are reproduced here because they are the ones
that make the *reasoning* legible:

* `01` is the existence proof — overlap is real, and it is not a hardware property you can assume.
* `02` isolates the condition that most often breaks in real code (a dataflow edge) with a control
  that changes nothing else.
* `03` is the instrument you need for the structural questions, and it is the one that runs without
  a device.

The chapter's other evidence — the 24-arm placement table (§2.4), the absorption percentages (§2.6),
the co-residency sweep and the TPC-count arm (§2.7) — comes from synthetic C-API and Python harness
runs in the source project (`Archive/by-date/2026-10-11/recipe_boundary/`,
`Archive/by-date/2026-10-10/t2_graph_overlap/`, `Archive/by-date/2026-10-11/overlap_dma_probe/`).
They are quoted with their conditions in the chapter rather than reproduced here, because their value
is in the *design* (what was held fixed) more than in the code.
