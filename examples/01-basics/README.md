# Examples — Chapter 1 (Gaudi/HPU Basics)

Small, illustrative programs for the hardware model in [`../../01-basics.md`](../../01-basics.md).

**Status:** these are distilled from code that *was* run and verified on Gaudi 2, but they have
**not been re-run in this repository**. They are illustrative skeletons: the API call sequences
and the "what to look for" notes are the valuable part, not the line count. Where a call sequence
was hard to get right, the comment says why.

| file | what it shows | needs a device? |
|---|---|---|
| [`synapse_cpp_minimal.cpp`](synapse_cpp_minimal.cpp) | the smallest correct SynapseAI C++ program: init → acquire → graph → one GEMM node → compile → launch | yes |
| [`dump_post_graphs.py`](dump_post_graphs.py) | reading **per-node engine identity** out of the compiler's own graph dump — the answer to "does this recipe use more than one engine?" | yes (to produce a dump) |
| [`trace_engines.py`](trace_engines.py) | attributing trace events to engines from `args["HW event name"]`; plus the `torch.copy_` trap | yes for `--selftest`, no for parsing a trace |
| [`devprobe.py`](devprobe.py) | the 20-line **pre-flight** that catches a silently-broken profiler shim | yes |
| [`gap_histogram.py`](gap_histogram.py) | decomposing the **`wall - merged` gap** by hole width, and testing rank alignment against a null — the instrument that decides whether a gap is communication | **no** — CPU-only, parses an existing trace; `--selftest` is CPU-only too |

---

## The five things worth taking from these examples

### 1. A guid is not an engine

`synNodeCreate(..., "gemm", ...)` names an **operation type**. The graph compiler decides the
engine during lowering. `gemm` does land on the MME — but that is the compiler's node-type →
engine mapping, not the guid selecting hardware. Corroboration: a TPC kernel named
`matrix_multiply_dram_f32` exists, so "the matmul guid" is not unique.

**Consequence:** you cannot place an op on an engine by choosing a guid. You influence placement
indirectly, by what the compiler can cluster (see Chapter 2).

### 2. Two API details that cost real time in the C++ example

- **`<synapse_common_types.hpp>` is required.** `synTensorDescriptor` is only *forward-declared*
  in the C header that `synapse_api.h` pulls in, so the type is incomplete without the C++ header.
  This is why the C++ API is mandatory for this route.
- **Tensor sizes are listed innermost-first.** A row-major `[rows, cols]` matrix is
  `{cols, rows}`. Getting this wrong gives a silently transposed GEMM — and a **square** test
  cannot catch it, which is why the example uses non-square dims (64×48×80).

Also: **graph inputs must live in a persistent, application-managed section, one section per
input.** Otherwise the compiler's persistence check rejects the graph; and two tensors sharing a
section at the same offset overlap, which the launch rejects.

### 3. `torch.copy_` is not the DMA engine (eager mode)

The single most expensive labelling error in the source material. `y.copy_(x)` in **eager** mode
lowers to a **TPC** kernel (`memcpy_u8`) — measured: 768 TPC kernel events, **zero** DMA-engine
events. Under `torch.compile` the same op *does* go to the engine.

So any "DMA engine bandwidth" figure measured via eager `torch.copy_` is **TPC** data. Run
`trace_engines.py --selftest` once and you will never have to assume again.

### 4. Pre-flight the profiler, every time

A stale `HABANA_PLUGINS_LIB_PATH` makes `synProfilerStart` **fail silently**. You still get a
trace; it parses; it contains `cpu_op` / `privateuse1_runtime` and **no `kernel` / `hpu_op` at
all**. That reads as "the device did nothing" — a conclusion someone will act on. `devprobe.py`
is the guard: run a known matmul under the profiler and check that `kernel` and `hpu_op` are
present in the category histogram.

### 5. The gap between the union and the step is a *shape*, not a residue

`merged` is an occupancy, so it is always below the step period, and the difference is the time no
engine was executing. The reflex — "that gap is communication" — is testable **without a device**:
merge the device intervals, take the complement, histogram the holes by width, and check whether the
wide ones are rank-aligned against a null.

`gap_histogram.py` does that on a trace you already have. On the manual's own W0 decode arm it
reproduces §3.6.4's `merged` to four decimals and shows that **~42 % of the gap is ~21,000 sub-50 µs
holes per step** (launch granularity, tracking op count) while the multi-millisecond part is **5
holes per step** — too few to be one per layer, and not aligned across the four ranks. The lesson
generalises: **count the holes before you name them.** The `--selftest` is the control that makes a
0 % alignment reading mean something.

---

## Gotchas these examples encode

| gotcha | where it bites |
|---|---|
| `DUMP_POST_GRAPHS=<path>` needs the directory to **pre-exist** | the compiler treats a missing path as a *file*; concurrent writers race and the compile fails |
| the dumps land in a **hidden** directory (`.graph_dumps/`) | shell `**` globs skip it — one recon saw **24 of 1,141** files. Use `os.walk`/`find` |
| `LOG_LEVEL_ALL=2` **alone** is enough for a cheap multi-engine confirmation | it prints per-recipe arc jobs by engine name; no dump directory needed |
| profiling **CPU + HPU** together inflates the host span ~3× | makes a device-bound arm look host-bound; profile HPU-only for device questions |
| a summed device duration is inflated by the per-TPC replication factor | measured 18–24× for TPC kernels; build durations from **unions**, never sums |
| an alignment rate with no null hypothesis is not a measurement | ~20,000 holes per step means a ±50 µs tolerance matches almost anything by chance; `gap_histogram.py --selftest` is the control |
| `DUMP_POST_GRAPHS` and the bridge's `PT_HPU_GRAPH_DUMP*` are **two different instruments** | setting one and reading the other's empty directory as a failure |
