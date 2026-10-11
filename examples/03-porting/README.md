# Examples — Chapter 3 (Porting a large language model to HPU)

Illustrative programs for the porting material in [`../../03-porting.md`](../../03-porting.md).

**Status:** annotated sketches. They were **not executed in this repository** — no HPU was used to
produce them. The valuable part is the *shape* of the code and the **"what to look for"** notes,
not the line count. Where a hazard is unrecoverable, the comment says so and says why.

| file | what it shows | needs a device? |
|---|---|---|
| [`fp8_scale_adjustment.py`](fp8_scale_adjustment.py) | the fp8 weight-loading wrapper: the ×0.5 / ×2.0 conversion, the **e8m0 raw-`uint8` expert-scale bug and its fix**, and the **structural guard** that makes the conversion safe. Runnable on the host (`python fp8_scale_adjustment.py`). | **no** — host-only by construction |
| [`specialization_probe.py`](specialization_probe.py) | counting compiled graphs at **both** compile levels, reading guard expressions, and the post-warmup assertions that gate a timing measurement. | yes for live counters; **no** for parsing a trace |
| [`moe_stage_sketch.py`](moe_stage_sketch.py) | the **six**-stage routed-expert path, the 2-D-only `fp8_gemm_v2` contract, the free-K-axis-order trick, and the measurement checklist. | yes to run; no to read |

---

## The four things worth taking from these examples

### 1. The fp8 conversion is safe by LAYOUT, not by design

The ×0.5 weight conversion works **only** because it happens on a **host/checkpoint** tensor,
before the value meets the device decoder. On a device tensor the same expression is a **no-op
that preserves the poison** — and the poison is **unrecoverable**:

* `inf * 0.5 == inf`, `nan * 0.5 == nan`;
* the post-decode map is many-to-one, with nothing downstream inverting it;
* `0x78` is a **fixed point** of the re-encode, and `0x79–0x7E` all collapse to **one** NaN code.

So `fp8_scale_adjustment.py` guards **structurally** — it refuses to run on a non-host tensor. A
scan for bad code values is *not* a substitute: it is data-dependent (a tensor whose `amax < 256`
passes even on the broken path), ambiguous (`0x7F` is a legitimate `e4m3fn` NaN; `0x78` a
legitimate checkpoint code), and detect-only — and the poison is already unrecoverable by then.

### 2. A "scale" tensor can arrive as raw exponent bytes

The e8m0 expert scales in some MoE checkpoints are stored as **raw `uint8` bytes**, not floats.
Fed to a naive wrapper they hit the `* 2.0` branch meant for fp32 scales and are **corrupted**
(byte 115 → 230). Decode `2 ** (byte - 127)` **before** scaling. The example runs this case on the
host so you can see both the wrong and the right number without touching a device.

### 3. There are two compile levels, and only one of them is guarded

Level 1 is the **recipe** (`recipe_key` — IR + declared shape *ranges*). Level 2 is the **Synapse
graph** (`graph_key` — recipe + concrete shapes + bucket). A guard on level 1 can be clean while
level 2 compiles every step.

**Naming trap:** the profiler event called `compileGraph` is the **level-2** compile, not a dynamo
compile. One port showed **93 `compileGraph` events per step** while level 1 was clean.
`specialization_probe.py` counts both and asserts both are zero after warmup.

### 4. The MoE path is six stages, and two of them get forgotten

The scale gather and the `repeat_interleave` expand are easy to dismiss as details. Measured on our
workload, the expand alone is **45,365 ops/step and 10.78 ms merged** — one of the largest op-count
items in the whole step. If your accounting counts only the kernels you wrote by hand, you will
undercount the path by **3.4×**.

---

## Dead ends these examples encode

Recorded so nobody re-walks them (chapter 3 §3.8 has the full register).

| dead end | why |
|---|---|
| **"Remove the `cat`"** | The graph compiler inserts its own `concat_logical` + `memcpy_dma` for GEMM operand staging **regardless** — a GEMM-containing recipe has DMA layout ops whether or not the user's `cat` exists. Verified in a controlled A/B: `concat_logical` present in **both** arms, 2 nodes each. The IR also cannot decide critical-path membership (no unique node ids), and the one synthetic delta was INCONCLUSIVE (it changed the engine mix and asserted no precondition). |
| **Re-expressing the dequant in native torch ops** | Bit-exact, and **21.7× slower** — 76 % of the cost is the `torch.gather` **int32 index** (a 100 MB index for a 25 MB payload). |
| **A 256-entry LUT for the fp4 decode** | The lookup instruction's table is a **built-in coefficient table** selected by `funcid`; the caller's table is never loaded, and four different tables give byte-identical output. |
| **Pre-dequantizing experts at load** | Doubles the weight bytes (36.8 → 71.4 GB/rank) on a card with 12.3 GB free. HBM-blocked. |
| **`torch.stack([a, a], dim=-1)`** | The same tensor twice costs **75×** two distinct tensors (22,532 µs vs 301 µs on a 50 MB output). It silently destroyed a component benchmark. |
| **Profiling CPU + HPU together** | Inflates the host span ~3× and manufactures a false empty queue (a queue-depth metric collapsed 23.3 → 1.67). |
| **`**`-globbing a dump directory** | Skips the hidden `.graph_dumps/` — one recon saw **24 of 1,141** files. Use `os.walk`/`find`. |

---

## Gotchas these examples encode

| gotcha | where it bites |
|---|---|
| the profiler trace **carries no byte counts** | every "GB/step" is a shape × call-count **model**; cross-check it against measured DMA merged time rather than trusting it |
| a device event is a **replication count**, not a launch | one event per (launch × TPC/DMA engine); kernel events ÷ recipe launches measured 544 / 569 / 971 across three traces of one model |
| `merged` is an **occupancy union**, not summed work | 925 ms of summed device work fitted inside 70.5 ms of occupancy (13.1×) — only the un-overlapped part counts |
| the device fp8 activation ceiling is **240**, not 448 | using 448 clips the largest element of every expert |
| a **routed-only** read vs **all-local** experts | 12.9× vs 1.21× traffic — check this *before* any fused-expert work |
