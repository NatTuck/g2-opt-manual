# Examples — Chapter 6 (The fast iteration workflow)

**Two files**, and only one of them is a template. The workflow's whole point is that each span is
different, so the harness is built per span against captured data — but the *gate* is the part that
generalises, and one of its failure modes needs no device at all.

| file | what it does | needs a device? |
|---|---|---|
| [`span_harness.py`](span_harness.py) | the harness pattern: load captured bytes → run the **model's own** span function → gate correctness **first** → time, reporting ops/launches/busy alongside → print what the proxy check needs | yes (it runs the span), but the *loop* it enables costs seconds |
| [`traced_state_gate.py`](traced_state_gate.py) | the gate for a span that **reads state**: a tensor cache inside a traced region is baked into the graph on a hit, and the input is then ignored. Runs three variants (no clear / per-region clear / per-use clear) against a reference; `--selftest` asserts it **fails** on the buggy constructs | **no** — CPU only |

## Why there is no worked example here

The chapter's worked example is a real span from a real run — `hpu_fused_moe.py:756` at
**TPC 27.7 ms + MME 21.2 ms** (chapter 5 §5.8.1) — and building its harness requires capturing that
model's routed-expert tensors, which is a multi-gigabyte artifact and a device run. Shipping a toy
substitute would teach the wrong lesson: the workflow is worth having *because* the span is real.

## The three things the template enforces

0. **Inputs mean every value the span reads, outputs every value it writes that a later step reads** — not the function signature. `--require-state` makes a stateful span without its state a hard failure rather than a silently self-consistent harness (§6.5.1).
1. **Captured bytes, not synthetic inputs.** `--capture` is required and there is no fallback. A
   harness that invents its inputs is measuring a different computation, and it will keep producing a
   number while doing so.
2. **The gate runs first, and a failure suppresses the timing.** A faster wrong answer is not a
   result. The gate is `bit-exact` by default because that is the right default for changes that are
   meant to be arithmetically identical; declare `ulp` with a bound *before* you run.
3. **The model's own function.** `--span module:function` imports the real code path. A
   reimplementation drifts from the model silently, and the drift is invisible in the output.

## Traps this template encodes

| trap | where it bites |
|---|---|
| a seed instead of saved bytes | every harness run compares against different data, so the gate compares nothing |
| **omitting state (KV cache, attention buffers, counters, router table)** | the omission is invisible twice over: timing is unchanged (buffers are allocated at full size whatever they hold) and the **gate passes** (reference and rewrite both see the same empty buffer). Use `--require-state` and audit the capture against the recipe's own tensor table (chapter 6 §6.5.1) |
| summing durations instead of unions | per-TPC replication inflates the sum; the same block reads DMA-bound one way and MME-bound the other (§2.9 rule 1) |
| reading a win as "faster" without asking *which* work went away | a win from fewer launches may not transfer; a win that removes device work does (chapter 0) |
| tuning the harness until the proxy check passes | the harness is then measuring itself |
| a harness that no longer runs in seconds | it has stopped being an iteration tool |
