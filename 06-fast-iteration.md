# Chapter 6 — The fast iteration workflow

**The loop this chapter describes is how optimisation actually gets done on this hardware.** A full
run of a ported model costs minutes to hours; a span harness costs seconds. Everything else here
follows from that ratio.

Measured, on the run that produced chapter 5's real-model example: **1,053 s** to load
`Qwen3.6-35B-A3B-FP8` and get a first token. A span harness for one recipe of that model runs in
seconds. That is the ~100× that makes an iterate-in-isolation workflow worth building, and it is the
only argument this chapter needs.

---

## 6.1 Why cutting at a recipe boundary works

This is not a heuristic. It follows from three things this manual has already measured.

1. **Device work across recipes is serial.** *"cross-recipe overlap is exactly zero"* (§3.6.4). The
   scheduler co-schedules independent nodes **inside** one submission and does nothing across
   submissions (§2.1).
2. **At a recipe boundary, a recipe's leave-one-out equals its union.** The manual's warning that
   "the union overstates what a removal buys, by 1.5–4.3×" is about cuts **inside** a recipe, where
   the rest of the recipe still occupies the time. Across a whole recipe there is nothing left to
   occupy it.
3. **The wall tracks the union of device occupancy**, `wall = 21.724 + 0.996812 · merged`
   (R² 0.999931, §3.6.1).

Chain them and you get the licence for the whole workflow:

> **A span cut at recipe boundaries has Δwall ≈ Δunion.** Make the span's device union smaller and
> the step gets faster by about that much.

**It fails in exactly two places, and both are worth naming up front:**

- **A cut inside a recipe.** The removal frees its leave-one-out, which is 23–66 % of the union on the
  rows §3.6.4 measures. You will over-predict by up to 4×.
- **A change that alters the host cost per submission.** The `a = 21.7 ms` term is *outside* the span;
  a change that adds a submission, or a sync, can win on device and lose on the wall. This is the one
  thing the harness cannot see, and it is what step 6 is for.

*(The slope `0.996812` is not validated out-of-sample and rests on one pair of runs — §3.6.2 caveat 1.
Quote it with that. The proportionality is robust; the third digit is not.)*

---

## 6.2 Step 1 — get one high-quality, fully attributed profile

Everything downstream depends on this, and it is chapter 5's job: `TORCH_TRACE` and
`DUMP_POST_GRAPHS` set before the run, a cold recipe cache, CPU+HPU activity, and the profiled window
in steady state.

**What the profile must contain before it is usable:**

| | why |
|---|---|
| **compile ids → `file:line:function`** | so a span can be named by the code that produces it |
| **recipes → post-graph dumps** | so the span's node mix is known, and so the harness can be checked against it |
| **device union per recipe** | the ranking, and the baseline the harness must reproduce |
| **a coverage number** | what fraction of the step the attribution actually accounts for — a table missing a third of the step will send you at the wrong span |

**One profile per configuration.** A profile is valid for the task and configuration it was taken in
— decode at batch size N, context length L, on the card count you actually run. Change any of those
and the recipe set changes with it.

### 6.2.1 The worked example's first span

From the run behind §5.8.1 (`Qwen3.6-35B-A3B-FP8`, one card, decode):

| source line | engine union |
|---|---|
| `hpu_fused_moe.py:756:…patched_fused_moe_forward…` | **TPC 27.7 ms, MME 21.2 ms** |
| `qwen3_next.py:549:forward` (decoder layer) | TPC 13.3 ms, DMA 8.3 ms |
| `hpu_gdn_pytorch.py:841:hpu_chunk_gated_delta_rule` | MME 4.5 ms |

The fused MoE is the first span by inspection: largest single contributor, its inputs are the
routed-expert tensors, and at TP=1 it contains no collectives.

### 6.2.2 That span's floor, declared before any rewrite

This is the §6.4 exercise done once, on real numbers, so the shape of it is concrete.

**The computation.** Per token, per layer: 8 of 256 routed experts, each 3 matrices of
`[hidden 2048, moe_intermediate 512]`, plus one shared expert of the same shape; 40 layers.

| quantity | value |
|---|---|
| routed-expert parameters per token | 8 × 3 × 2048 × 512 × 40 = **1.007 G** |
| shared-expert parameters per token | 3 × 2048 × 512 × 40 = **0.126 G** |
| **bytes per token at fp8** | **1.132 GB** |
| FLOPs per token | 2.27 GFLOP |
| arithmetic intensity | **2.0 FLOP/byte** — against a ridge of ~200, so **bandwidth-bound by ~100×** |

**The floor.** At the two envelopes this manual has measured:

| carrier | rate | floor |
|---|---|---|
| DMA engine, 16–64 MB band (§3.2.2) | 1,776 GB/s | **0.638 ms/token** |
| TPC stream envelope (§3.1.1) | 2,147 GB/s | **0.527 ms/token** |

So the declared floor is **~0.53–0.64 ms per token**, and the compute term (0.005 ms) is irrelevant.
Anything slower than that is not paying for arithmetic and not paying for the bytes it must move.

**The measured value.** The decode recipe for this span executes once per layer per step; in the
profiled window it runs 21 steps:

| | |
|---|---|
| merged union, decode recipe | **34.1 ms** |
| steps in the window | **21** |
| **measured** | **1.62 ms/step** |
| **ratio to floor** | **2.5×** |

*(The span's other recipe — 30.0 ms — is the prefill path, 80 execution clusters against the decode
recipe's 643. Attributing it as decode would have inflated the figure to 3.05 ms/step. Check which
recipe is which before dividing.)*

**So there is ~1.0 ms/step of headroom in this span, and the floor says where to look: bytes.** The
plausible explanations are the ones this manual has already named — a static expert count read where
fewer are routed (§3.1.1's over-read), bandwidth not reached at these tile sizes (§3.2.1's cliff), or
scaffolding around the GEMMs (the prefill recipe of the same span carries 1,024 `MoeMultiplexer`
nodes, 514 `ExpandDims` and 258 `Reshape`). **Which one it is has not been established** — that is the
rewrite's job, and §6.3.1 says to look for exactly that scaffolding while choosing the span.

**What this example does not yet have:** the rewrite. The floor is declared, the measured value is
attributed, and the gap is quantified — but no rewrite has been attempted and no harness exists for
this span. §6.11 records that as the open item.

---

## 6.3 Step 2 — pick a span

A span is a region of the computation with a boundary you can cut at. Five tests, and a span that
fails any of them is not a span yet:

| test | what it means in practice |
|---|---|
| **clear inputs** | a finite, enumerable set of tensors you can save and reload |
| **replicable outputs** | same inputs → same outputs, so a correctness gate can compare them |
| **timable alone** | it can run without the rest of the model — no dependency on state you cannot capture |
| **harness-able in seconds** | seconds, not minutes. ~10 s is comfortable; a minute is arguable; more is a full run with extra steps |
| **covers what it touches** | if the span contains a collective, the harness must run it — which means multi-card |

**Cut between recipes, and grow the range until you can.** The cut is the whole point: a boundary
inside a recipe costs you up to 4× in over-prediction (§6.1). If the natural span ends mid-recipe,
extend it to the recipe edge — take the neighbouring work in rather than cutting inside one.

**Rank by the union column.** At recipe boundaries union ≈ leave-one-out, so the ranking in the
profile is the ranking you want, and §3.6.4's "quote the LOO when you price" is satisfied by
construction.

**Prefer spans with no collectives, for iteration speed.** A collective-free span harnesses in one
process. A span with a collective needs the whole communication setup, which makes the harness
expensive and slow to rebuild — and slow to rebuild is the one thing this workflow cannot afford.
That is a preference, not a rule: some of the biggest spans contain collectives, and they are still
worth doing.

**What "timable alone" does not require.** It does not require the span to be a pure function of its
inputs. A span that reads a KV cache or a running state is fine — capture the state as part of the
inputs (§6.5).

### 6.3.1 Look for removable work at the boundary — it is what makes a span worth rewriting

While you are choosing the span, look at what sits **immediately around it**: a dtype conversion, a
materialisation, a staging copy, a `cat`, a transpose-for-layout. Those are not just context — they
are the most profitable thing a rewrite can delete, and a span whose boundary is full of them is a
better target than one that is already clean.

**What to look for**, in the profile and the dumps you already have from step 1:

| tell | where it shows up | what a rewrite may be able to do |
|---|---|---|
| **a dtype conversion** | `_cast` / `_to_copy` nodes; a cast fused into a kernel prologue | keep the value in the dtype the consumer wants, or fold the conversion into the producer |
| **a materialisation** | `cat`, `contiguous`, a `memcpy_dma` staging node, `num_of_ROIs` in the hundreds | produce the result in the layout the consumer reads, so the copy never exists (§3.6.5's KV slab is the canonical case) |
| **an avoidable transpose** | `DmaTranspose` / `transpose_dma` nodes | choose the producer's output layout |
| **an eager-placed round trip** | `aten._to_copy` ×2 placed `eager`, which also splits the recipe (§2.2) | remove the host round trip entirely |
| **a re-read of the same bytes** | the same tensor appearing as an input to several nodes | fuse, so it is read once |

**Why this is worth a rule rather than a habit:** deleting work is the only kind of win that
transfers unconditionally. A rewrite that removes a staging pass or a cast reduces the span's device
union by construction, and by §6.1 that union reduction is a wall reduction. A rewrite that merely
*moves* the same work around has to earn its win against the launch/overlap accounting — which is
exactly where the manual's dead ends live.

**And it changes what you capture.** If the avoidable work sits just outside the span's natural
inputs, capture the tensors **on the far side of it** — the payload before the cast, the source
before the materialisation — so the harness can express the rewrite that eliminates it. A capture
taken only at the tight boundary bakes the avoidable work into the harness, and the best rewrite is
then unavailable.

---

## 6.4 The conceptual task: an optimal rewrite against a ceiling

Once the span is chosen, the job is **not** "make it faster". That has no target and no stopping
condition. The job is:

> **Estimate the fastest this computation could possibly run on this hardware, then rewrite the span
> from scratch to reach it.**

The estimate is the target. The rewrite is the work. The gap between them is the headroom, and the
headroom is the only thing worth spending iterations on.

**"Rewrite" is meant literally.** Not "tune the existing op", but *what is the fastest way to produce
these outputs from these inputs on this hardware* — constrained only by the correctness gate. This is
where the harness pays for itself: a from-scratch implementation can be dropped in and gated
immediately, against the same captured bytes, in seconds.

### 6.4.1 Compute the ceiling first, and name which term binds

The ceiling is not one number, it is one of three regimes, decided by the span's arithmetic
intensity `AI = FLOPs / bytes`:

| regime | test | what binds | what the rewrite must do |
|---|---|---|---|
| **bandwidth-bound** | `AI` below the machine's ridge point | bytes moved | **move fewer bytes** — fuse producer and consumer, stop materialising intermediates, keep tiles above the ~16 MB cliff (§3.2.1) |
| **compute-bound** | `AI` above the ridge | FLOPs at the MME rate | fewer FLOPs, or better tiling |
| **launch-bound** | neither — the span is small | per-launch latency | fewer, larger ops |

**§3.1.1 is the template for the calculation:** build the floor from the config and the checkpoint
headers — bytes the span must move, FLOPs it must do — and see which term dominates. It is a
pre-code exercise, not something read off a trace. §3.1.2 gives the FLOPs test, and the answer for
decode is usually that FLOPs do not bind at all.

**Getting the regime wrong sends you at the wrong lever**, which is the failure §3.6.4 documents: a
decomposition that ranked data movement first sent the port at DMA, when the same work later showed
the DMA families were the ones the staging rewrite had already removed.

### 6.4.2 State the floor with its assumptions

A floor is a claim of the form: *this span must move X bytes and do Y FLOPs; at Z GB/s and W TFLOP/s
that is T ms, provided …*.

The provisions are load-bearing:

- **which engine carries the bytes** — the TPC stream envelope (2,147 GB/s) is not the DMA engine's
  (~1,776 GB/s in the 16–64 MB band, §3.2.2), and the two differ by ~17 %;
- **which dtype** — the byte count is `elements × bytes_per_element`, and fp8/fp4 change it by 2–4×;
- **which tile size** — below ~16 MB the GEMV cliff means you are paying launch, not bandwidth
  (§3.2.1), and the floor is not the binding term at all;
- **whether the bytes are actually read once** — a materialisation reads and writes the same payload
  twice, so the *avoidable* byte count is larger than the necessary one (§6.3.1).

**A floor without its assumptions is a wish.** And a floor computed from the wrong bytes is worse than
no floor, because it will be believed — the trace carries no byte field at all, so every byte figure
is a *population × per-event-size* model (§3.4.6), and the model can be wrong by a factor.

### 6.4.3 Declare it before the rewrite

Write the ceiling down before you touch the code — same discipline as the correctness bound (§6.6). A
ceiling chosen after the fact is a description of what you achieved, and it will always be reached.

### 6.4.4 Judge the rewrite against the floor, not against the previous version

Comparing against the previous version tells you whether you moved; comparing against the floor tells
you whether you are done. Those are different questions, and only the second one terminates.

### 6.4.5 This is the loop's stopping condition

| where you land | what it means | what to do |
|---|---|---|
| **at the floor** | the span is done | stop, pick the next span |
| **2× off** | the gap is real headroom | the floor's binding term tells you which lever: bytes, FLOPs, or launches |
| **faster than the floor** | **the floor was wrong** | recompute it. That is a finding, not a victory — and it usually means a term was mis-costed (wrong engine, wrong tile regime, a byte count that double-counted) |

Without this, the loop has no exit: you stop when you get bored, and "done" is a feeling.

### 6.4.6 The floor moves when the computation changes

If the rewrite fuses, changes dtype, removes staging or changes the tile regime, the floor for the
*new* computation is a different number. Recompute it. Iterating toward the floor of a computation you
no longer perform is chasing a stale target — and the stale target is always the flattering one,
because it was computed for the slower version.

### 6.4.7 When the floor is unreachable, say which reason

Some spans cannot reach their floor, and the useful output is *which* limit stopped them:

- **launch granularity** — the span is below the tile cliff and pays per-launch cost (§3.2.1);
- **dependency structure** — the work is serial by construction, and recipes are serial across
  submissions (§6.1);
- **staging the compiler chose** — a layout decision outside the span, like the KV-slab
  materialisation (§3.6.5);
- **a term outside the span** — the host constant (§6.1), which the harness cannot see.

Naming the reason is what stops the next person re-attempting it, and it is the difference between
"we could not do better" and "this is where the limit is".

---

## 6.5 Step 3 — capture the inputs and outputs

Do an instrumented run of the real model and save, at the span boundary, the tensors going in and
the tensors coming out.

```python
# at the span boundary in the real model's own code
if os.environ.get("SPAN_CAPTURE"):
    torch.save({"inputs": inputs, "outputs": outputs, "state": kv_view},
               f"{capture_dir}/span-{call_index}.pt")
```

**Save bytes, not seeds.** A seed reproduces the *distribution*, not the values, so a later harness
run compares against different data and the correctness gate compares nothing. Every harness run must
load the identical bytes.

**Capture state, not just arguments.** If the span reads a KV cache, a counter, a running statistic
or a router table, that state is an input. A harness that omits it will produce plausible numbers for
a computation the model never performs.

**Capture more than one step.** One capture is one sample; a span whose timing depends on data (a
routed-expert pattern, a sparsity level) needs a handful, or the harness is measuring one lucky
iteration. Three to five is usually enough, and they must all pass the gate.

**Size.** Activations for a large model are routinely tens of MB, so captures go to `./data/` with a
`data-inventory.md` entry — they are records, not scratch, and they are not re-derivable without
another instrumented run. Trim to the tensors the span actually reads: a capture of "everything in
the layer" is mostly payload the span never touches.

---

## 6.6 Step 3b — build the correctness gate

**The gate runs on every harness run, and a failing gate refuses to report a time.** A harness that
gets faster while producing different answers is measuring a different computation, and the speedup
is fiction. This is not a formality: it is what makes the iteration loop safe.

Pick the strongest gate the span admits:

| tier | when it applies | what it is |
|---|---|---|
| **bit-exact** | the change is supposed to be arithmetically identical (a layout change, a fused-but-same-order kernel, a scheduling change) | every output tensor equal, element for element |
| **ulp histogram** | the change reorders or reassociates arithmetic | count the ulps of difference per element and compare the distribution against a bound |
| **relaxed ulp histogram** | the span stacks a few layers of error, or crosses a dtype boundary | same, with the bound scaled to the number of accumulation stages |
| **weaker** | the span's output feeds a sampling step, or is only meaningful statistically | a declared statistic with a declared margin — and *say* that it is weaker |

**Declare the bound before you measure.** A bound chosen after seeing the result is a description of
the result. State the tier and the number, then run.

**A ulp count is only meaningful against a stated format** — an ulp of bf16 is not an ulp of fp8, and
the ordered-integer key differs per format. Use a per-format instrument, not a generic
`abs(a-b) < tol`.

---

## 6.7 Step 4 — check the harness is a proxy

**This is a gate, and it is separate from "iterate" for that reason.** A harness that is faster than
the model is not the goal; a harness that is faster *and is the same computation* is.

Compare, against the profile from step 1:

| | must match |
|---|---|
| **recipe count** | the harness compiles to the same number of recipes as the span has in situ |
| **node mix** | the same node types in the dump — the same fusion, not a different one |
| **op and launch counts** | the same population; a harness with 29 ops where the span has 155 is a different span |
| **device union** | within a few percent of the in-situ union for that recipe |

The first three come straight out of chapter 5's artifacts and cost minutes. They are also the ones
that catch the real failure: **a harness that quietly computes something slightly different** —
different shapes, a different fusion, an extra cast — and then gets optimised instead of the span.

**If the proxy check fails, do not tune the harness until it agrees.** Either fix the harness's
fidelity, or accept that this span needs a full run and pick a different one. A harness that has been
tuned into agreement is measuring itself.

---

## 6.8 Step 5 — iterate

**The work here is the rewrite described in §6.4** — a from-scratch attempt at the computation, aimed
at the floor and judged against it. What this step adds is the instrumentation that tells you whether
each attempt moved you, and why.

Change one thing, run the harness, read five numbers:

| number | what it tells you |
|---|---|
| **time** | the thing you are optimising |
| **device union** | whether the win is real device work or overlap |
| **op count** | whether you removed work or hid it |
| **launch / submission count** | whether you moved work to the host |
| **device busy fraction** | how much of the harness's wall the device was actually working |

**Read the busy fraction as a diagnostic, not a gate.** It is not a precondition for measuring — a
relative comparison between two equally-starved arms is still a comparison, and a harness that goes
from 1 s to 0.5 s is a useful result whatever the queue was doing. But a **starved** harness has a
known asymmetry: it under-weights device-side wins and over-weights launch-side wins. So when a
change wins, ask which of these happened:

- **real device work was removed** → it transfers, and the harness may understate the gain;
- **the launch count dropped** → it may not transfer at all. That is the trap chapter 0 exists to
  kill: on this hardware the wall tracks occupancy, not launches, and a launch-count win in a
  starved harness is exactly what a full run fails to reproduce.

Reporting ops and launches alongside time is what lets you tell those apart. It is one extra counter
and it is the difference between a harness that predicts and one that flatters.

**Keep the gate on every run.** The correctness gate and the proxy check both run at every iteration,
not once at the start — an optimisation that breaks correctness is usually an optimisation that
removed work it was supposed to keep.

---

## 6.9 Step 6 — confirm in situ

Once per span, when the harness has stopped improving. One full run with the replacement in place,
and compare against the same full run before it.

**Set the expectation before you look.** If the predicted saving is smaller than the full run's
run-to-run spread, the confirmation will come back inconclusive — that is arithmetic, not failure.
The rank-to-rank spread on `merged` is on the order of a millisecond (§3.6.4's rank-spread note), so
a span win worth a fraction of that cannot be confirmed by a full run at all. Take it on the harness
evidence, say that is what you did, and move on.

**A confirmation that comes back flat is a result, and it means one of two things:** the span was not
on the critical path in the way §6.1 assumes (a cut inside a recipe), or the change moved cost into
the host term. Both are worth knowing and both are cheaper to diagnose from the harness's four numbers
than from another full run.

---

## 6.10 Anti-patterns

**The harness that measures itself.** The fastest harness is not the goal — a *faithful* one is. The
manual's own dead-end register has the canonical case: a component benchmark silently destroyed by
`torch.stack([a, a])` — the same tensor twice, **75× slower** than two distinct tensors — and the
number was believed (§3.8 dead end 10). Every harness should be assumed to be measuring itself until
the proxy check says otherwise.

**Reimplementing the span.** Write the harness so it calls the model's own function. A reimplemented
span drifts from the real one the first time the model changes, and the drift is invisible — the
harness keeps producing a number.

**The harness that becomes the project.** A harness accumulates features until it is a second model,
with its own correctness problems and its own compile time. The tell is a harness that no longer runs
in seconds. At that point it has stopped being an iteration tool.

**Tuning the harness until the proxy check passes.** Covered above, and it is the failure mode this
chapter most wants to prevent, because it looks like diligence.

**Optimising without a floor.** If you do not know what the computation *could* cost, you cannot tell
a win from noise, you cannot tell which lever to pull, and you have no stopping condition — so you
stop when you get bored, and "done" becomes a feeling. §6.4 exists to prevent this, and the cost of
skipping it is not a wrong number but an endless loop.

---

## 6.11 What this chapter does not establish

| # | open | what would settle it |
|---|---|---|
| 1 | **The Δunion → Δwall transfer has not been measured end to end** for a span optimised in a harness and confirmed in situ. §6.1's arithmetic is a composition of three measured facts, not a measurement of the workflow itself. | One span, taken through all six steps, with both the harness delta and the in-situ delta recorded. |
| 2 | **How far a harness can drift before the proxy check matters.** The check compares recipe count, node mix, op counts and union, but no threshold on any of them is calibrated. | Two harnesses of the same span at deliberately different fidelity, and the point at which their speedups diverge. |
| 3 | **The cost of a faithful multi-card harness** for a span containing a collective — measured only as "expensive", not as a number. | Build one for a collective-bearing span and time the rebuild cycle. |
| 4 | **Whether a captured span reproduces the in-situ data distribution.** A capture is one sample; a span whose cost depends on data (routed-expert patterns, sparsity) may be unrepresentative. | Compare harness union across several captures against the in-situ union across several steps. |
| 4a | **No span has yet been *rewritten* against a declared floor.** §6.2.2 now does the first half on
real numbers — floor declared, measured value attributed, gap quantified at **2.5×** — but the rewrite
itself has not been attempted and no harness exists for that span. So "the floor was reached" has no
worked instance anywhere in this manual. | Build the harness for §6.2.2's span (§6.5–6.7), rewrite it
(§6.4), and report the achieved/floor ratio — including the case where the floor turns out to be wrong
(§6.4.5). |
| 4b | **Which of three explanations owns the 2.5× gap in §6.2.2** — a static expert count read where
fewer are routed (§3.1.1), bandwidth not reached at these tile sizes (§3.2.1), or scaffolding around
the GEMMs (the same span's prefill recipe carries 1,024 `MoeMultiplexer` nodes). | The rewrite, plus a
per-node byte/op accounting of the decode recipe against the 1.132 GB/token the floor assumes. |
| 5 | **The correctness-gate tier for a span that ends in a reduction over experts.** Bit-exactness is unavailable once routing changes; what bound is defensible is not established here. | A ulp histogram across a set of realistic routing patterns, with the bound declared first. |

---

**Tagging convention** as throughout: **MEASURED** / **ASSUMED** / **INFERRED** / **UNKNOWN** — see
`README.md`. The load-time figure is from one run on one box; the seriality and wall-fit figures are
chapter 3's, quoted with their caveats.
