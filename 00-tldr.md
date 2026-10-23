# Chapter 0 — TL;DR

**Read this first.** If you take nothing else from this manual, take these six things. Each
points at the chapter that develops it.

---

## The five facts that change how you write code

**1. Everything is statically compiled — there are no dynamic shapes.**
HPU bakes every tensor shape into the compiled graph; a `dynamic=True` request is forced to
static compilation. **You cannot pass a dynamic slice bound or shape and have one graph serve
several cases.** Design for a fixed shape set, and treat every new shape as a new program.
→ *Chapter 1*

**2. Integers are specialized exactly like shapes.**
This is the one people do not expect. A Python `int` used in traced code — a layer index, a
per-layer byte offset, a constant — becomes part of the **guard set**, and produces **a
separate compiled graph per distinct value**, precisely as a new shape does. **There is no
dynamic-argument escape hatch for ints.** If your code computes an offset from a loop index,
you have written one program per iteration.
→ *Chapter 1*

**3. Therefore the graph count multiplies by layer × constant × shape.**
An N-layer model is an **O(N × buckets)** specialization case **by construction, not by bug**.
A 43-layer model is an O(10²)-graph model before anything is done wrong. Do not treat a large
graph count as evidence of a mistake until you have checked it against this arithmetic.
→ *Chapters 2 and 3*

**4. A graph break costs twice.**
A break splits the compiled program into **more regions**, which means **more graph launches**
*and* **more host synchronizations** — and in this stack each launch blocks the host. So a
break costs you once in the extra region and again in the launch/sync traffic around it.
→ *Chapter 2*

**5. Prefer fewer, larger regions — but verify the specialization count fell, and do not
mistake coalescing for a fix.**
Coalescing is a real lever, but it is **necessary and not sufficient**. A region is only the
*unit inside which* one submission is possible — so you can coalesce successfully and still
compile to several submissions. And a coalesced region spanning G layers **still needs G
graphs** if it guards on each layer's int constants (fact 2). **Check both counts, and do not
report "we coalesced" as "we fixed the overlap."**
→ *Chapter 2*

---

## The one measurement rule that changes what you optimise

**Price in device occupancy, not in summed work.**
Measured: **925 ms of summed device work fits inside a 70.5 ms wall — 13.1× parallelism.** The
wall tracks the **union** of device-occupied intervals, not the sum. **So reducing total device
work buys you only its un-overlapped part**, and an optimisation that removes 34 % of kernel
time can leave the wall flat. Before you optimise anything, ask what it removes from the
**union**.
→ *Chapter 2*

**But the rule has two levels, and "never sum" is wrong on its own.** Union **within** a recipe;
sum **across** recipes. Recipes execute sequentially, so the per-recipe unions **partition** the
step exactly — on the reference arm, **110 recipes summing to 66.135 ms against `merged` = 66.1399**
(0.007 %). What is illegal is summing **event** durations, which double-count. Summing recipe unions
is the correct model, and it is what lets you price a span by its own union.
→ *Chapter 2 §2.9 rule 1; chapter 3 §3.6.4*

---

## The one stale claim to watch for

**A widely-repeated framing says a decode step is "launch/serialization-bound" with the device
largely idle. On our measurements that framing is STALE, and the lever it implies is the wrong
one.** Two competing models were fitted to the same archived runs:

| model | R² | RMS residual |
|---|---|---|
| `wall = k × node-executions` (launch/node model) | **0.346** | **107.85 ms** |
| `wall = a + b × merged` (occupancy model) | **0.999950** | **0.95 ms** |

**The occupancy model wins by two orders of magnitude in residual**, so the step is
**device-occupancy-bound and the lever is the union, not the launch count.** Independent notes
reach the same conclusion. The launch model also has **no usable coefficient** — its per-run
`k` varies by **171 % of its own mean**, so there is no stable µs-per-node to design against.
Excluding the two outlier runs, occupancy still wins by ~5×.

*(Provenance note, so a re-run is not a surprise: the launch figures come from one of three
closely-related fits in the same artifact — rank-mean n=13, rank-0 n=13, rank-0 n=12 — giving
R² 0.346 / 0.344 / 0.313. All three agree on the conclusion; 0.313 is the same result, not a
contradiction.)*

**And this reconciles an observation that otherwise looks like a contradiction.** Removing
**34 % of kernel time** once left the step wall **flat**. Under the launch model that is a
puzzle. Under the occupancy model it is exactly what you would expect — **removing summed work
does not reduce the union.** The two facts are one fact.

So: if you inherit a "launch-bound" claim, **check it against the union before acting on it.**
The stale framing is the single most likely thing to send you optimising the wrong quantity.

**And when you find a gap between the union and the wall, do not name it "communication" — count
the holes.** Merge the device intervals, take the complement, and histogram the holes by width: the
sub-50 µs ones are launch granularity and track **op count**, and a multi-millisecond hole is only a
collective if it is **aligned across the ranks that take part** — tested against a null, because at
~20,000 holes per step anything matches by chance. → *[Chapter 1 §1.8](01-basics.md)*,
[`examples/01-basics/gap_histogram.py`](examples/01-basics/gap_histogram.py)

---

**And when you write your own TPC kernel, the unit of cost is the VLIW *bundle*, not the instruction** — a `{...}` group in a `tpc-clang -S` listing is a 4-slot issue bundle, the device retires one per cycle, and the shipping scheduler leaves 1.2–3.4× of bundle count on the table. → *[Chapter 4](04-execution-model.md)*

---

**Tagging convention** (used throughout): **MEASURED** / **ASSUMED** / **INFERRED** /
**UNKNOWN**. See `README.md`. The numbers above are from one workload on one box — treat the
*shape* of each fact as general to Gaudi 2 and the specific figures as that workload's.
