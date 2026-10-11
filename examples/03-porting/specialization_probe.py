#!/usr/bin/env python3
"""Example: a specialization / graph-count probe for a ported model.

**Status: annotated sketch. Not executed as part of the manual. No HPU was used to produce it.**
Read chapter 3 §3.5 before using it. Some API surfaces below are version-dependent — where that
is so it is marked `VERIFY`.

WHY THIS FILE EXISTS
--------------------
On HPU there are no dynamic shapes, and **Python ints used in traced code are specialized too**.
So the number of compiled graphs is not "one per region" — it multiplies by
**layer x constant x shape**. A port that bakes a per-layer index into traced code silently
allocates a graph per layer.

There are **two independent compile levels**, each with its own cache and key, and **both must
be zero after warmup** or any timing measured over that window is invalid:

  * level 1 — the **recipe** (`recipe_key`): the IR plus the declared shape *ranges*.
  * level 2 — the **Synapse graph** (`graph_key`): recipe plus concrete shapes plus the bucket
    and dynamic-bucket info.

NAMING TRAP: the profiler event called `compileGraph` is the **level-2 (Synapse)** compile, not
a dynamo compile. A guard can be clean at level 1 while level 2 is wide open. One port showed
93 `compileGraph` events per decode step while level 1 was clean.

TAGS: MEASURED / ASSUMED / INFERRED / UNKNOWN, per the manual's convention.
"""

from __future__ import annotations

import collections
import gzip
import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# PART 1 — in-process counters. Call `snapshot()` around a warmup and a measured window.
# ---------------------------------------------------------------------------


@dataclass
class SpecializationReport:
    unique_graphs: int = 0
    guard_sets_evaluated: int = 0
    recompiles: int = 0
    recipe_cache_additions: int = 0
    notes: list = field(default_factory=list)

    def delta(self, later: "SpecializationReport") -> "SpecializationReport":
        """Difference between two snapshots — this is the quantity you report."""
        return SpecializationReport(
            unique_graphs=later.unique_graphs - self.unique_graphs,
            guard_sets_evaluated=later.guard_sets_evaluated - self.guard_sets_evaluated,
            recompiles=later.recompiles - self.recompiles,
            recipe_cache_additions=later.recipe_cache_additions - self.recipe_cache_additions,
        )


def snapshot() -> SpecializationReport:
    """Read the specialization counters that the compile stack exposes.

    `torch._dynamo.utils.counters` is a `defaultdict(Counter)`. The `stats` bucket carries the
    distinct-graph count. MEASURED: this is the standing metric used by the source project
    (`counters["stats"]["unique_graphs"]`).
    """
    import torch

    rep = SpecializationReport()
    try:
        counters = torch._dynamo.utils.counters  # VERIFY: private API, may move
    except Exception as exc:  # noqa: BLE001
        rep.notes.append(f"cannot read dynamo counters: {exc}")
        return rep

    stats = counters.get("stats", {})
    rep.unique_graphs = int(stats.get("unique_graphs", 0))

    # `guard_added_fast` carries the guard EXPRESSIONS; its count is the guard-set evaluation
    # count. The *expressions* are what you want to read when a key is unintentional.
    rep.guard_sets_evaluated = int(stats.get("guard_added_fast", 0))

    metrics = counters.get("compilation_metrics", {})
    rep.recompiles = int(metrics.get("recompiles", 0)) if metrics else 0

    # Bridge-side signal that a NEW Synapse graph was compiled at launch (level 2).
    # VERIFY: the exact counter name; the bridge log line is
    #   "HabanaOp recipe cache :: adding new recipe to cache ::"
    rep.recipe_cache_additions = int(stats.get("recipe_cache_additions", 0))
    return rep


def arm_strict_mode(strict: bool = True) -> None:
    """Make a recompile a hard error, so a leak cannot pass silently.

    MEASURED: the source project arms `torch._dynamo.config.error_on_recompile` plus the
    backend's own `PT_HPU_DISALLOW_TORCH_COMPILE` (env var, must be set before process start).
    This catches level 1 only — see the docstring's naming trap.
    """
    import torch

    torch._dynamo.config.error_on_recompile = bool(strict)  # VERIFY: private config


def guard_expressions(top: int = 40) -> list:
    """Return the most frequent guard expressions.

    This is how you find an UNINTENTIONAL specialization key: a Python int that **varies
    between steps** is a bug to fix (make the dim static), not a case to warm.
    MEASURED: the source project reads these from `guard_added_fast` expressions.
    """
    import torch

    out = collections.Counter()
    counters = torch._dynamo.utils.counters  # VERIFY
    for bucket in ("guard_added_fast", "guard_added"):
        for expr, n in counters.get(bucket, {}).items():
            out[str(expr)] += int(n)
    return out.most_common(top)


# ---------------------------------------------------------------------------
# PART 2 — offline reduction of a profiler trace: count level-2 compiles.
# ---------------------------------------------------------------------------


def count_compile_graph_events(trace_path: str) -> dict:
    """Count `compileGraph` events in a torch profiler trace, per step.

    `compileGraph` is the SYNAPSE graph compile. It should be **zero** in a steady-state
    window. MEASURED: this is the instrument that exposed 93/step at level 2 while level 1
    was clean.

    Trace files are `<rank>.<ts>.pt.trace.json.gz`; the phase is NOT in the filename — decide
    by content (look for `forward_decode` vs `forward_prefill` python frames, or by kernel
    count).
    """
    if trace_path.endswith(".gz"):
        with gzip.open(trace_path, "rt") as fh:
            events = json.load(fh)["traceEvents"]
    else:
        with open(trace_path) as fh:
            events = json.load(fh)["traceEvents"]

    # Count steps from the phase marker, so "per step" means something.
    phase_frames = [
        e for e in events
        if e.get("cat") == "python_function" and e.get("ph") == "X"
        and str(e.get("name", "")).endswith(("forward_decode", "forward_prefill"))
    ]
    decode_frames = [e for e in phase_frames
                     if str(e.get("name", "")).endswith("forward_decode")]
    steps = max(1, len(decode_frames) // 43) if decode_frames else 1

    compiles = [e for e in events
                if e.get("ph") == "X" and str(e.get("name", "")) == "compileGraph"]

    # The recipe names the compile belongs to, when present.
    by_recipe = collections.Counter(
        str((e.get("args") or {}).get("recipeName", "")) for e in compiles
    )

    return {
        "trace": trace_path,
        "steps_assumed": steps,
        "compile_graph_events": len(compiles),
        "compile_graph_per_step": len(compiles) / steps,
        "top_recipes": by_recipe.most_common(10),
        "verdict": (
            "CLEAN at level 2"
            if not compiles
            else "LEVEL 2 NOT CLEAN: steady-state compilation is happening; any timing from "
                 "this window is invalid"
        ),
    }


# ---------------------------------------------------------------------------
# PART 3 — the assertions to run after warmup, before trusting any number.
# ---------------------------------------------------------------------------


def post_warmup_assertions(before: SpecializationReport, after: SpecializationReport,
                           trace_path: str | None = None) -> list:
    """Return a list of (name, ok, detail). All must be True before you trust a timing.

    MEASURED: each of these corresponds to a failure the source project actually hit.
    """
    delta = before.delta(after)
    results = [
        (
            "level 1 (recipe) clean after warmup",
            delta.unique_graphs == 0,
            f"unique_graphs grew by {delta.unique_graphs}",
        ),
        (
            "no recompiles after warmup",
            delta.recompiles == 0,
            f"recompiles grew by {delta.recompiles}",
        ),
        (
            "no new Synapse graphs at launch",
            delta.recipe_cache_additions == 0,
            f"recipe cache additions grew by {delta.recipe_cache_additions}",
        ),
    ]
    if trace_path:
        info = count_compile_graph_events(trace_path)
        results.append(
            (
                "level 2 (Synapse graph) clean in the trace",
                info["compile_graph_events"] == 0,
                f"{info['compile_graph_per_step']:.1f} compileGraph events/step",
            )
        )
    return results


if __name__ == "__main__":
    print(__doc__.split("TAGS")[0])

    print("--- in-process counters (requires torch + dynamo) ---")
    rep = snapshot()
    print("snapshot:", rep)
    if rep.notes:
        print("notes:", rep.notes)

    print("\n--- the assertions that gate a timing measurement ---")
    a = SpecializationReport(unique_graphs=100, recompiles=0)
    b = SpecializationReport(unique_graphs=194, recompiles=0)
    for name, ok, detail in post_warmup_assertions(a, b):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    print("\n  (the second one FAILS on purpose: 94 new graphs after warmup means the"
          "\n   specialization count is growing with the workload, not fixed by warmup.)")

    print("\n--- offline trace mode ---")
    print("  usage: count_compile_graph_events('path/to/<rank>.<ts>.pt.trace.json.gz')")
    print("  a non-zero per-step count means level 2 is still compiling.")
