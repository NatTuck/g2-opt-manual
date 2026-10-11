#!/usr/bin/env python3
"""dump_post_graphs.py -- read per-node ENGINE identity out of the compiler's own graph dump.

WHY THIS EXISTS
---------------
"Does this recipe use more than one engine?" is the question the whole overlap story turns on,
and it used to be *inferred* from a launch count plus a shared recipe handle. It is not inferred
any more: the graph compiler will write one JSON per compiled graph, and **every node in it
carries a literal `"engine"` field**.

HOW TO TURN IT ON
-----------------
    mkdir -p /tmp/my-graph-dump            # <-- MUST pre-exist; see GOTCHA 1 below
    DUMP_POST_GRAPHS=/tmp/my-graph-dump \
    LOG_LEVEL_ALL=2 \
    python your_script.py

`LOG_LEVEL_ALL=2` on its own is worth having too: it prints, per recipe, the arc jobs by engine
name -- `(for 0[TPC])`, `(for 1[MME])`, `(for 2[DMA])`. That is the cheapest independent
confirmation that a recipe is multi-engine, and it needs no dump directory at all.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
  * the per-recipe engine multiset: `{DMA: 3, MME: 3, TPC: 1}` means one recipe is using three
    engines. That is the thing this script exists to show.
  * nodes with `"engine": "None"` -- these are the LOGICAL WRAPPERS (`Split`, `Concatenate`,
    `StaticReshape`, `Identity`). They are not engine work and must not be counted as such.
  * `origin_nodes` and `bundle_index` -- see the manual; a naive node count over-counts.
"""

import json
import os
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------------------
# GOTCHA 1: DUMP_POST_GRAPHS must point at a directory that ALREADY EXISTS. If it does not,
#           the compiler treats the path as a *file*, concurrent writers race, and the compile
#           fails (a size-mismatch / rename failure, surfacing as a generic compile error).
#           `mkdir -p` first. This is not a hypothetical -- it killed a run.
#
# GOTCHA 2: the dumps land in a HIDDEN directory (`.graph_dumps/`). A shell glob like
#           `*.json` or Python's `glob.glob("**/*.json")` SKIPS dot-directories, so you can
#           silently see a small fraction of the files -- one recon saw 24 of 1,141.
#           Use os.walk, which does not skip them.
# ---------------------------------------------------------------------------------------

FIELDS = ("engine", "guid", "type", "is_logical", "origin_nodes",
          "exec_order_idx", "bundle_index", "operation_index")


def find_dumps(root):
    """Every *.json under root, INCLUDING hidden directories (gotcha 2)."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def nodes_of(doc):
    """The dump nests graphs -> nodes; tolerate a bare node list too."""
    if isinstance(doc, dict) and "graphs" in doc:
        for g in doc["graphs"]:
            yield from g.get("nodes", [])
    elif isinstance(doc, dict) and "nodes" in doc:
        yield from doc["nodes"]


def main(root):
    files = find_dumps(root)
    if not files:
        print(f"no dumps under {root!r} -- did DUMP_POST_GRAPHS point at an existing dir? "
              f"(gotcha 1), and is the run actually compiling graphs?")
        return 1
    print(f"{len(files)} dump file(s) under {root}\n")

    per_recipe = defaultdict(Counter)     # dump file -> Counter(engine)
    field_coverage = Counter()            # how often each field is present
    n_nodes = 0

    for path in files:
        try:
            doc = json.load(open(path))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        engines = Counter()
        for node in nodes_of(doc):
            n_nodes += 1
            for f in FIELDS:
                if f in node:
                    field_coverage[f] += 1
            engines[str(node.get("engine"))] += 1
        per_recipe[os.path.basename(path)] = engines

    # --- The headline: which recipes are multi-engine?
    print(f"{'recipe dump':<52s} {'nodes':>6s}  engines")
    print("-" * 96)
    multi = 0
    for name, engines in sorted(per_recipe.items()):
        # "None" is the logical wrappers; count only real engine work.
        real = {k: v for k, v in engines.items() if k not in ("None", "null")}
        if len(real) > 1:
            multi += 1
        shown = ", ".join(f"{k}:{v}" for k, v in sorted(engines.items()))
        print(f"{name[:52]:<52s} {sum(engines.values()):>6d}  {shown}")
    print("-" * 96)
    print(f"{multi} of {len(per_recipe)} recipes use MORE THAN ONE engine.\n")

    # --- Field coverage: which cost/structure fields the compiler actually emits.
    print("field coverage across %d nodes (a field absent here is absent for THIS run):" % n_nodes)
    for f in FIELDS + ("tpc_working_engines", "mme_expected_compute_cycles",
                       "mme_node_strategy", "dma_expected_duration", "fused_node_graph"):
        print(f"  {f:<30s} {field_coverage[f]:>6d} / {n_nodes}")

    print("""
HOW TO READ THIS
----------------
* A recipe showing TPC + MME + DMA is the multi-engine case. In the source project 989 of
  4,765 recipe handles carried all three engines; multi-engine recipes are the NORM, not the
  exception. So "which engine does this recipe use?" is usually the wrong question.
* `engine: None` nodes are logical wrappers. Excluding them is correct.
* DO NOT count nodes as a proxy for engine work. Two measured traps:
    (a) `is_logical=true` marks ONLY the wrappers (all engine=None);
    (b) `origin_nodes` is many-to-one in BOTH directions -- one origin node can expand into
        nine graph nodes, while a fused TPC node carries three origins -- and `bundle_index`
        records ONE IR node becoming several physical engine tasks.
  One IR node is not one engine task.
* `dma_expected_duration` and `mme_expected_compute_cycles` are the compiler's own estimates.
  They are useful, but they have DIFFERENT error signs -- see the manual's cost-model section
  before using either as a reference.""")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/my-graph-dump"))
