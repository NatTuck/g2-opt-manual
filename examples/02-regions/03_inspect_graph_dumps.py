#!/usr/bin/env python
"""Example 3 — reading a `DUMP_POST_GRAPHS` directory.

WHAT THIS DEMONSTRATES
----------------------
Chapter 2 section 2.9.  For structural questions -- which engine is this node on,
which recipe is it in, what did the compiler think it would cost -- the right
instrument is the compiler's own post-graph dump, not a profiler trace.  A trace
tells you what executed and when; the dump tells you what the compiler BUILT, per
node, before anything ran.

This file is CPU-only: it reads dumps that some earlier run produced.  Run it with
`--demo` to generate a tiny synthetic dump so you can see the output shape without
an HPU.

    ./venv/bin/python 03_inspect_graph_dumps.py --dump /tmp/gcpost
    ./venv/bin/python 03_inspect_graph_dumps.py --demo

WHAT TO LOOK FOR
----------------
  * per-node `engine` -- the compiler's own literal field: TPC / MME / DMA / None.
    This is engine identity, not an op-name heuristic (section 2.8, rule 3).
  * per-graph ENGINE COMPOSITION -- the set of engines in one recipe.  Two engines
    in one recipe is the precondition for overlap (section 2.2, condition 2).
  * `dma_expected_duration` -- the DMA cost MODEL, in microseconds.  It is a
    LOWER BOUND (optimistic by ~16 % at 64 MB, ~3 % at >= 256 MB).  Never treat it
    as a measured time (section 2.9, trap 3).
  * the filter cross-tab at the end -- the general habit that catches the bug where
    a filter does not test what it claims to.

THE TWO TRAPS THIS FILE DEFENDS AGAINST
---------------------------------------
1. `**`-globbing silently SKIPS the hidden `.graph_dumps/` directory.  One recon saw
   24 of 1,141 files on its first pass.  Use `os.walk`.
2. `mme_node_strategy` is a structured string.  PARSE IT BY FIELD, never by
   substring: `'Height=1' in s` is True for `'Height=128'`.  That exact bug
   produced a retracted figure in the source project.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import tempfile

DUMP_SUBDIR = ".graph_dumps"          # hidden -- this is why `**` globs miss it


# ---------------------------------------------------------------------------
# finding the files: os.walk, because the directory is HIDDEN
# ---------------------------------------------------------------------------
def find_dumps(root):
    """Every *.post.json under <root>/.graph_dumps (and any nested ones).

    `glob(root + '/**/*.post.json')` returns nothing here: `**` does not descend
    into dot-directories by default, and `.graph_dumps` is one.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".post.json"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


# ---------------------------------------------------------------------------
# the field-wise parse of the MME geometry string (trap 2)
# ---------------------------------------------------------------------------
def parse_strategy(s):
    """`mme_node_strategy` -> dict.  Field-wise, NEVER substring.

    The string looks like:
      MME Geometry: Name=2xh, Width=256, Height=512, Pattern=fck, Raster=True, ...
    Splitting on commas then on the FIRST '=' gives the fields.  The retired bug
    was `'Height=1' in s` (True for 'Height=128') -- a substring test cannot see
    field boundaries.
    """
    d = {}
    for part in (s or "").replace("MME Geometry:", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
    return d


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------
def census(paths):
    engines = collections.Counter()
    per_graph = {}                      # graph name -> Counter of engines
    dma_total = 0.0
    dma_nodes = 0
    mme_geom = collections.Counter()
    logical = collections.Counter()
    rows = []

    for p in paths:
        try:
            doc = json.load(open(p))
        except Exception:
            continue
        for g in doc.get("graphs", []):
            gn = g.get("name")
            c = per_graph.setdefault(gn, collections.Counter())
            tens = {t["name"]: t for t in g.get("tensors", [])}
            for nd in g.get("nodes", []):
                eng = nd.get("engine") or "None"
                engines[eng] += 1
                c[eng] += 1
                logical[bool(nd.get("is_logical"))] += 1
                if eng == "DMA":
                    dma_nodes += 1
                    dma_total += float(nd.get("dma_expected_duration") or 0.0)
                if eng == "MME":
                    f = parse_strategy(nd.get("mme_node_strategy"))
                    mme_geom[(f.get("Name"), f.get("Width"), f.get("Height"),
                              f.get("Pattern"))] += 1
                # operand shapes come from the graph's own `tensors` list
                ins = [tens.get(t, {}).get("max_shape") for t in nd.get("input_tensors", [])]
                outs = [tens.get(t, {}).get("max_shape") for t in nd.get("output_tensors", [])]
                rows.append({"file": p, "graph": gn, "engine": eng,
                             "guid": nd.get("guid"), "type": nd.get("type"),
                             "is_logical": bool(nd.get("is_logical")),
                             "exec_order_idx": nd.get("exec_order_idx"),
                             "bundle_index": nd.get("bundle_index"),
                             "origin_nodes": nd.get("origin_nodes"),
                             "in_shapes": ins, "out_shapes": outs})
    return dict(engines=engines, per_graph=per_graph, dma_total=dma_total,
                dma_nodes=dma_nodes, mme_geom=mme_geom, logical=logical, rows=rows)


def report(c):
    print("=" * 72)
    print("  DUMP_POST_GRAPHS census")
    print("=" * 72)
    print("  nodes by engine:", dict(c["engines"]))
    print("  is_logical:", dict(c["logical"]),
          "  (a wrapper is engine=None and is_logical=true -- do not count it as work)")
    print()
    print("  ENGINE COMPOSITION PER GRAPH (= per recipe).  Two engines in one")
    print("  graph is the precondition for overlap; one engine cannot overlap")
    print("  with itself (section 2.2, condition 2).")
    combos = collections.Counter(tuple(sorted(k for k in v if k != "None"))
                                 for v in c["per_graph"].values())
    for k, v in combos.most_common(12):
        print("    %-26s %d graph(s)" % (str(k), v))
    print()
    if c["dma_nodes"]:
        print("  DMA cost model: %d nodes, sum(dma_expected_duration) = %.1f us"
              % (c["dma_nodes"], c["dma_total"]))
        print("    REMINDER: this is a MODELLED LOWER BOUND, not a measured time.")
        print("    A measured window / this number near 1.0 validates; a ratio of 5")
        print("    does not mean the hardware is slow -- it means the model has no")
        print("    queue wait and no contention in it.")
    print()
    if c["mme_geom"]:
        print("  MME geometry, PARSED BY FIELD (Name, Width, Height, Pattern):")
        for k, v in c["mme_geom"].most_common(8):
            print("    %-34s %d node(s)" % (str(k), v))
        print("    Width/Height are the MME TILE, not the GEMM's M/N/K.  Read the")
        print("    operand shapes from `in_shapes`/`out_shapes` for those.")
    print()


def cross_tab(c):
    """The general habit: cross-tabulate a filter against the thing it claims.

    A real example from the source project: a derived 'row dimension M' was
    reported as absent for 3,896 of 6,901 nodes, and the conclusion drawn was
    'there are no M=1 nodes'.  Cross-tabulating revealed that 'M is absent' and
    'the output shape is not 2-D' were the SAME set -- i.e. the filter was testing
    the shape rank, not the dimension.  Two review rounds missed it because nobody
    cross-tabulated.
    """
    print("  FILTER CROSS-TAB (the habit that catches a filter that does not test")
    print("  what it claims): does 'engine == MME' coincide with anything else?")
    tab = collections.Counter()
    for r in c["rows"]:
        tab[(r["engine"] == "MME", bool(r["is_logical"]))] += 1
    print("    (engine==MME, is_logical): ", dict(tab))
    rank = collections.Counter()
    for r in c["rows"]:
        o = r["out_shapes"][0] if r["out_shapes"] else None
        rank[(r["engine"], len(o) if o else None)] += 1
    print("    (engine, len(out_shapes[0])): ", dict(rank))
    print("    If two of these coincide perfectly, your filter is a proxy for the")
    print("    other thing and is probably not testing what you think.")
    print()


# ---------------------------------------------------------------------------
# a tiny synthetic dump so the file is runnable without an HPU
# ---------------------------------------------------------------------------
def make_demo_dump(root):
    d = os.path.join(root, DUMP_SUBDIR)
    os.makedirs(d, exist_ok=True)      # <- mkdir -p first, ALWAYS (trap 3 below)
    doc = {"name": "demo", "version": 1, "graphs": [{
        "name": ".graph_dumps/graph_0000_fused_0_syn_1000000_1000000_123",
        "id": 0, "recipe_debug_id": 12345, "device": 0,
        "tensors": [
            {"name": "0", "max_shape": [4096, 384], "dtype": "bf16"},
            {"name": "1", "max_shape": [4096, 1536], "dtype": "bf16"},
            {"name": "2", "max_shape": [1536, 384], "dtype": "bf16"},
        ],
        "nodes": [
            {"engine": None, "guid": "split_logical", "type": "Split", "is_logical": True,
             "exec_order_idx": 0, "bundle_index": 0, "origin_nodes": [394],
             "input_tensors": ["0"], "output_tensors": ["0"]},
            {"engine": "DMA", "guid": "memcpy_dma", "type": "DmaMemcpy", "is_logical": False,
             "exec_order_idx": 1, "bundle_index": 0, "origin_nodes": [394],
             "dma_expected_duration": 6.5923,
             "input_tensors": ["0"], "output_tensors": ["0"]},
            {"engine": "MME", "guid": "gemm", "type": "GEMM", "is_logical": False,
             "exec_order_idx": 2, "bundle_index": 0, "origin_nodes": [394],
             "mme_expected_compute_cycles": 24576,
             "mme_node_strategy": ("MME Geometry: Name=2xh, Width=256, Height=512, "
                                   "Pattern=fck, Raster=True, BatchConcurrency=1"),
             "input_tensors": ["0", "1"], "output_tensors": ["2"]},
            {"engine": "TPC", "guid": "mult_fwd_bf16", "type": "mult", "is_logical": False,
             "exec_order_idx": 3, "bundle_index": 0, "origin_nodes": [395],
             "tpc_working_engines": 24,
             "input_tensors": ["2"], "output_tensors": ["2"]},
        ]}]}
    p = os.path.join(d, "graph_0000_fused_0_syn_1000000_1000000_123.0.12345.post.json")
    json.dump(doc, open(p, "w"))
    return root


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", help="a DUMP_POST_GRAPHS directory (or its parent)")
    ap.add_argument("--demo", action="store_true",
                    help="generate and inspect a tiny synthetic dump (no HPU needed)")
    a = ap.parse_args(argv)

    if a.demo:
        tmp = tempfile.mkdtemp(prefix="g2-demo-dump-")
        root = make_demo_dump(tmp)
        print("demo dump written under %s" % root)
    elif a.dump:
        root = a.dump
    else:
        ap.error("pass --dump DIR (or --demo)")

    paths = find_dumps(root)
    print("found %d *.post.json file(s) via os.walk" % len(paths))
    if not paths:
        print("Nothing found.  Two usual causes:")
        print("  * you pointed at the wrong root, or the run never set DUMP_POST_GRAPHS;")
        print("  * you used a `**` glob, which skips the hidden `.graph_dumps/`.")
        return 1

    c = census(paths)
    report(c)
    cross_tab(c)
    print("  Trap reminder: the dump directory must PRE-EXIST.  A DUMP_POST_GRAPHS")
    print("  pointing at a non-existent path is treated as a FILE, and concurrent")
    print("  writers then race and FAIL THE COMPILE (synStatus 26, 'size mismatch")
    print("  after 3 retries').  `mkdir -p` first, every time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
