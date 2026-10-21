#!/usr/bin/env python3
"""attribute_chain.py -- walk a profile back to source lines, through recipes.

THE CHAIN, and the key each leg uses:

    device event --(recipeId)--> recipe --(recipeName)--> post-graph dump --> node types/engines
        ^
        |
    launch_recipe (inside the region span) --(1:1 pairing)--> enqueueWithExternalEventsExt
        |
        +--(containment in a `Torch-Compiled Region` span)--> compile id --(TORCH_TRACE)--> file:line:func

WHAT IS A KEY AND WHAT IS AN INFERENCE (this is the whole point of the script):

  * `enqueueWithExternalEventsExt` carries `recipeHandle`, `recipeId` and `recipeName`. That is a
    real key: recipe -> trace events, and recipe -> dump, are exact.
  * `launch_recipe` is on the Python thread and is contained in the region span, but carries
    **no args at all**. So region -> recipe needs a PAIRING between the two host streams, and the
    pairing is only sound when they are 1:1. This script tests that and says so.
  * A device event is ASYNCHRONOUS and does not sit inside the region span that enqueued it.
    Attribute the LAUNCH, never the device event. (Measured on a 3-block program: attributing
    device events by containment leaves 3551 of 4309 outside every region; the launches, 0.)

USAGE
    attribute_chain.py <trace.json.gz> --spec <TORCH_TRACE dir> --dumps <DUMP_POST_GRAPHS dir>
    attribute_chain.py --selftest
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import sys

REGION_PREFIX = "Torch-Compiled Region: "
MARKER_TO_ENGINE = {
    "TPC_SPU_START_TO_SPU_HALT": "TPC",
    "MMEH_WB1_MON_TS_BIT1": "MME",
    "MMEH_WB0_MON_TS_BIT1": "MME",
    "DBG_DMA_TRC_WR_DATA_LAST": "DMA",
    "DBG_DMA_TRC_WR_FRST_ADDR_PUSH": "DMA",
    "DBG_DMA_TRC_RD_FRST_ADDR_PUSH": "DMA",
}


def engine_of(event):
    return MARKER_TO_ENGINE.get((event.get("args") or {}).get("HW event name"))


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path) as fh:
        return json.load(fh)["traceEvents"]


def load_spec(spec_dir):
    """{compile_id: 'file:line:function'} from a TORCH_TRACE directory."""
    labels = {}
    for path in sorted(glob.glob(os.path.join(spec_dir, "*"))):
        if not os.path.isfile(path):
            continue
        with open(path, errors="ignore") as fh:
            for line in fh:
                i = line.find("{")
                if i < 0:
                    continue
                try:
                    rec = json.loads(line[i:])
                except Exception:
                    continue
                meta = rec.get("compilation_metrics")
                if isinstance(meta, dict) and meta.get("compile_id") and meta.get("co_name"):
                    labels[str(meta["compile_id"])] = (
                        f"{meta.get('co_filename')}:{meta.get('co_firstlineno')}:{meta.get('co_name')}")
    return labels


def load_dumps(dump_dir):
    """{recipe_debug_id: (filename, node type counts, sorted node types)}.

    os.walk, NOT glob("**"): the dumps land in a DOT directory (`.graph_dumps`) and a `**` glob
    skips hidden directories. That is not hypothetical -- the first version of this script
    reported "0 recipes" against a directory holding 6.
    """
    out = {}
    files = []
    for root, _dirs, names in os.walk(dump_dir):
        files.extend(os.path.join(root, n) for n in names if n.endswith(".json"))
    for path in files:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        graphs = []
        if isinstance(d, dict) and "graphs" in d:
            graphs = d["graphs"]
        elif isinstance(d, list) and d and isinstance(d[0], dict):
            graphs = [{"recipe_debug_id": d[0].get("recipe_debug_id"), "nodes": d[1:]}]
        for g in graphs:
            rid = g.get("recipe_debug_id")
            if rid is None:
                continue
            types = [n.get("type") for n in (g.get("nodes") or [])]
            out[int(rid)] = (os.path.basename(path), collections.Counter(types), sorted(set(types)))
    return out


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def innermost(regions, ts):
    """Narrowest region span containing ts."""
    best = None
    for ts0, ts1, cid in regions:
        if ts0 <= ts <= ts1 and (best is None or (ts1 - ts0) < (best[1] - best[0])):
            best = (ts0, ts1, cid)
    return best[2] if best else None


def preflight(events, regions, dev):
    """Refuse to attribute a trace that cannot be attributed. Loudly."""
    if dev and not regions:
        print("\n*** THIS TRACE CANNOT BE ATTRIBUTED TO CODE ***")
        print(f"    {len(dev)} device events, ZERO `Torch-Compiled Region` spans.")
        print("    `Torch-Compiled Region` is a CPU-side dynamo annotation, so an HPU-only")
        print("    capture drops exactly the field that carries the compile id. The run is a")
        print("    timing instrument, not an attribution instrument.")
        print("    Fix: re-run with ProfilerActivity.CPU included (and TORCH_TRACE set BEFORE")
        print("    the run -- the compile-id -> source map is written by the run, not after it).")
        return False
    return True


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dumps", required=True)
    args = ap.parse_args(argv[1:])

    labels = load_spec(args.spec)
    dumps = load_dumps(args.dumps)
    events = load_trace(args.trace)

    regions = [(e["ts"], e["ts"] + e.get("dur", 0), str(e["name"])[len(REGION_PREFIX):])
               for e in events
               if str(e.get("name", "")).startswith(REGION_PREFIX) and e.get("ph") == "X"]
    raw_dev = [e for e in events
               if e.get("cat") in ("kernel", "hpu_op") and e.get("ph") == "X" and e.get("dur")]
    # An event with no `HW event name` is a HOST runtime call that merely shares the hpu_op
    # category (enqueue, eventRecord, synchronizeEvent, profilingProlog, allocateDeviceMemory).
    # Counting it as device work inflates every engine bucket -- measured: a 64 ms profilingProlog.
    dev = [e for e in raw_dev if engine_of(e)]
    host = [e for e in raw_dev if not engine_of(e)]

    print(f"trace      {args.trace}")
    print(f"  region spans {len(regions)}   compile ids {len({c for _, _, c in regions})}")
    print(f"  device events (engine-labelled) {len(dev)}   host runtime events {len(host)}")
    print(f"  spec labels {len(labels)}   dump recipes {len(dumps)}")
    if not preflight(events, regions, dev):
        return 1

    launches = sorted((e for e in events
                       if e.get("ph") == "X" and e.get("name") == "launch_recipe"), key=lambda e: e["ts"])
    enq = sorted((e for e in events
                  if e.get("ph") == "X"
                  and str(e.get("name", "")).startswith("enqueueWithExternalEventsExt")),
                 key=lambda e: e["ts"])

    # ---- region -> recipe: try the 1:1 pairing first, and PROVE it before using it -------------
    method, pair_map = None, {}
    if len(enq) == len(launches) and launches:
        cand = collections.defaultdict(collections.Counter)
        for lr, ev in zip(launches, enq):
            cid = innermost(regions, lr["ts"])
            rid = (ev.get("args") or {}).get("recipeId")
            if cid is not None and rid:
                cand[cid][int(rid)] += 1
        # "diagonal" = every compile id resolves to exactly one recipe and vice versa
        diag = all(len(c) == 1 for c in cand.values())
        seen = [next(iter(c)) for c in cand.values()]
        diag = diag and len(seen) == len(set(seen))
        if diag:
            method = "key join, 1:1 verified"
            pair_map = {cid: next(iter(c)) for cid, c in cand.items()}

    if method is None:
        # Fall back to compile order: compile index N <-> graph_<N:04d>_* dump <-> its recipe id.
        gidx = {}
        for rid, (fname, _c, _u) in dumps.items():
            if fname.startswith("graph_"):
                try:
                    gidx[int(fname.split("_")[1])] = rid
                except (IndexError, ValueError):
                    pass
        if gidx:
            method = "ORDER-BASED (compile index -> graph_NNNN -> recipe)"
            for _ts, label, cid in ((lr["ts"], None, innermost(regions, lr["ts"])) for lr in launches):
                if cid is None or "/" not in cid:
                    continue
                rid = gidx.get(int(cid.split("/")[0]))
                if rid is not None:
                    pair_map[cid] = rid
        else:
            method = "AMBIGUOUS -- no pairing available"

    print(f"\n=== region -> recipe: {method}")
    print(f"    {len(launches)} launch_recipe, {len(enq)} enqueueWithExternalEventsExt")
    if method.startswith("AMBIGUOUS"):
        print("    The two host streams are not 1:1 and no graph dump names a compile index, so the")
        print("    region->recipe leg cannot be established from this trace. Emitting nothing rather")
        print("    than a plausible-looking table.")

    # ---- assemble: code -> recipe -> engine unions --------------------------------------------
    dev_by_recipe = collections.defaultdict(list)
    for e in dev:
        rid = (e.get("args") or {}).get("recipeId")
        if rid is not None:
            dev_by_recipe[int(rid)].append(e)

    print(f"\n=== which code produced the device time (per-engine UNION, ms)")
    print(f"{'source location':58s} {'engine':6s} {'union ms':>9s} {'events':>7s} {'recipe':>8s}")
    rows = []
    for cid, rid in pair_map.items():
        label = labels.get(cid, f"<unlabeled {cid}>")
        per = collections.defaultdict(list)
        for e in dev_by_recipe.get(rid, []):
            per[engine_of(e)].append((e["ts"], e["ts"] + e["dur"]))
        for eng, iv in per.items():
            rows.append((label, eng, sum(b - a for a, b in merge(iv)) / 1000, len(iv), rid))
    for label, eng, ms, n, rid in sorted(rows, key=lambda r: -r[2]):
        print(f"{label:58s} {eng:6s} {ms:9.3f} {n:7d} {rid:8d}")

    print("\n=== leg 3 (a real key join): recipe -> dump -> nodes")
    for cid, rid in sorted(pair_map.items(), key=lambda kv: kv[1]):
        fname, counts, _uniq = dumps.get(rid, (None, None, None))
        label = labels.get(cid, f"<unlabeled {cid}>")
        print(f"\n{label}   (compile id {cid})")
        print(f"   recipeId {rid}   dump {fname or '<not dumped>'}")
        if counts:
            print(f"      node types: {dict(counts)}")

    # ---- verification against what the code was written to do ---------------------------------
    print("\n=== verification against what the code SHOULD produce")
    expect = {"mme_block": "MME", "tpc_block": "TPC", "dma_block": "DMA"}
    ok = True
    for cid, rid in sorted(pair_map.items()):
        fn = labels.get(cid, "").rsplit(":", 1)[-1]
        want = expect.get(fn)
        per = collections.defaultdict(list)
        for e in dev_by_recipe.get(rid, []):
            per[engine_of(e)].append((e["ts"], e["ts"] + e["dur"]))
        # UNION, not sum: summing is inflated by per-TPC replication (400 DMA events against
        # 160 MME events made DMA look larger on a block that is MME-bound).
        union = {k: sum(b - a for a, b in merge(v)) / 1000 for k, v in per.items()}
        top = max(union, key=union.get) if union else "?"
        hit = (want is None) or (top == want)
        ok &= hit
        print(f"   {fn:12s} code says {str(want):5s} -> profile says {top:5s} "
              f"{'OK' if hit else 'MISMATCH'}   (UNION ms: "
              f"{ {k: round(v, 3) for k, v in sorted(union.items())} })")
    if not pair_map:
        print("   (skipped -- no region->recipe pairing)")
        return 1
    print("\n" + ("PASS -- engine attribution matches the code's intent."
                  if ok else "FAIL -- attribution disagrees with the code; do not trust the table."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
