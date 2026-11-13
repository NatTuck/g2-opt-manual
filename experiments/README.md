# Experiments: Python → dynamo → bridge → Synapse → HPU

A ladder of the smallest programs that show **how Python code becomes an executable HPU program, where
the artefacts live, and which layer owns which decision**. Each rung adds exactly one layer to the rung
below it, so "what did this layer do" is a diff rather than a claim.

Every rung runs under [`env.sh`](env.sh), which exists to guarantee one thing: **every byte the process
writes incidentally lands under `/tmp/outer-galliform-root/experiments/<rung>/`** — logs, caches, dumps,
temp files. The scripts live here; their artefacts do not.

```sh
RUNG=01 bash experiments/run.sh 01_one_region.py      # fresh root
FRESH=0 RUNG=01 bash experiments/run.sh 01_one_region.py   # reuse the root (cache-hit rungs)
RUNG=00 bash experiments/00_cache_contract.sh         # prove containment
```

## Rung 0 — the cache contract (`00_cache_contract.sh`, `env.sh`)

**What it does.** Points every incidental-write location at the rung root, runs the smallest possible
device op (`x + 1` on a 1024-element tensor), and then enumerates every file created since a marker —
under the root, in the CWD, and in `/tmp`.

**Why it exists.** Measured, with only the obvious variables set, **two writers escaped**:

| writer | default | now |
|---|---|---|
| Habana runtime/compiler logs | `/var/log/habana_logs` (read-only here) | `$HABANA_LOGS` |
| post-graph dumps | — | `DUMP_POST_GRAPHS` (one JSON per compiled graph, into `<dir>/.graph_dumps/`) |
| Synapse recipes | — | `PT_HPU_RECIPE_CACHE_CONFIG` |
| torch temp | `$TMPDIR` | `TMPDIR` |
| **inductor cache** | **`/tmp/torchinductor_$USER`** — ignores `TMPDIR` | `TORCHINDUCTOR_CACHE_DIR` |
| **graphviz graph dumps** | **`<cwd>/.graph_dumps`** — a *second* dump mechanism | CWD is inside the root; the mode is unset by default |

**A trap, measured.** `PT_HPU_GRAPH_DUMP_MODE=0` is **rejected** —
`Value '0' is not allowed. Allowed values: compile | eager | all` — and kills the process with a FATAL
ERROR. The variable must be left unset, or set to one of those three values.

**Result.** With the contract in place, everything lands under the root: 22 log files, 3 post-graph
dumps, 0 recipes (no `torch.compile` in this rung), 0 files outside.

## Rung 1 — one compiled region, one tensor size (`01_one_region.py`)

**The program.** A 1024-element `float32` tensor, `torch.compile(lambda t: t + 1)`, one call. The
question: **does one region become exactly one recipe at one tensor size?**

**A prerequisite, measured.** `torch.compile` **without** a backend picks **inductor** and dies:

```
torch._inductor.exc.InductorError: KeyError: 'hpu'
```

The HPU backend is registered by the plugin (`habana_frameworks.torch`) but is not the default, so the
region must name it: `torch.compile(fn, backend="hpu_backend")`.

**Result — the hypothesis holds.**

| | value |
|---|---|
| dynamo | `graph_count=1  break_count=0  op_count=1` |
| FX graph | one `placeholder → operator.add(%t, 1) → return` |
| **recipes** | **1** — `4169968267915495333_a16fb581eee_syn1.24.2.79e284b.recipe` |
| post-graph dumps | 5 — four `N.post.json` plus one `graph_0000_fused_0_syn_….post.json` |
| first call | **140.5 ms** (trace + bridge + compile) |
| second call, same process | **0.5 ms** (dispatch only) |

The recipe filename is the cache key made visible, and it has three parts:

```
4169968267915495333   a16fb581eee   syn1.24.2.79e284b
└── graph id ─────┘   └ code hash ┘  └ Synapse build ┘
```

The `graph_0000_fused_0_syn_…` dump is the graph that became the recipe; the four `N.post.json` dumps are
the individual ops around it.

**Cross-process reuse — measured.** A second *process* against the same cache root produced **the same
filename with the same mtime** — the recipe was reused, not rewritten — and the first call dropped from
**140.5 ms to 76.7 ms**. So on this path the recipe key is **stable across processes**: nothing
per-process enters it. (That is the property jchen's per-region code-object naming was suspected of
breaking; here, with no such naming, it holds.)

## What each rung is for

| rung | adds | question it answers |
|---|---|---|
| 0 | the contract | where does anything get written? |
| 1 | `torch.compile` + one shape | one region → one recipe? (**yes**) |
| 2 | a second tensor size | one recipe **per size**, and only one? |
| 3 | perturbations (dtype, int, kernel source, env) | what mints a new recipe and what reuses one? |
| 4 | knobs (mode, cache on/off, shape refinement, guard skip) | the control surface, as a table |
| 5 | a custom TPC kernel in the region | how a kernel's source digest enters the identity |

## Basis tags

As elsewhere in this manual: **MEASURED** (ran here, with the artefact named), **source read**,
**INFERRED**, **UNKNOWN**. Everything above marked "measured" was produced by these scripts on this
host, on 2026-11-13, with the artefacts under `/tmp/outer-galliform-root/experiments/`.
