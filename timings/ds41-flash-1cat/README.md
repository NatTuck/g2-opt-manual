# DeepSeek-V4.1-Flash on 4×Gaudi2 — the 1Cat vLLM fork

Two documents from one measurement campaign: a report of what the stack did, and the procedure to
rebuild and re-measure it.

| file | what it is |
|---|---|
| [`report.md`](report.md) | the measurements, the findings, what is open, and the limits of the report |
| [`replication.md`](replication.md) | one ordered path from a stock Habana 1.24.2 container to those numbers: build, weights, launch, protocol, gates, traps, and what is not reproducible as-is |

## Provenance

- **Host:** one 4 × HL-225 (Gaudi2) machine, TP4×PP1, inside a standard Habana 1.24.2 container.
- **Engine:** `NatTuck/1Cat-vLLM-Gaudi` @ `e2266e0` (a vLLM fork), vLLM `0.28.1rc1.dev392+ge47aa780b`.
- **Weights:** `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a`, prepared to 48 rank shards and 12
  sidecars.
- **When:** 2026-11-09/10.

## What these documents are not

- **Not a Gaudi2 performance claim.** Every rate is one host's result at one configuration. The
  transferable parts are the shapes and ratios; §6 of `replication.md` says which is which.
- **Not a checkable citation trail.** The campaign's raw cells, logs and per-run findings live in
  an unpublished working tree. Both documents reproduce the numbers in tables so that no claim
  rests only on a pointer, but the paths they name are provenance, not links.
- **Not a qualified configuration.** The fork's own contract for this model is single-request,
  greedy; batch above width 1 does not work at this configuration, and one kernel on the fused
  decode path has an out-of-bounds defect that the documents describe.

## Corrections in this copy

The campaign's internal record is host-specific (absolute paths, module ids, ports, seeds, every
void run). These copies are the publication: paths are parameterized as `$CASE`, device ids and
ports are parameters, resource figures are stated as requirements with the observed values marked,
and the claims carry the manual's basis tags. Where a figure appears in both documents, the report
is the source of truth and the replication document references it.
