#!/usr/bin/env bash
# Source this before any experiment. It exists to make ONE promise:
#
#   every byte the experiment process writes incidentally lands under
#   $EXP_ROOT, and nowhere else.
#
# "Incidentally" means logs, caches, dumps, temp files — everything that is not
# the script itself. The scripts live in this repository; their artefacts do not.
#
# Measured 2026-11-13 (the probe that motivated this file): with only the obvious
# variables set, two directories still escaped — `/tmp/torchinductor_$USER` (it
# ignores TMPDIR) and a `.graph_dumps` directory relative to the CWD (a second,
# separate dump mechanism from DUMP_POST_GRAPHS). So this file sets both of those
# explicitly AND puts the CWD inside the root, so even a CWD-relative writer is
# contained.
#
# Usage:
#   RUNG=01 source experiments/env.sh
#   cd "$EXP_RUN" && python experiments/01_one_region.py
#
# After a run, `experiments/00_cache_contract.sh` proves containment by listing
# every file that appeared outside $EXP_ROOT.

: "${RUNG:?set RUNG (e.g. RUNG=01) before sourcing experiments/env.sh}"

EXP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/tmp/outer-galliform-root/experiments/$RUNG}"

# One directory per artefact kind, all under the root. DUMP_POST_GRAPHS requires
# its directory to exist already, so they are created here rather than by the
# experiment.
EXP_RUN="$EXP_ROOT/run"
EXP_LOGS="$EXP_ROOT/logs"           # Habana runtime + compiler logs
EXP_DUMPS="$EXP_ROOT/dumps"         # DUMP_POST_GRAPHS: one JSON per compiled graph
EXP_RECIPE="$EXP_ROOT/recipe-cache" # Synapse compiled recipes
EXP_TMP="$EXP_ROOT/tmp"
EXP_INDUCTOR="$EXP_ROOT/inductor"
EXP_XDG="$EXP_ROOT/xdg-cache"
EXP_HOME="$EXP_ROOT/home"           # anything that resolves ~/.cache
mkdir -p "$EXP_RUN" "$EXP_LOGS" "$EXP_DUMPS" "$EXP_RECIPE" \
         "$EXP_TMP" "$EXP_INDUCTOR" "$EXP_XDG" "$EXP_HOME"

# --- incidental writes: pointed at the root --------------------------------
export HOME="$EXP_HOME"
export TMPDIR="$EXP_TMP"
export XDG_CACHE_HOME="$EXP_XDG"
export TORCHINDUCTOR_CACHE_DIR="$EXP_INDUCTOR"   # ignores TMPDIR (measured)
export TORCH_COMPILE_DEBUG=0                     # would write torch_compile_debug/ in CWD
unset TORCH_TRACE 2>/dev/null || true            # only set when a trace is wanted

export HABANA_LOGS="$EXP_LOGS"
export PT_HPU_RECIPE_CACHE_CONFIG="$EXP_RECIPE,false,20480"
export DUMP_POST_GRAPHS="$EXP_DUMPS"

# The second dump mechanism: graphviz-style dumps, CWD-relative by default.
# Leave it UNSET by default -- measured: PT_HPU_GRAPH_DUMP_MODE=0 is rejected
# ("Value '0' is not allowed. Allowed values: compile | eager | all") and kills the
# process with a FATAL ERROR. A rung that wants graphviz dumps sets it to one of
# those values itself; the CWD is inside the root regardless.
if [ -n "${PT_HPU_GRAPH_DUMP_MODE:-}" ]; then
  export PT_HPU_GRAPH_DUMP_MODE
else
  unset PT_HPU_GRAPH_DUMP_MODE 2>/dev/null || true
fi

# --- the HPU runtime needs these -------------------------------------------
export GC_KERNEL_PATH="${GC_KERNEL_PATH:-/usr/lib/habanalabs/libtpc_kernels.so}"
export LD_PRELOAD="${LD_PRELOAD:-/lib/x86_64-linux-gnu/libtcmalloc.so.4}"

# --- what the experiment runs ----------------------------------------------
export EXP_PYTHON="${EXP_PYTHON:-/usr/bin/python3.12}"

echo "[env] RUNG=$RUNG  root=$EXP_ROOT" >&2
echo "[env] logs=$EXP_LOGS dumps=$EXP_DUMPS recipes=$EXP_RECIPE tmp=$EXP_TMP" >&2
