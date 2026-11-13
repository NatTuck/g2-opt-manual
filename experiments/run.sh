#!/usr/bin/env bash
# Run one experiment under the cache contract, from a clean root, and report what
# it produced.
#
#   RUNG=01 bash experiments/run.sh 01_one_region.py
#
# FRESH=0 keeps the previous root (useful for the cache-hit rungs, where the whole
# point is that the second run finds what the first one wrote).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNG="${RUNG:?set RUNG (e.g. RUNG=01)}"
SCRIPT="${1:?usage: RUNG=<rung> bash experiments/run.sh <script.py>}"
FRESH="${FRESH:-1}"
EXP_ROOT="${EXP_ROOT:-/tmp/outer-galliform-root/experiments/$RUNG}"

if [ "$FRESH" = "1" ]; then
  rm -rf "$EXP_ROOT"
  echo "[run] fresh root: $EXP_ROOT" >&2
fi

# shellcheck source=env.sh
source "$HERE/env.sh"

cp "$HERE/$SCRIPT" "$EXP_RUN/" 2>/dev/null || cp "$SCRIPT" "$EXP_RUN/"
BASE="$(basename "$SCRIPT")"
MARK="$EXP_ROOT/.marker"; touch "$MARK"

echo "=== $RUNG / $BASE" >&2
( cd "$EXP_RUN" && "$EXP_PYTHON" "$BASE" ) 2>&1 \
  | grep -v -i 'mark_step\|add_step_closure\|iter_mark_step' | tail -25

echo
echo '=== artefacts under the root'
printf '  recipes:  %s\n' "$(find "$EXP_RECIPE" -name '*.recipe' 2>/dev/null | wc -l)"
printf '  dumps:    %s\n' "$(find "$EXP_DUMPS" -name '*.json' 2>/dev/null | wc -l)"
printf '  logs:     %s\n' "$(find "$EXP_LOGS" -type f 2>/dev/null | wc -l)"
echo '  --- recipe files:'
find "$EXP_RECIPE" -name '*.recipe' 2>/dev/null | sed "s#$EXP_RECIPE/##" | sort | sed 's/^/    /' | head -20
echo '  --- dump files:'
find "$EXP_DUMPS" -name '*.json' 2>/dev/null | sed "s#$EXP_DUMPS/##" | sort | sed 's/^/    /' | head -20
