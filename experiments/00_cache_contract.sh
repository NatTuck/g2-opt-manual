#!/usr/bin/env bash
# Rung 0 — the cache contract.
#
# Runs the smallest possible device program and then enumerates every file the
# process created ANYWHERE: under the experiment root, in the CWD, and in /tmp.
# The assertion is that the set is exactly $EXP_ROOT. Anything outside it is a
# writer we have not captured, and it must be added to env.sh before any later
# rung counts recipes or times anything.
#
#   RUNG=00 bash experiments/00_cache_contract.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNG="${RUNG:-00}"
# shellcheck source=env.sh
source "$HERE/env.sh"

MARK="$EXP_ROOT/.marker"
touch "$MARK"

cat > "$EXP_RUN/op.py" <<'PY'
import torch, torch_hpu
x = torch.arange(1024, dtype=torch.float32, device="hpu")
y = x + 1
torch.hpu.synchronize()
print(f"  y[0]={y[0].item()}  y[-1]={y[-1].item()}")
PY

echo "=== running the smallest device op under the contract" >&2
( cd "$EXP_RUN" && "$EXP_PYTHON" op.py ) 2>&1 | grep -v -i 'warning\|mark_step\|closure' | tail -3

echo
echo '=== every file created since the marker, under the root'
find "$EXP_ROOT" -type f -newer "$MARK" 2>/dev/null | sed "s#$EXP_ROOT#<root>#" | sort | head -40
echo
echo '=== created OUTSIDE the root (must be empty)'
{
  find "$EXP_RUN" "$EXP_REPO" /tmp -xdev -type f -newer "$MARK" 2>/dev/null \
    | grep -v "^$EXP_ROOT/" | grep -v '/\.cmds/' | grep -v 'shell-jobs' | grep -v "$MARK"
} > /tmp/outer-galliform-root/escaped.txt 2>/dev/null
if [ -s /tmp/outer-galliform-root/escaped.txt ]; then
  echo '  ESCAPED:'; sed 's/^/    /' /tmp/outer-galliform-root/escaped.txt | head -20
else
  echo '  (none)'
fi
echo
echo '=== counts under the root'
printf '  logs:    %s\n' "$(find "$EXP_LOGS" -type f 2>/dev/null | wc -l)"
printf '  dumps:   %s\n' "$(find "$EXP_DUMPS" -type f 2>/dev/null | wc -l)"
printf '  recipes: %s\n' "$(find "$EXP_RECIPE" -name '*.recipe' 2>/dev/null | wc -l)"
printf '  tmp:     %s\n' "$(find "$EXP_TMP" -type f 2>/dev/null | wc -l)"
printf '  inductor:%s\n' "$(find "$EXP_INDUCTOR" -type f 2>/dev/null | wc -l)"
