#!/usr/bin/env bash
#
# Run the demo and the gate, with the environment the custom op needs.
#
# GC_KERNEL_PATH is a colon-separated list, and BOTH entries are load-bearing
# (09-tpc-kernels.md §9.5, measured):
#
#   * without our library, Synapse cannot resolve the GUID at all;
#   * without the stock library, the *eager* path still works but every
#     torch.compile region containing the op fails to compile, with an error
#     that names the graph and not the missing library.
#
# Do not add PT_HPU_RECIPE_CACHE_CONFIG to this script: on this stack a
# colon-separated GC_KERNEL_PATH is rejected outright as soon as that variable
# is set ("Flag GC_KERNEL_PATH: Path is not a regular file: <the whole string>",
# then an abort).
#
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

PYTHON=${PYTHON:-python3}
STOCK=${STOCK_KERNELS:-/usr/lib/habanalabs/libtpc_kernels.so}
export GC_KERNEL_PATH="$here/tpc_kernel_lib/build/libadd1_kernels.so:$STOCK"
export PT_HPU_LAZY_MODE=0

echo "== demo: $PYTHON add1_demo.py"
"$PYTHON" add1_demo.py

echo
echo "== gate: $PYTHON gate_add1.py"
"$PYTHON" gate_add1.py

echo
echo "upstream's pytest test, if pytest is installed:"
echo "  $PYTHON -m pytest test_add1_op.py -v"
