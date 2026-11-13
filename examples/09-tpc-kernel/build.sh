#!/usr/bin/env bash
#
# Build the add1 TPC kernel example from a clean checkout.
#
# Produces two artifacts, in this order (see 09-tpc-kernels.md §9.1):
#   1. tpc_kernel_lib/build/libadd1_kernels.so   -- device ELF + host glue
#   2. build/lib.linux-x86_64-*/hpu_custom_add1.*.so -- the PyTorch extension
#
# The only absolute paths are the toolchain's own defaults, and each can be
# overridden from the environment. Nothing here refers to a particular home
# directory.
#
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

TPC_CLANG=${TPC_CLANG:-/usr/bin/tpc-clang}
OBJCOPY=${OBJCOPY:-/usr/bin/objcopy}
CXX=${CXX:-/usr/bin/g++}
HABANATOOLS_INCLUDE=${HABANATOOLS_INCLUDE:-/usr/lib/habanatools/include}
PYTHON=${PYTHON:-python3}

out=tpc_kernel_lib/build
mkdir -p "$out"

echo "== 1/4  device ELF: tpc-clang -> $out/add1_fwd_f32_gaudi2.o"
"$TPC_CLANG" -Wall -Werror -march=gaudi2 -O2 \
    -I"$HABANATOOLS_INCLUDE" \
    -Itpc_kernel_lib/kernels/include \
    tpc_kernel_lib/kernels/gaudi2/add1_fwd_f32_gaudi2.c \
    -c -o "$out/add1_fwd_f32_gaudi2.o"

echo "== 2/4  embed the ELF as a binary object"
# The linker symbol name comes from the INPUT FILENAME, and the glue in
# src/gaudi2_src/add1_fwd_f32_gaudi2.cpp declares
#     _binary___add1_fwd_f32_gaudi2_o_start
# which is what objcopy mints for an input called "__add1_fwd_f32_gaudi2.o".
# Hence the copy to the double-underscore name. See 09-tpc-kernels.md §9.3.
cp "$out/add1_fwd_f32_gaudi2.o" "$out/__add1_fwd_f32_gaudi2.o"
(
    cd "$out"
    "$OBJCOPY" -I binary -O elf64-x86-64 -B i386:x86-64 \
        --rename-section .data=.rodata,alloc,load,readonly,data,contents \
        __add1_fwd_f32_gaudi2.o __add1_fwd_f32_gaudi2_kernel.o
)

echo "== 3/4  host glue: $out/libadd1_kernels.so"
"$CXX" -shared -fPIC -O2 \
    -I"$HABANATOOLS_INCLUDE" \
    -Itpc_kernel_lib/src/gaudi2_src \
    -std=c++17 -Wall -Wl,--allow-multiple-definition \
    tpc_kernel_lib/src/entry_points.cpp \
    tpc_kernel_lib/src/gaudi2_src/add1_fwd_f32_gaudi2.cpp \
    "$out/__add1_fwd_f32_gaudi2_kernel.o" \
    -o "$out/libadd1_kernels.so"

echo "== 4/4  PyTorch extension: setup.py build"
"$PYTHON" setup.py build

echo
echo "built:"
ls -l "$out/libadd1_kernels.so"
ls -l build/lib.linux-x86_64-*/hpu_custom_add1*.so
