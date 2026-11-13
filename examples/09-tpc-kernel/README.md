# `examples/09-tpc-kernel` — a complete FP32 `add 1` TPC kernel

The worked example for [chapter 9](../../09-tpc-kernels.md). It is a **complete, minimal, running**
custom TPC kernel for Gaudi 2: one device kernel, one host glue library, one PyTorch op.

* **device kernel** — `tpc_kernel_lib/kernels/gaudi2/add1_fwd_f32_gaudi2.c`
* **kernel GUID** — `add1_fwd_f32_gaudi2`
* **torch op** — `torch.ops.custom_op.custom_add1(input_fp32_hpu) -> output_fp32_hpu`

## Upstream

Vendored from **<https://github.com/NatTuck/add1-gaudi2>** (commit
`2220281d7dae0d7a8407ff0770c0ff378d1a4208`, 2026-07-27). The kernel's structure — the index-space
loop nest, the glue class, the entry points — follows Habana's `Habana_Custom_Kernel` examples
(<https://github.com/HabanaAI/Habana_Custom_Kernel>); `tpc_kernel_lib/kernels/include/kernel_config.h`
is copied from that repository (commit `06e73bbac4f50debaf67609a0dbd49aa87cb39a4`) with its copyright
and licence header intact.

## Build

```sh
./build.sh
```

Two artifacts come out, in this order:

| # | artifact | what it is |
|---|---|---|
| 1 | `tpc_kernel_lib/build/libadd1_kernels.so` | the TPC ELF, embedded as a binary object, linked with the host glue |
| 2 | `build/lib.linux-x86_64-*/hpu_custom_add1*.so` | the PyTorch extension that registers `custom_op::custom_add1` |

`build.sh` needs `tpc-clang`, `objcopy`, `g++` and a Python with `torch` + the HPU plugin. Each is
taken from the toolchain's default location and each can be overridden: `TPC_CLANG`, `OBJCOPY`, `CXX`,
`HABANATOOLS_INCLUDE`, `PYTHON`.

## Run

```sh
./run.sh
```

which is the demo and then the test:

```sh
export GC_KERNEL_PATH="$PWD/tpc_kernel_lib/build/libadd1_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so"
export PT_HPU_LAZY_MODE=0
python3 add1_demo.py
python3 gate_add1.py          # the bit-exact gate, no test framework needed
```

`GC_KERNEL_PATH` is a **colon-separated list and both entries are load-bearing**. Three measured
facts, all in chapter 9 §9.5:

* **without our library** — `GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so` alone — the GUID
  cannot be resolved and the op fails with `Graph compile failed … synStatus 26 [Generic failure] …
  Current graph is nullptr`, which names neither the kernel nor the path;
* **without the stock library** — our library alone — the *eager* path works, but every
  `torch.compile` region containing the op fails to compile, with an error that blames the graph;
* **do not add `PT_HPU_RECIPE_CACHE_CONFIG`** to a colon-separated `GC_KERNEL_PATH`: the bridge then
  rejects the list outright — `Flag GC_KERNEL_PATH: Path is not a regular file: <the whole colon-joined
  string>` — and the process aborts.

`test_add1_op.py` is upstream's pytest test and needs `pytest`, which is **not installed in this
container** (`pip` here cannot run: it walks a directory it may not read). `gate_add1.py` is the same
bit-exact check without a framework, so `run.sh` always has a gate to run.


## The two check scripts

Neither is part of upstream. Both exist because the chapter needs the check, not the claim.

```sh
# §9.4: does the GUID become a device event, and how many TPCs did the node get?
export GC_KERNEL_PATH="$PWD/tpc_kernel_lib/build/libadd1_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so"
export PT_HPU_LAZY_MODE=0
python3 check_guid_in_trace.py      # writes build/add1_trace.json

# §9.6: does the op compose under torch.compile on this stack?
python3 check_torch_compile.py
```

## Layout

```
examples/09-tpc-kernel/
├── build.sh                      # device ELF -> embed -> glue .so -> torch extension
├── run.sh                        # demo + test, with the environment set
├── add1_demo.py                  # 64 fives in, 64 sixes out
├── test_add1_op.py               # bit-exact gate against a CPU reference
├── check_guid_in_trace.py        # the GUID as a device event; per-TPC census
├── check_torch_compile.py        # eager vs torch.compile, plus a control arm
├── custom_add1.py                # loads the extension, nn.Module wrapper
├── hpu_custom_add1.cpp           # the op: registration + call
├── setup.py                      # cpp_extension build, links habana_pytorch2_plugin
├── __init__.py
└── tpc_kernel_lib/
    ├── CMakeLists.txt            # alternative CMake build of artifact 1
    ├── kernels/
    │   ├── gaudi2/add1_fwd_f32_gaudi2.c   # THE DEVICE KERNEL
    │   └── include/kernel_config.h        # VECTOR/VECTOR_SIZE/v_mov_s for this dtype
    └── src/
        ├── entry_points.cpp               # GetKernelGuids / InstantiateTpcKernel
        └── gaudi2_src/
            ├── add1_fwd_f32_gaudi2.cpp    # the glue: index space + access pattern + ELF copy
            └── add1_fwd_f32_gaudi2.hpp
```

## What was changed from upstream

Upstream's README carried absolute paths from one machine (`$HOME/FastQwen/...` and a virtualenv
that does not exist) and its kernel `#include "kernel_config.h"` resolved only through one of them.
The changes here are:

1. **`README.md` rewritten** — this file. No absolute paths, and the two artifacts are named.
2. **`build.sh` and `run.sh` added** — upstream documented the commands inline in prose.
3. **`kernel_config.h` vendored** at `tpc_kernel_lib/kernels/include/`. It is *not* part of the
   `habanatools` package, which ships only `tpc_kernel_lib_interface.h`, `gc_interface.h` and the
   test-core headers, so the kernel cannot compile without it.
4. **`tpc_kernel_lib/CMakeLists.txt`** — the kernel include directory is now
   `${CMAKE_SOURCE_DIR}/kernels/include` instead of the absolute path.
5. **`check_guid_in_trace.py`, `check_torch_compile.py` added** — the two verifications the chapter
   quotes.
6. **`run.sh` uses `setup.py build`, not `setup.py install --user`** — upstream's install wrote into
   `$HOME/.local`; `custom_add1.py` already globs `build/lib.linux-x86_64-*/`, so an in-tree build is
   enough and nothing is installed.

The kernel source, the glue, the entry points, `hpu_custom_add1.cpp`, `setup.py`, `custom_add1.py`,
`add1_demo.py` and `test_add1_op.py` are upstream's, byte for byte.

## Requirements

* `tpc-clang` (TPC C/C++ compiler 2.7.33 here), `objcopy`, `g++` with C++17
* `/usr/lib/habanatools/include/` — `tpc_kernel_lib_interface.h`, `gc_interface.h`
* a Python with `torch` + the Habana plugin, `pybind11`, and `/usr/include/habanalabs/` for
  `perf_lib_layer_params.h`
* `PT_HPU_LAZY_MODE=0` — the plugin here is eager-only (lazy-mode APIs are present and warn that they
  do nothing)
