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
* **without the stock library** — our library alone — the *eager* path works, but **every**
  `torch.compile` region fails to compile, *including one that contains no custom op at all*: a plain
  `t * 2.0` region dies with `synStatus 26 … recipe_manager.cpp:304 … Can not compile graph`. The
  stock library is needed for any compile on this stack, and the error blames the graph. This is the
  most misleading failure in the example — `check_torch_compile.py`'s control arm is what catches it;
* **do not *enable* the disk recipe cache** alongside a colon-separated `GC_KERNEL_PATH`. A non-empty
  cache directory makes the bridge validate the list as a single file path, reject it
  (`Flag GC_KERNEL_PATH: Path is not a regular file: <the whole colon-joined string>`) and abort with
  exit **134**. The default value `,false,1024,false` has an empty path and is harmless — it is the
  *enabled* cache that collides, not the variable's presence.

One more, and it is the opposite of what the variable names suggest: the disk cache's filename is
`<graph hash>_a16fb581eee_syn<version>.recipe`, and that middle field is a **constant**. Editing the
kernel and rebuilding does not invalidate the cache — the chapter **infers** that you are then served
the stale recipe (§9.5.3, which tags it INFERRED and says what would settle it). Keep one cache
directory per build, or set `delete_on_init`. Chapter 9 §9.5.3 has the source and the measurements.

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
`diff -r` against upstream at `2220281` gives exactly **three modified files, five added files and one
added directory** — a complete accounting, so nothing in this tree is unattributed:

| | file | change |
|---|---|---|
| **modified** | `README.md` | rewritten as this file: no absolute paths, and the two artifacts are named |
| **modified** | `.gitignore` | the ignore list is grouped (`# artifacts`, `# runtime dumps`) and `.traces/` is added |
| **modified** | `tpc_kernel_lib/CMakeLists.txt` | `TPC_INCLUDE_DIR_KERNELS` is `${CMAKE_SOURCE_DIR}/kernels/include` instead of the absolute path inside one developer's checkout |
| **added** | `build.sh` | upstream documented the four build commands inline in prose |
| **added** | `run.sh` | the environment, then demo + gate. It uses `setup.py build`, not `setup.py install --user`: upstream's install wrote into `$HOME/.local`, and `custom_add1.py` already globs `build/lib.linux-x86_64-*/`, so an in-tree build is enough and nothing is installed |
| **added** | `check_guid_in_trace.py` | §9.4's verification — the GUID as a device event, and the per-TPC census |
| **added** | `check_torch_compile.py` | §9.6's verification — eager vs `torch.compile`, plus a control arm |
| **added** | `gate_add1.py` | 57 lines. The bit-exact gate as a standalone program: `pytest` is not installed in this container, and a gate is what has to run everywhere |
| **added** | `tpc_kernel_lib/kernels/include/kernel_config.h` | vendored from `HabanaAI/Habana_Custom_Kernel` @ `06e73bba`, licence header intact. It is *not* part of the `habanatools` package, which ships only `tpc_kernel_lib_interface.h`, `gc_interface.h` and the test-core headers, so the kernel cannot compile without it |

**Everything else is upstream's, byte for byte** — checked with `cmp` against upstream at `2220281`:
`__init__.py`, `add1_demo.py`, `custom_add1.py`, `hpu_custom_add1.cpp`, `setup.py`, `test_add1_op.py`,
`tpc_kernel_lib/kernels/gaudi2/add1_fwd_f32_gaudi2.c`, `tpc_kernel_lib/src/entry_points.cpp`,
`tpc_kernel_lib/src/gaudi2_src/add1_fwd_f32_gaudi2.cpp` and
`tpc_kernel_lib/src/gaudi2_src/add1_fwd_f32_gaudi2.hpp`.

## Requirements

* `tpc-clang` (TPC C/C++ compiler 2.7.33 here), `objcopy`, `g++` with C++17
* `/usr/lib/habanatools/include/` — `tpc_kernel_lib_interface.h`, `gc_interface.h`
* a Python with `torch` + the Habana plugin, `pybind11`, and `/usr/include/habanalabs/` for
  `perf_lib_layer_params.h`
* `PT_HPU_LAZY_MODE=0` — the plugin here is eager-only (lazy-mode APIs are present and warn that they
  do nothing)
