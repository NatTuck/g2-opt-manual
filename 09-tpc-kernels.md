# 9. Writing and shipping a TPC kernel

**Status: draft.** Tags follow the manual's convention — **MEASURED** / **source read** / **ASSUMED** /
**INFERRED** / **UNKNOWN** ([README](README.md#how-to-read-this-manual)). Every number below is restated
from the artifact beside it, and each carries the conditions it is valid under.

**What this chapter is.** The **workflow** of shipping a hand-written TPC kernel: the two artifacts, the
compiler invocation, the `objcopy` embed, the host glue, how Synapse finds the kernel, how to expose it to
PyTorch, how to test it, and what goes wrong. The programming model — VLIW bundles, vector registers,
dtypes, the index space as the work-division unit — is [chapter 4](04-execution-model.md), and **this
chapter cross-references it rather than restating it**. It assumes [§4.2.9](04-execution-model.md) (the
index-space point is the cost unit), [§4.3.1](04-execution-model.md) (one 256-byte register, 64 lanes for
`f32`), [§2.3–§2.4](02-regions.md) (the `custom_op` namespace short-circuit, which is why §9.7 exists) and
[§7.4](07-sram.md) (a TPC pointer is 32 bits, so a literal address compiles to a *local* access).

**It is chapter 9 because the manual appends.** [Chapter 4](04-execution-model.md)'s preface states the
rule: *"The material is the layer under chapter 1, but renumbering the three existing chapters would
rewrite their cross-references and their history. The manual appends; it does not renumber."* Nine is the
next free number, and the position is honest rather than chronological.

**The worked example is in this repo and it runs.** [`examples/09-tpc-kernel/`](examples/09-tpc-kernel/) is
an FP32 `add 1` kernel, GUID `add1_fwd_f32_gaudi2`, exposed as `torch.ops.custom_op.custom_add1`, vendored
from <https://github.com/NatTuck/add1-gaudi2> (commit `2220281`, 2026-07-27). **Every paste below is that
example's real output**, on the container this chapter was written in: Gaudi 2 `HL-225` modules `4,5`,
driver `1.24.2-52a555d`, firmware `hl-gaudi2-1.24.0-fw-62.6.2-sec-11`, `tpc-clang` 2.7.33 (clang 14.0.5),
`torch` 2.13.0a0+git2469d18, Python 3.12. Where a claim is instead read out of a source file it says so and
gives the file and line. **Where I could not make something work, §9.5 and §9.10 say so.**

---

## 9.1 The two artifacts you must produce

A TPC kernel reaches the device as **two** artifacts, built in this order, and they fail differently. A
third — the PyTorch extension — is optional for the device and required for the op.

| # | artifact | what it contains | failure if wrong |
|---|---|---|---|
| **1** | **device ELF** `add1_fwd_f32_gaudi2.o` | the TPC machine code: one `main`, one `tpc_metadata`, one `tpc_compiler` section | compile error (loud), or a wrong access in the listing (silent — §9.9) |
| **2** | **host glue library** `libadd1_kernels.so` | the ELF **embedded as a byte array**, plus `GetKernelGuids` / `InstantiateTpcKernel` / `GetShapeInference` | links clean, runs with a generic error (§9.3, §9.5) |
| **3** | **torch extension** `hpu_custom_add1…so` | the `TORCH_LIBRARY(custom_op, …)` registration and the call into the Habana op registry | import or dispatch error, at the Python layer |

**The dependency is one-way and total**: artifact 2 cannot exist without 1 (it embeds its bytes), and
artifact 3 is useless without 2 (it names a GUID only 2 can supply). **Nothing in the chain is checked by the
next link** — `g++` links a glue library whose embedded symbol is missing (§9.3), and Synapse accepts a GUID
whose kernel never runs (§9.5).

In the example: artifact 1 is `tpc_kernel_lib/kernels/gaudi2/add1_fwd_f32_gaudi2.c` → `tpc_kernel_lib/build/`;
artifact 2 is `tpc_kernel_lib/src/` (entry points + glue) → the same `build/`; artifact 3 is
`hpu_custom_add1.cpp` + `setup.py` → `build/lib.linux-x86_64-cpython-312/`; `custom_add1.py` is the Python
face.

The sizes on this container: device ELF **14,544 bytes** (`0x38d0`), glue library **28,832 bytes**, torch
extension **10,959,632 bytes** (it embeds debug info, so it moves with the build path). **MEASURED** —
§9.3's build paste.

> **The glue library is not a library in the usual sense.** It exports **three** functions and nothing else
> matters: `GetKernelGuids`, `InstantiateTpcKernel`, `GetShapeInference` (`nm -D --defined-only`,
> **MEASURED**). Synapse `dlopen`s it, asks for its GUID list, and calls `InstantiateTpcKernel` per node.
> **There is no symbol per kernel: one library holds many kernels**, and the example's entry points are a
> one-element dispatch table you extend by adding a class and a `strcmp` branch.

---

## 9.2 The device side, annotated from the vendored kernel

The kernel is 83 lines. This section walks it against what `tpc-clang` emits, because the correspondence
between the two is the thing that has to be learned once.

### 9.2.1 The signature, and what a `tensor` argument becomes

```c
void main(tensor input, tensor output)
```

`tensor` is not a pointer — it is an **opaque handle resolved by index**, and the listing shows it: loads
name tensor **`0x0`**, stores name tensor **`0x1`**, which is the argument order.

```
	{
		mov  b11111 %I3, %I2
		ld_tnsr  %V1, 0x0, %I2      # input  == argument 0
	}
	{
		add.f32  %V5, %V0, %V1
		st_tnsr  0x1, %I2, %V5      # output == argument 1
	}
```

**MEASURED** *(listing, lines 95–133 of the 238-line `-S` output; the command is §9.3's with `-c` replaced
by `-S`)*.

### 9.2.2 The index space: `int5`, offset and size

```c
const int5 indexSpaceStart = get_index_space_offset();
const int5 indexSpaceEnd   = get_index_space_size() + indexSpaceStart;
```

Those two calls read the core's slice of the 5-D index space out of the **index register file**, and the
listing shows both the read and the arithmetic:

```
	add.i32  b11111 %I3, %I1, %I0        # %I1 + %I0  -> size + offset  == the end
	mov_irf_dim  0x0 %S0, %I0            # %I0 = offset, dim 0
	mov_irf_dim  0x0 %S1, %I3
	shl.i32  %S0, %S0, 0x6               # offset * 64  == depthStep
	shl.i32  %S1, %S1, 0x6
	cmp_geq.i32  %SP1, %S0, %S1
	jmpr .LBB0_13, %SP1                  # an empty slice skips the whole kernel
```

**MEASURED** *(listing, lines 27–39)*. `%I0` is the slice's **offset** and `%I1` its **size**, in
index-space units. The `shl.i32 … 0x6` is ×64 — `VECTOR_SIZE` for `f32`, i.e. the source's
`indexSpaceStart[depth] * depthStep` (`add1_fwd_f32_gaudi2.c:24–26`) folded to a shift. **That 64 is the same
64 as [§4.3.1](04-execution-model.md)'s lane count**: one index-space step on dim 0 is one whole 256-byte
register.

**`int5` is the five dims, and the kernel names them:** `depth = 0`, `width = 1`, `height = 2`,
`batch = 3`, `fifthDim = 4` (`:12–16`). The five loops are nested in that order and each carries
`#pragma loop_taken`, which tells the compiler the trip count is unknown but the loop matters — without it
`-O2` may treat a zero-trip loop as dead. **Why the point, not the byte, is the unit is
[§4.2.9](04-execution-model.md)'s result**; what matters *here* is that the number of points is decided by the
**glue**, not by the kernel.

### 9.2.3 The steps: what one index-space unit means per dimension

```c
const int depthStep = VECTOR_SIZE;   // 64 floats
const int widthStep = 4;             // 4 vectors, i.e. 256 floats
const int heightStep = 1;  const int batchStep = 1;  const int fifthDimStep = 1;
```

(`:24–46`.) The steps are the *kernel's half* of a contract whose other half is the glue's
`TensorAccessPattern` `a` field (§9.4.3): `depthStep = 64` must match `mapping[0].a = 64`, `widthStep = 4`
must match `mapping[1].a = 4`. **They agree because the same two constants are written twice**, once in C on
the device side and once in C++ on the host side, with nothing checking that they match — the most likely
silent bug in a kernel of this shape.

For a `1024 × 1024` `f32` tensor the grid is `[ceil(1024/64), ceil(1024/4), 1, 1, 1] = [16, 256, 1, 1, 1]`,
i.e. **4,096 points** of 4 × 64 = 256 floats each. **INFERRED** from the two constants and the shape, and
consistent with §9.4.4's trace: 24 TPCs at 4,096/24 = 170.7 points per TPC.

### 9.2.4 The body, and the four-instruction unrolled group

```c
const VECTOR one = v_mov_s((SCALAR)1.0);      // :50
#pragma unroll 4
for (int w = widthStart; w < widthEnd; w += 1) {
    ifmCoords[width] = w;
    x00 = v_ld_tnsr_i(ifmCoords, input);      // :73
    x00 = v_add_v_v(x00, one, 0);             // :75
    st_tnsr_i_v(ifmCoords, output, x00);      // :77
}
```

The scalar broadcast is one prologue instruction — `mov.f32 %V0, 0x3f800000`, the constant `1.0f` in a vector
register, inside a three-instruction bundle:

```
// %bb.2:                               // %for.body.lr.ph
{
	mov_irf_dim  0x1 %S5, %I0
	set_indx  %I2, b11111, 0x0
	mov.f32  %V0, 0x3f800000
}
```

The loop is unrolled by four, so the inner body is four load/add/store triples in a tight bundle group with a
`loop %S10, %S6, 4, <, .LBB0_9` back-edge. **MEASURED** *(listing, lines 42–46, 115, 123–156)*. The unroll
does **not** reduce the bundle count per element — the add is one vector instruction either way, and
[§4.3.5](04-execution-model.md) measures a bigger `unroll` as a **loss**. It gives the load/store slots
something to overlap ([§4.1.2](04-execution-model.md)); it saves no vector ops.

**The whole kernel is 96 bundles**, by both methods: the default listing's brace groups, and the line count
of `-mllvm --tpc-asm-format=2`. **MEASURED.** *(The two do **not** agree on the per-slot count — 82 non-NOP
lines against 94 — so no slot count is quoted. §4.1.1 says why the default listing's line count is the wrong
thing to count; §4.5 gives the instrument.)*

---

## 9.3 `tpc-clang` and the `objcopy -I binary` embed

### 9.3.1 The compiler

```
$ tpc-clang --version
TPC C/C++ compiler version 2.7.33
Target: tpc
Thread model: posix
InstalledDir: /usr/bin
```

**MEASURED.** The compile is one line and a **plain shell invocation** — there is no CMake, Make or ninja
step for the TPC stage on this stack, the same finding as [§4.2.4](04-execution-model.md):

```sh
tpc-clang -Wall -Werror -march=gaudi2 -O2 \
    -I/usr/lib/habanatools/include -Itpc_kernel_lib/kernels/include \
    tpc_kernel_lib/kernels/gaudi2/add1_fwd_f32_gaudi2.c \
    -c -o tpc_kernel_lib/build/add1_fwd_f32_gaudi2.o
```

(`build.sh:29–32`.) The compiler supports **a single translation unit**, hence `-c`.

> **The kernel does not compile from the toolchain alone.** `#include "kernel_config.h"`
> (`add1_fwd_f32_gaudi2.c:8`) resolves to a header that is **not** in `habanatools` — that package ships
> `tpc_kernel_lib_interface.h`, `gc_interface.h` and the test-core headers, and nothing else
> (`ls /usr/lib/habanatools/include/`, **MEASURED**). Upstream's README pointed the `-I` at a path inside one
> developer's checkout of `Habana_Custom_Kernel`. The example vendors the header at
> `tpc_kernel_lib/kernels/include/kernel_config.h`, licence header intact: 92 lines of `#define` per dtype —
> `VECTOR_SIZE 64`, `typedef float SCALAR`, `v_mov_s(a) → v_f32_mov_b(a)`.

### 9.3.2 The embed, and the symbol-name trap

The TPC ELF is not *loaded*; it is **copied into the glue library as bytes** and handed to Synapse at
instantiation. `objcopy -I binary` turns any file into an object with three symbols derived from the **input
filename**:

```sh
cp tpc_kernel_lib/build/add1_fwd_f32_gaudi2.o tpc_kernel_lib/build/__add1_fwd_f32_gaudi2.o
cd tpc_kernel_lib/build
objcopy -I binary -O elf64-x86-64 -B i386:x86-64 \
    --rename-section .data=.rodata,alloc,load,readonly,data,contents \
    __add1_fwd_f32_gaudi2.o __add1_fwd_f32_gaudi2_kernel.o
```

and the symbols that come out:

```
$ nm __add1_fwd_f32_gaudi2_kernel.o
00000000000038d0 R _binary___add1_fwd_f32_gaudi2_o_end
00000000000038d0 A _binary___add1_fwd_f32_gaudi2_o_size
0000000000000000 R _binary___add1_fwd_f32_gaudi2_o_start
```

**MEASURED.** Read the name: `_binary_` + `__add1_fwd_f32_gaudi2_o` + `_start`, because the input file is
`__add1_fwd_f32_gaudi2.o` and every non-alphanumeric becomes `_`. **That double underscore in the copy is the
whole reason the copy exists**: the glue declares `_binary___add1_fwd_f32_gaudi2_o_start` — three underscores
— at `gaudi2_src/add1_fwd_f32_gaudi2.cpp:13–14`.

> ⚠ **Embed the file under its own name and the link still succeeds.** Measured: `objcopy` on
> `add1_fwd_f32_gaudi2.o` mints `_binary_add1_fwd_f32_gaudi2_o_start` (**two** underscores), and
> `g++ -shared` links the glue against it with **exit 0** — a shared library may carry undefined symbols.
> The `.so`'s dynamic table then contains `U _binary___add1_fwd_f32_gaudi2_o_start`, and the failure appears
> only at run time, as §9.5.2's generic graph-compile error. **The linker will not catch this.** Check with
> `nm -D libadd1_kernels.so | grep _binary_`: all three should be `R`, none `U`.

`--rename-section .data=.rodata,…` is not required — the shorter form in upstream's README also links and
runs — but it puts the kernel image in a read-only section, which is what it is. **MEASURED** *(both forms
link; `objdump -h` shows `.rodata … READONLY` with it and `.data … DATA` without)*.

### 9.3.3 The link, and the whole build

```sh
g++ -shared -fPIC -O2 -I/usr/lib/habanatools/include -Itpc_kernel_lib/src/gaudi2_src \
    -std=c++17 -Wall -Wl,--allow-multiple-definition \
    tpc_kernel_lib/src/entry_points.cpp \
    tpc_kernel_lib/src/gaudi2_src/add1_fwd_f32_gaudi2.cpp \
    tpc_kernel_lib/build/__add1_fwd_f32_gaudi2_kernel.o \
    -o tpc_kernel_lib/build/libadd1_kernels.so
```

`--allow-multiple-definition` is there because the embedded object and the two compilation units each pull
in the same inline definitions from `gc_interface.h`. The `missing .note.GNU-stack` warning is a property of
`objcopy`'s output.

**The real `build.sh` output, verbatim** (two `[…]` elisions, both long `-I`/`-L` lists; nothing else cut):

```
== 1/4  device ELF: tpc-clang -> tpc_kernel_lib/build/add1_fwd_f32_gaudi2.o
== 2/4  embed the ELF as a binary object
== 3/4  host glue: tpc_kernel_lib/build/libadd1_kernels.so
/usr/bin/ld: warning: tpc_kernel_lib/build/__add1_fwd_f32_gaudi2_kernel.o: missing .note.GNU-stack section implies executable stack
/usr/bin/ld: NOTE: This behaviour is deprecated and will be removed in a future version of the linker
== 4/4  PyTorch extension: setup.py build
running build
running build_ext
building 'hpu_custom_add1' extension
creating build/temp.linux-x86_64-cpython-312
[1/1] c++ -MMD -MF build/temp.linux-x86_64-cpython-312/hpu_custom_add1.o.d -fno-strict-overflow -Wsign-compare -DNDEBUG -g -O2 -Wall -fPIC -I/usr/local/lib/python3.12/dist-packages/torch/include […] -I/usr/include/python3.12 -c -c hpu_custom_add1.cpp -o build/temp.linux-x86_64-cpython-312/hpu_custom_add1.o -std=c++17 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=hpu_custom_add1
/usr/local/lib/python3.12/dist-packages/torch/include/c10/core/AutogradState.h:89:33: warning: default member initializers for bit-fields only available with '-std=c++20' or '-std=gnu++20' [-Wc++20-extensions]
/usr/local/lib/python3.12/dist-packages/torch/include/c10/core/AutogradState.h:90:45: warning: default member initializers for bit-fields only available with '-std=c++20' or '-std=gnu++20' [-Wc++20-extensions]
creating build/lib.linux-x86_64-cpython-312
x86_64-linux-gnu-g++ -fno-strict-overflow -Wsign-compare -DNDEBUG -g -O2 -Wall -shared -Wl,-O1 -Wl,-Bsymbolic-functions build/temp.linux-x86_64-cpython-312/hpu_custom_add1.o […] -lhabana_pytorch2_plugin -lc10 -ltorch -ltorch_cpu -ltorch_python -o build/lib.linux-x86_64-cpython-312/hpu_custom_add1.cpython-312-x86_64-linux-gnu.so

built:
-rwxr-xr-x 1 nat nat   28832 tpc_kernel_lib/build/libadd1_kernels.so
-rwxr-xr-x 1 nat nat 10959632 build/lib.linux-x86_64-cpython-312/hpu_custom_add1.cpython-312-x86_64-linux-gnu.so
```

**MEASURED**, from a clean tree — `./build.sh` in a checkout with no `build/` directories, exit 0. The two
`AutogradState.h` warnings are torch's own headers under `-std=c++17` and appear in every extension build on
this stack.

**Two things to notice.** The compile line's `-I` list is **torch's**, not the example's — `setup.py` names
only `-std=c++17` and three include directories, and `cpp_extension` adds the rest. And the extension links
**`-lhabana_pytorch2_plugin`**, the *eager* plugin, chosen at `setup.py:23–26`: that is what makes
`habana::custom_op::registerUserCustomOp` resolvable at load time. Link the *lazy* plugin instead and the op
compiles and fails to load — and on this stack lazy mode is gone anyway (`PT_HPU_LAZY_MODE=0` is required,
and the lazy-only APIs warn that they do nothing).

---

## 9.4 The host side: entry points, the GUID, and how Synapse resolves a node

### 9.4.1 Three functions, one dispatch table

`entry_points.cpp` exports exactly the three symbols Synapse looks for, in `extern "C"`:

| function | called when | what the example does |
|---|---|---|
| `GetKernelGuids(deviceId, &count, guids)` | the library is registered | rejects any device but Gaudi 2 (`:22`), reports `count = 1`, writes `"add1_fwd_f32_gaudi2"` |
| `InstantiateTpcKernel(params, instance)` | **per node**, at graph compile | `strcmp(params->guid.name, …)` → `GetGcDefinitions(...)`, else `GLUE_NODE_NOT_FOUND` (`:48–51`) |
| `GetShapeInference(...)` | if the graph needs it | returns `GLUE_SUCCESS`, does nothing (`:54–60`) |

**The GUID string appears twice with no single source of truth**: in `GetKernelName`
(`gaudi2_src/add1_fwd_f32_gaudi2.cpp:19`) and as the second argument to `registerUserCustomOp`
(`hpu_custom_add1.cpp:36`). If they disagree the op registers, the library loads, and every launch fails
with §9.5.2's generic error.

### 9.4.2 What `GetGcDefinitions` decides

It is the only substantive host function, and it does three things in order:

1. **Validate the node.** Input/output counts (`:30–39`), matching shapes across four dims (`:42–45`),
   `DATA_F32` only (`:51–57`). Each mismatch *rewrites the field it objected to* and returns
   `GLUE_INCOMPATIBLE_*` — a rejected node comes back corrected and is retried, not dropped.
2. **Declare the index space** (`:68–73`): `indexSpaceGeometry[0] = ceil(n0/64)`,
   `[1] = ceil(n1/4)`, `[2] = n2`, `[3] = n3`, `[4] = 1`, with `indexSpaceRank = 5`.
3. **Copy the ELF in** (`:117–126`), returning `GLUE_INSUFFICIENT_ELF_BUFFER` if Synapse's buffer is too
   small — and writing `elfSize` *before* the comparison, so the caller learns the size it needed.

### 9.4.3 The access pattern is the other half of the contract

```cpp
out_defs->inputTensorAccessPattern[0].mapping[0].a = elementsInVec;   // 64
out_defs->inputTensorAccessPattern[0].mapping[1].a = c_unrollCount;   // 4
```

(`:76–92` for the input, `:94–111` for the output, identical.) `a` is *elements per index-space unit* —
exactly the step the device kernel multiplies by (§9.2.3). **`a` is also a first-class performance knob**:
[§4.2.9](04-execution-model.md) measures **5.235×** on a fixed 16 MB tensor from `a[1]` alone with the binary
unchanged, which is why it must be quoted with any throughput number and why the device and host constants
have to move together. **Nothing in the build complains if they disagree**; §9.8's gate is what catches it.

### 9.4.4 The GUID is the device event's name — read it, do not infer it

`check_guid_in_trace.py` runs the op five times inside `torch.profiler` and greps the exported trace:

```
trace events: 524   events named 'add1_fwd_f32_gaudi2': 120
  event name      add1_fwd_f32_gaudi2
  category        kernel
  EventName       1/add1_fwd_f32_gaudi2/0
  HW event name   TPC_SPU_START_TO_SPU_HALT
  dataType        float32
  Device          accel5
  TPCs (distinct tids) 24  over 5 launches = 120 events
  distinct recipeId per launch: 5
  span per launch (us): [3.605, 3.993, 3.833, 3.615, 3.95]
  mean span: 3.799 us
  traffic: 8.39 MB over 3.799 us = 2.21 TB/s
```

**MEASURED** (`examples/09-tpc-kernel/check_guid_in_trace.py`, 1024 × 1024 `f32`). Five things are
load-bearing:

* **The event's `name` is the GUID** — the identity chain's first link. If it does not appear, the kernel did
  not run, whatever the Python call returned.
* **`EventName` is `1/add1_fwd_f32_gaudi2/0`**: the `1/…/0` wrapper is Synapse's, so filter on a substring.
* **`HW event name` is `TPC_SPU_START_TO_SPU_HALT`**, the only true start→halt span on this hardware
  ([§1.8](01-basics.md), [§2.9](02-regions.md)): the duration is a device span, not a host guess.
* **24 distinct `tid`s per launch** — that is `tpc_working_engines`, and [§4.2.8](04-execution-model.md) says
  to read it rather than assume 24 (the observed set is `{4, 16, 22, 24}`). Here 24, at 4,096 points, i.e.
  170.7 points per TPC.
* **Five launches, five distinct `recipeId`s.** With no recipe cache configured — the default here — each
  eager launch mints a new recipe. **Whether that is five recompiles or five identities over one recipe is
  UNKNOWN**: a `recipeId` count is not a compile count, and [§2.4](02-regions.md) records a `recipeHandle`
  count that varied 1–13 across runs with a stable per-step recipe count.

**A cross-check that lands where the manual says it should.** 1,048,576 `f32` in and out is 8.39 MB; over
3.799 µs that is **2.21 TB/s**, against [§4.1.2](04-execution-model.md)'s streaming floor of
**2.26–2.43 TB/s**. So this kernel is **memory-bound at ~92 % of the floor**, and its cost is not the
96-bundle body at all. **MEASURED** *(span and element count; the bandwidth is arithmetic on them and the
comparison to §4.1.2's floor is **INFERRED**)*. The example is therefore a poor vehicle for a *kernel-side*
optimisation story, and a good one for a *shipping* story.

---

## 9.5 `GC_KERNEL_PATH` — and the sharp edge in the recipe cache

### 9.5.1 The list, and which entries are load-bearing

`GC_KERNEL_PATH` is a **colon-separated list of `.so` files** that Synapse searches for kernel libraries.
Three measured configurations, all with `PT_HPU_LAZY_MODE=0`:

| `GC_KERNEL_PATH` | eager op | `torch.compile` region containing the op |
|---|---|---|
| `libadd1_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so` | **works** | **works** |
| `/usr/lib/habanalabs/libtpc_kernels.so` (ours absent) | fails — §9.5.2 | fails |
| `libadd1_kernels.so` (stock absent) | **works** | **fails** |

**MEASURED.** The third row is the trap: **the stock library is not needed to *run* a custom kernel, but it
is needed to *compile a graph* that contains one.** Our kernel depends on no stock kernel; the recipe compiler
does, and it does not say so. Ship the two-entry list. *(This container ships
`GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so` by default, which cost me a wrong conclusion once:
**a measurement on this path must state the variable's value, not just that it was "set".**)*

### 9.5.2 What a missing kernel looks like: one error for every cause

With our library absent, the demo dies at `torch.hpu.synchronize()`:

```
RuntimeError: Graph compile failed. Recipe: .graph_dumps/0, synStatus=synStatus 26 [Generic failure].
Last synapse error: file: syn_singleton_graph.cpp:735 function: compileGraph, scope: SYN_API,
failed condition: (graph == nullptr), message: Current graph is nullptr, graph name: .graph_dumps/0
```

**MEASURED.** Nothing in it names a kernel, a GUID or a path — and it is **not specific to a missing
library**: the *identical* text is produced by the embed-symbol mismatch of §9.3.2, by a GUID string that
disagrees between `GetKernelName` and `registerUserCustomOp`, and by any glue function returning `GLUE_*`.
The error says **the node could not be instantiated** and nothing about why. Diagnosis order: (1) `nm -D` the
glue library for `U _binary_*`; (2) print both GUID strings and compare; (3) only then suspect the kernel.

### 9.5.3 The sharp edge: the list is hashed into the recipe cache key

The bridge's **disk** recipe cache keys every entry on a suffix built from a hash of the *contents* of the
kernel libraries plus the driver version. From `gaudi-pytorch-bridge` at tag `v1.24.2`:

```cpp
// backend/kernel/hpu_recipe_cache.cpp:293
cache_id_suffix_ = absl::StrCat("_", CacheVersion::libs_env_hash(), "_syn", versionStr.data());  // :305
recipe_cache_.store(hashCode + cache_id_suffix_, val.rl_->recipe_, ss);                          // :317
auto recipe = recipe_cache_.lookup(std::to_string(spec.hashCode()) + cache_id_suffix_, ss);        // :354
```

and `libs_env_hash()` (`pytorch_helpers/habana_helpers/habana_serialization/cache_version.cpp:174`):

```cpp
hash = at::hash_combine(hash, hash64_file_content(path_to_syn_helpers));    // :178  the bridge .so
hash = at::hash_combine(hash, hash64_file_content(get_synapse_lib_path())); // :179  libSynapse.so
if (IS_ENV_FLAG_DEFINED_NEW(GC_KERNEL_PATH)) {                              // :181
  std::string gc_kernel_path = GET_ENV_FLAG_NEW(GC_KERNEL_PATH);
  auto foundComma = gc_kernel_path.find(':');                               // :184
  if (foundComma != std::string::npos) {
    do { hash = at::hash_combine(hash, hash64_file_content(<path before ':'>));   // :193
         gc_kernel_path = <strip past the ':'>;
         foundComma = gc_kernel_path.find(',');                             // :198  <-- ','
    } while (foundComma != std::string::npos);
  }
  hash = at::hash_combine(hash, hash64_file_content(gc_kernel_path));        // :203
}
```

**source read** *(v1.24.2; the plugin here is the same stack; two statements are abridged to
`<angle brackets>` where the source builds a substring)*. Four consequences, none of them the one the
variable's name suggests:

1. **What is hashed is the file's bytes, not the string** (`hash64_file_content` `mmap`s and hashes it,
   `cache_version.cpp:120–170`). So **renaming or moving an identical library does not invalidate the cache**,
   and **rebuilding it does** — every TPC edit costs a full cache miss, which is what you want.
2. **It is the disk cache's key, not the in-memory LRU's.** `RecipeArgumentSpec::hash_code`
   (`hpu_habana_cache.cpp.h:69`) is what the in-process LRU compares, and `libs_env_hash` is **not in it** —
   it enters only through `cache_id_suffix_`. Changing `GC_KERNEL_PATH` evicts nothing in a running process;
   it makes the *next* process miss.
3. **⚠ The parser splits on `:` once and then on `,`.** `:184` finds the first `:`; `:198` then looks for a
   **comma**, which is not the list separator — the comment at `:186` says *"comma separated"*. A two-entry
   colon list works by accident; for **three or more**, the rest is hashed as one path string,
   `hash64_file_content` fails to open it and returns **0**, so only the **first** entry's content is keyed.
4. **⚠ On this stack the disk recipe cache and a colon list are mutually exclusive.** Setting
   `PT_HPU_RECIPE_CACHE_CONFIG` makes the bridge validate `GC_KERNEL_PATH`, and a colon list is rejected
   outright:

   ```
   RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_SYNHELPER Flag GC_KERNEL_PATH:
   Path is not a regular file: <our lib>:/usr/lib/habanalabs/libtpc_kernels.so
   terminate called without an active exception
   ```

   **MEASURED**, exit **134**, with the cache flag `true` *and* with it `false` — the mere presence of the
   variable is enough. A single-path `GC_KERNEL_PATH` passes. So: **a custom TPC kernel can have the
   two-entry search path, or the disk recipe cache, but not both** — and since §9.6 needs `torch.compile`,
   the recipe cache is what goes.

**What is UNKNOWN.** I could only *read* the hash, not measure it: the disk cache never populated here —
with `PT_HPU_RECIPE_CACHE_CONFIG` set, even a trivial `x * 2.0` region fails to compile
(`syn_singleton_graph.cpp:735`, `synStatus 26`) before any cache file is written, and without it there is no
cache. The manual's own [`experiments/`](experiments/README.md) rung 1 shows the key's shape from a working
configuration — `4169968267915495333_a16fb581eee_syn1.24.2.79e284b.recipe`, whose middle field is the
11-hex-digit `libs_env_hash` — but that is a **different container state**. What would settle it: a compiled
region with the cache on, then the same region with one byte appended to the custom library, then a diff of
the two filenames.

---

## 9.6 Exposing it to PyTorch

### 9.6.1 The C++ op: four parts, none optional

`hpu_custom_add1.cpp` is 73 lines and is the whole PyTorch face.

**1. An output-metadata callback** — the registry must know the output's dtype and shape before the kernel
runs (`:11–16`): `{c10::ScalarType::Float, inputs[0].toTensor().sizes().vec()}`.

**2. A parameters callback** (`:21–31`). The kernel has no scalar parameters and the descriptor still
requires a stub — hence the empty struct and `HPU_PARAMS_STUB`. A kernel with real parameters fills them
here; the returned size is what Synapse copies to the device.

**3. The registration**, at static-initialisation time (`:34–38`, `:73`):

```cpp
habana::custom_op::registerUserCustomOp(
    "custom_op::custom_add1", "add1_fwd_f32_gaudi2", output_meta, fill_params);
static const auto& KernelReg = register_custom_add1();
```

The first string is the **op name in a namespace**; the second is the **kernel GUID** — many op names may
point at one kernel, and the registry is what connects them.

**4. Two `TORCH_LIBRARY` blocks plus a Meta kernel** (`:55–70`):

```cpp
TORCH_LIBRARY(custom_op, m)            { m.def("custom_add1(Tensor self) -> Tensor"); }
TORCH_LIBRARY_IMPL(custom_op, HPU, m)  { m.impl("custom_add1", custom_add1); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("custom_add1", custom_add1_meta); }
```

> **The namespace is `custom_op`, and that is not cosmetic.** [§2.4](02-regions.md) measures that a node in
> the **`custom_op` namespace short-circuits the shared-layer validator** and is placed `hpu_cluster`
> **regardless** of what the validator would say — which is what makes it co-schedulable with MME and DMA in
> one submission. A new namespace (the `kchain` mistake) buys isolation and pays for it with the overlap.
> **That is §9.7's subject; it is why the `TORCH_LIBRARY` line matters more than the kernel body.**

The Meta implementation (`:64–66`) returns `input_a.new_empty(...)` so a tracing pass can run the op with no
device present: **measured, without it the op cannot be traced**.

### 9.6.2 Loading, and the `nn.Module` wrapper

`custom_add1.py` finds the built extension by glob and loads it (`custom_add1.py:12–16`):
`torch.ops.load_library(<path>)`,
where the path is `build/lib.linux-x86_64-*/hpu_custom_add1.cpython-*-x86_64-linux-gnu.so`. Two
consequences: **nothing is installed** — `setup.py build` is enough and `install --user` (upstream's route)
is not needed; and **the import has a side effect**, so `import custom_add1` is what makes
`torch.ops.custom_op.custom_add1` exist. Import it before anything that references the op. The `nn.Module`
wrapper (`:19–27`) is three lines that call the op.

### 9.6.3 Under `torch.compile` — measured, and the trap in the measurement

```
  eager              op                  OK equal=True
  compile fullgraph  op                  OK equal=True
  compile fullgraph  control x*2.0       OK equal=True
```

**MEASURED** (`check_torch_compile.py`, each arm in its own process, 64-element `f32`, `fullgraph=True`,
`backend="hpu_backend"`). The op **composes**: it traces, the Meta kernel supplies the fake tensor, and the
compiled region runs bit-exactly.

> ⚠ **The same three arms, with `GC_KERNEL_PATH` set to our library alone, read `FAIL / FAIL / OK`.** I wrote
> that down as a finding before checking the control's environment, and it was wrong: the cause was
> §9.5.1's missing stock library, not the op. So **a control must run in the same environment as the arm it
> controls**, and **the first `FAIL` you see is a hypothesis**. It also took a second trap to notice: **once
> the custom op's recipe has failed to compile, a later compiled graph in the same process fails too**, so an
> in-process control reports the wrong thing. Hence the subprocess-per-arm structure.

---

## 9.7 The performance fact that matters most — and it is not in this chapter

A hand-written kernel's *shipping* win is not its instruction count. It is whether it lands in the same
Synapse submission as the engines around it — **placement**, which is decided by **namespace**:

* [§2.3](02-regions.md) — "customness" is not the discriminator. Two custom ops with the same registration
  API differ only in submission count (1.00 vs 2.00 `Launch`/step), and `CROSS` goes 0.2217 → 0.0000.
* [§2.4](02-regions.md) — the mechanism: `shared_layer.py:273` returns
  `True, "Graph support for user's CustomOp"` for any node whose `target.namespace == "custom_op"`, **before**
  the shared-layer validator is consulted. So `custom_op::*` is placed `hpu_cluster` regardless, joins the
  clusterable run, and lands in the same recipe as the GEMM. §2.4's engineering recipe (register in the
  **existing** library, keep the body edge-free, duration-match, then **verify placement with a
  `PRE_PARTITIONER` pass that prints `node.meta["placement"]`**) is the procedure.

**None of that is re-derived here, and nothing in this chapter verifies it.** What this chapter adds is that
the example's `TORCH_LIBRARY(custom_op, m)` line is where the namespace is chosen — not the kernel body.

> **Placement is not overlap.** Whether a clustered node *overlaps* is [§2.2](02-regions.md)'s four-condition
> law, and a TPC node with no second context cannot hide anything inside itself
> ([§4.1.4](04-execution-model.md)). **Do not infer either from the namespace**; read the trace.

---

## 9.8 Testing: bit-exact, and a gate that can fail

### 9.8.1 The demo and the gate

`./run.sh` is the demo, then the gate. The demo's verdict line, and the gate:

```
All 64 elements equal 6.0?  True
== gate: python3 gate_add1.py
  bit-exact  shape=(64,)        equal=True
  bit-exact  shape=(100,)       equal=True
  bit-exact  shape=(256,)       equal=True
  bit-exact  shape=(1024, 1024) equal=True
  dtype gate float16                    refused=True  (Input input_a expected to be Float tensor)
  PASS (0 failing arm(s))
```

**MEASURED** (`./run.sh`, exit 0; the demo's tensor prints are elided). Three deliberate choices in the gate:

* **The input is 5.0, not 0.0 or 1.0.** A kernel that adds 0, adds 1 twice, or writes the input through
  cannot pass by coincidence at 5.0 → 6.0 as it could at 0.0 → 1.0.
* **`torch.equal`, not `allclose`.** `+1.0f` is exact for every float below 2²³, so there is no tolerance to
  grant.
* **A dtype the op does not implement must be *refused*, not reinterpreted**: a silent reinterpretation of
  `float16` bits as `float32` would pass every shape check and be wrong by construction. The `TORCH_CHECK` at
  `hpu_custom_add1.cpp:42` is the thing under test.

`test_add1_op.py` is upstream's pytest test and is kept as upstream wrote it. **`pytest` is not installed in
this container** and `pip` cannot run here (`PermissionError: [Errno 13] Permission denied: '/root'`), hence
the framework-free gate. **MEASURED** *(both facts)*.

### 9.8.2 A gate that cannot fail is decoration

[§6.6.3](06-fast-iteration.md)'s rule: a gate is strengthened by **a case the bug would fail**, not by more
assertions on the same inputs, and the cheap way to find one is to write the negative control first. So: same
tree, one constant changed — `v_mov_s((SCALAR)1.0)` → `(SCALAR)2.0` at `add1_fwd_f32_gaudi2.c:50`, rebuild:

```
  bit-exact  shape=(64,)        equal=False
  bit-exact  shape=(100,)       equal=False
  bit-exact  shape=(256,)       equal=False
  bit-exact  shape=(1024, 1024) equal=False
  dtype gate float16                    refused=True  (Input input_a expected to be Float tensor)
  FAIL (4 failing arm(s))
```

exit **1**. **MEASURED.** The dtype arm still passes, because a wrong constant has nothing to do with dtype
validation. A gate with only that arm would be decoration; this one separates the two failure classes. It still cannot tell you the kernel ran **once** (use §9.4.4's trace) or that no partial block
wrote past the tensor's end (§9.10 item 5).

---

## 9.9 What goes wrong

Every row is something this chapter hit or read. **Almost nothing fails loudly at the layer that caused it.**

| # | cause | what you see | basis |
|---|---|---|---|
| 1 | kernel library not on `GC_KERNEL_PATH` | `Graph compile failed … synStatus 26 [Generic failure] … Current graph is nullptr` | MEASURED |
| 2 | embed symbol name mismatch (§9.3.2) — **links clean** | the *same* text as row 1 | MEASURED |
| 3 | GUID string differs between `GetKernelName` and `registerUserCustomOp` | the *same* text again | **INFERRED** from rows 1–2 (one error for every `GLUE_*` failure); not separately built |
| 4 | colon list + `PT_HPU_RECIPE_CACHE_CONFIG` set (§9.5.3) | `FATAL ERROR … Flag GC_KERNEL_PATH: Path is not a regular file: <whole string>`, then abort, exit 134 | MEASURED |
| 5 | stock library absent, `torch.compile` in use (§9.5.1) | `MODULE:PT_EAGER HabanaLaunchOpPT Run returned exception … recipe_manager.cpp:304 … Can not compile graph` — blames the graph | MEASURED |
| 6 | `#include "kernel_config.h"` with no `-I` | `fatal error: 'kernel_config.h' file not found` — the one loud failure | MEASURED |
| 7 | `import pytest` without pytest | `ModuleNotFoundError: No module named 'pytest'` at import, not at test time | MEASURED |
| 8 | `prof.export_chrome_trace(path)` | `TypeError: … takes 2 positional arguments but 3 were given` — a monkey-patch out of step with torch 2.13; use `prof.profiler.export_chrome_trace(path)` | MEASURED |
| 9 | a wrong access that compiles cleanly | nothing — see below | [§7.4](07-sram.md), [§4.5](04-execution-model.md) |

**Row 9 is the one that matters, and it is why the listing is the ground truth.** Two instances already in
this manual:

* **[§7.4](07-sram.md) — the 32-bit pointer.** `tpc-clang -march=gaudi2` has no 64-bit integer type and
  reports `__SIZEOF_POINTER__ 4`. Writing `*(volatile uint*)0x1000fffffd100000 = x` compiles to
  `ld_l %S0, -0x2f00000` / `st_l` — **local** memory, not global — because the high half of the address is
  discarded and the low half is a signed 32-bit offset into an 80 KiB local space. The round trip holds, at
  the wrong address, and the failure is an asynchronous engine fault later.
* **[§4.5](04-execution-model.md) — an instruction count cannot tell a live op from a dead one.** The broken
  and corrected LUT kernels have byte-identical instruction histograms (22 instructions per iteration, same
  mnemonics) and differ only in the `lookup_1c` **operand form**; the broken one's table load had been
  dead-code-eliminated, so it never read its input.

So the check, before you believe anything, is one `grep` on the `-S` listing —
`grep -E 'ld_tnsr|st_tnsr|ld_g|st_g|ld_l|st_l'`: every input as an `ld_tnsr …, 0x<id>` with the **right id**
(0 for the first argument), every store as `st_tnsr 0x<id>`, and **no** `ld_l`/`st_l` outside the prologue's
MMIO pair. `ld_g`/`st_g` is a global access through a tensor descriptor; `ld_l`/`st_l` is local.

---

## 9.10 What this chapter leaves UNKNOWN

| # | UNKNOWN | status | what would settle it |
|---|---|---|---|
| 1 | **Whether the recipe-cache key changes when the custom library is rebuilt** | **source read only** — the `libs_env_hash` → `cache_id_suffix_` → store/lookup chain is read from v1.24.2, but the disk cache never populated here | a compiled region with the cache on, one byte appended to the custom library, then a diff of the two cache filenames |
| 2 | **Whether the `:`/`,` split hashes only the first entry of a 3-entry list** | **source read** — `:184` splits on `:`, `:198` then looks for `,`; the comment at `:186` says "comma separated" | two runs with different third entries and a populated disk cache; the filenames should be identical if the reading is right |
| 3 | **Why the disk recipe cache's compile path fails on this stack at all** | **UNKNOWN** — `syn_singleton_graph.cpp:735`, `synStatus 26`, for a graph with no custom op in it; not a custom-kernel problem | a different plugin build, or a cache directory with different ownership |
| 4 | **What MMIO `0x538` controls** — the kernel prologue's `ld_l mmio %S0, 0x538` / `or.i32 %S0, %S0, 0x1000000` / `st_l mmio 0x538, %S0` (listing lines 8–21) | **UNKNOWN** — no header, no vendor statement found; it is compiler-generated, not source-generated | a TPC ISA document, or a disassembly that names the register |
| 5 | **Whether a partial block writes past the tensor** (§9.8.2) | **UNKNOWN** — outputs are right for 64/100/256/1000/1048576, and the glue has no tail guard; `maxSizes` was not read back | a `maxSizes` dump at a length that is not a multiple of 256, or a canary element after the tensor |
| 6 | **Whether five `recipeId`s over five identical eager launches mean five compiles** (§9.4.4) | **UNKNOWN** — a `recipeId` count is not a compile count | a run with the cache populated (blocked by item 3), or a compile counter |
| 7 | **Whether one library can hold several kernels end to end** (and whether `--rename-section .data=.rodata` matters) | **not investigated** — the entry points are a dispatch table, so the shape says yes, but this chapter built one kernel; both section forms link and run | a second kernel in the same library and namespace, then §9.4.4's trace check on both |

---

## What to carry forward

1. **Two artifacts, built in order, and neither checks the other.** The device ELF is compiled by
   `tpc-clang`; the host glue library embeds its **bytes** with `objcopy -I binary` and exports
   `GetKernelGuids` / `InstantiateTpcKernel` / `GetShapeInference`. The torch extension is a third artifact
   and only names a GUID.
2. **The embed's symbol name comes from the input filename**, so the copy to `__<name>.o` is what makes
   `_binary___<name>_o_start` match the glue's `extern`. Get it wrong and **the link succeeds**; the failure
   is a runtime graph-compile error.
3. **`GC_KERNEL_PATH` is a colon list and both entries are load-bearing** — ours for the GUID, the stock
   library for `torch.compile`. A missing entry produces the same uninformative error as a wrong symbol, a
   wrong GUID string, or a bad glue return. Its **contents** are hashed into the disk recipe cache key, and
   on this stack that cache and a colon list are **mutually exclusive**: if you need `torch.compile`, the
   recipe cache is what goes.
4. **Register in the existing `custom_op` namespace**, and know that the namespace is a *placement* decision
   made by `shared_layer.py`'s short-circuit ([§2.4](02-regions.md)) — not by anything in your kernel.
   Verify placement; never infer it from the name.
5. **The GUID is the device event's name.** `EventName` is `1/<guid>/0`, `HW event name` is
   `TPC_SPU_START_TO_SPU_HALT`, and the `tid` count is `tpc_working_engines` — read it, do not assume 24.
6. **Gate bit-exactly, on a value that is not 0 or 1, and prove the gate fails.** A `+2.0` kernel turns four
   arms red and leaves the dtype arm green, which is what a gate with more than one arm buys you.
7. **The error message is the least informative part of the system.** Four distinct causes in §9.9 share one
   text; the diagnosis order is `nm -D` the glue, compare both GUID strings, then read the listing.
8. **The listing is the ground truth, and it is cheap**: `-S` for the code, `--tpc-asm-format=2` to count
   bundles, `grep ld_tnsr|st_tnsr` to prove the inputs are read and the outputs written, and `ld_g`/`st_g`
   versus `ld_l`/`st_l` to prove a global access is global.

---

## Provenance of every figure

Paths without a leading `examples/` are files in `examples/09-tpc-kernel/`. "measured" means run in this
container, with the command given.

| figure(s) | value | provenance |
|---|---|---|
| compiler and stack versions | `tpc-clang` 2.7.33 (clang 14.0.5); `torch` 2.13.0a0+git2469d18; driver `1.24.2-52a555d`; fw `hl-gaudi2-1.24.0-fw-62.6.2-sec-11`; 2 × `HL-225` | measured — `tpc-clang --version`, `torch.__version__`, `hl-smi -L`, §9.3.1 |
| three artifact sizes | ELF 14,544 B (`0x38d0`); glue `.so` 28,832 B; extension 10,959,632 B | measured — `./build.sh` from a clean tree, §9.3.3; `0x38d0` corroborated by `nm`'s `_binary_…_o_size`, §9.3.2 |
| exported glue symbols | 3 | measured — `nm -D --defined-only`; declared at `tpc_kernel_lib/src/entry_points.cpp:17,39,54` |
| kernel bundle count | **96** | measured — default `-S` listing parsed by brace groups **and** `-mllvm --tpc-asm-format=2` line count, which agree; §9.2.4 |
| per-slot count | **not quoted** (82 vs 94 disagree) | measured — deliberately withheld, §9.2.4 |
| `VECTOR_SIZE` for f32 | 64 | source read — `tpc_kernel_lib/kernels/include/kernel_config.h:19`; cross-ref §4.3.1 |
| index-space step constants | `depthStep = VECTOR_SIZE`; `widthStep = 4` | source read — `add1_fwd_f32_gaudi2.c:24,29` |
| host-side `a` values | `elementsInVec = 64`; `c_unrollCount = 4` | source read — `gaudi2_src/add1_fwd_f32_gaudi2.cpp:28,60,78,83` |
| `shl.i32 … 0x6` is ×64 | the `depthStep` fold | measured — listing lines 32–33; INFERRED as the fold of `:25–26` |
| tensor ids in the listing | `0x0` load, `0x1` store | measured — listing lines 97, 101, 130 |
| grid for 1024×1024 f32 | `[16, 256, 1, 1, 1]` = 4,096 points | INFERRED — from the two constants and the shape; consistent with §9.4.4's 24 TPCs × 170.7 points |
| `habanatools` headers | 5 files, **no** `kernel_config.h` | measured — `ls /usr/lib/habanatools/include/`, §9.3.1 |
| `kernel_config.h` provenance | `HabanaAI/Habana_Custom_Kernel` @ `06e73bba`, BSD header intact | source read — the vendored file's own header, lines 1–15 |
| embed symbol names | `_binary___add1_fwd_f32_gaudi2_o_{start,end,size}` | measured — `nm` on the embedded object, §9.3.2 |
| the two-underscore trap | link exit 0; `U _binary___add1_fwd_f32_gaudi2_o_start` in `nm -D` | measured — §9.3.2 |
| section with/without `--rename-section` | `.rodata … READONLY` vs `.data … DATA` | measured — `objdump -h`, §9.3.2 |
| demo, gate, and the `+2.0` negative control | demo `True`; gate 4/4 `equal=True` + float16 refused + `PASS (0 failing arm(s))`; `+2.0` kernel 4/4 `equal=False`, dtype arm `True`, `FAIL (4 failing arm(s))`, exit 1; lengths 64/100/256/1000/1048576 all bit-exact | measured — `./run.sh` and §9.8.2's rebuilt variant |
| trace: census, event fields, span | 120 of 524 events named by the GUID; 5 launches; 24 distinct `tid`s; 5 distinct `recipeId`s; `EventName` `1/add1_fwd_f32_gaudi2/0`; `HW event name` `TPC_SPU_START_TO_SPU_HALT`; `dataType` `float32`; `Device` `accel5`; span 3.605–3.993 µs, mean 3.799 µs | measured — `check_guid_in_trace.py`, §9.4.4 |
| achieved bandwidth | 8.39 MB / 3.799 µs = 2.21 TB/s | arithmetic on the row above; the comparison to 2.26–2.43 TB/s is [§4.1.2](04-execution-model.md)'s floor, INFERRED |
| `torch.compile` arms | eager OK, compiled op OK, control OK (colon list) | measured — `check_torch_compile.py`, §9.6.3 |
| the same arms, our library alone | FAIL / FAIL / OK | measured — and identified as an environment artefact, not an op defect, §9.6.3 |
| in-process contamination | a later compiled graph fails after the op's recipe fails | measured — §9.6.3 |
| error texts | `synStatus 26 … Current graph is nullptr`; `Path is not a regular file: <whole string>` + abort, exit 134; `recipe_manager.cpp:304 … Can not compile graph` | measured — §9.5.2, §9.5.3, §9.5.1 |
| colon list + recipe cache | rejected with the cache flag both `true` and `false` | measured — §9.5.3 |
| container's default `GC_KERNEL_PATH` | `/usr/lib/habanalabs/libtpc_kernels.so` | measured — `env`, §9.5.1 |
| `libs_env_hash` definition | `cache_version.cpp:174`; libs `:178–179`; `GC_KERNEL_PATH` `:181`; first split `:184`; content hash `:193`; second split `:198`; tail `:203` | source read — `gaudi-pytorch-bridge` v1.24.2 |
| cache key construction | `cache_id_suffix_` `hpu_recipe_cache.cpp:305`; store `:317`; lookup `:354` | source read — same tag |
| `RecipeArgumentSpec::hash_code` | `hpu_habana_cache.h:69`; does **not** include `libs_env_hash` | source read — same tag |
| recipe filename shape | `4169968267915495333_a16fb581eee_syn1.24.2.79e284b.recipe` | [experiments/README.md](experiments/README.md) rung 1 — **a different container state**, cited for shape only |
| `pip` and profiler failures | `PermissionError: [Errno 13] Permission denied: '/root'`; `takes 2 positional arguments but 3 were given` | measured — §9.8.1, §9.9 rows 7–8 |
| cross-referenced, not re-measured | 32-bit pointers and `ld_l`/`st_l` vs `ld_g`/`st_g` ([§7.4](07-sram.md)); the 22 = 22 dead-LUT case and the operand rule ([§4.5](04-execution-model.md)); the `custom_op` short-circuit at `shared_layer.py:273` ([§2.4](02-regions.md)); `a` as a knob, 5.235× ([§4.2.9](04-execution-model.md)); `tpc_working_engines` ∈ {4, 16, 22, 24} ([§4.2.8](04-execution-model.md)); the 2.26–2.43 TB/s streaming floor ([§4.1.2](04-execution-model.md)) | those sections, as cited |
