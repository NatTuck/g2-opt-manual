# REPLICATE — 1Cat-vLLM-Gaudi serving DeepSeek-V4.1-Flash on 4×Gaudi2

One ordered path from a **stock Habana 1.24.2 container** to **our measured numbers**.
Written for someone who has the box and no context. Every command below is quoted from a
script, a log or a running tree in this workspace — not reconstructed — and each is followed
by the file that proves it ran.

| | |
|---|---|
| target | `NatTuck/1Cat-vLLM-Gaudi` @ `e2266e0`, serving `deepseek-ai/DeepSeek-V4.1-Flash` |
| hardware | 4 × HL-225 (Gaudi2), one box, module ids `0,1,2,3` |
| serving config | TP4 / PP1, 524,288-token window, prefix caching on, block 128, port 8021 |
| status | **build, weights, boot and measurement all completed**; one required deviation (`MHC_RRMS_POST=0`) and one local correctness patch (halo tail) are described in §2.11 and §7. The two fusion flags are covered in §4.5: the arm that produced every number runs them at `0`, and the conformant alternative is documented there and is **not faster** |
| companion docs | `INSTALL.md` (narrative install), `STATUS.md` (per-phase gate + every surprise), `measurements/*/FINDINGS.md` (per-cell results), `manifest.json` (every input with a hash) |

**Reading note on `INSTALL.md`.** Its header block still says "Phases 2–9 are marked *NOT YET
RUN*". That header is stale: the phases are written and their logs exist under `logs/`. The
*content* of `INSTALL.md` is accurate where it agrees with `logs/` and `scripts/`; where it
disagrees, this document follows the scripts and logs and says so (§2.7 is the one case).

---

## 0. The order of work, and why this order

**How this document was checked.** Every command line below was compared against a file on
disk: the driver script that contains it, or the log that records its output. The document holds
**115 top-level command lines** in its shell blocks (76 of which launch a process; the other 39
are `export`/assignment/`cd`/`source` scaffolding), and every one of the 76 was traced to a
script under `cases/1cat/scripts/`, `cases/1cat/measurements/*/`, or a `cases/1cat/logs/` file.
Where a script and `INSTALL.md` disagree, the script wins and the disagreement is stated (§2.7).
Numbers are quoted with the file they come from, and anything that could not be verified is
labelled **UNVERIFIED** (§9) rather than smoothed.

```
§1  container contract            (verify, install nothing)
§2  build                         (clone → patch → venv → HCL → Synapse → Bridge → TP2 → kernels → engine)
§3  weights                       (audit → adopt → prepare → sidecars → audit aliases)
§4  launch                        (profile → launcher → entrypoint → teardown gate)
§5  measurement protocol          (client → cache epoch → cells → gates)
§6  the numbers                   (what to expect, from which file)
```

Nothing in §3–§6 is meaningful if §2 was built from different sources: the native library
hashes are pinned into the runtime profile and verified by the entrypoint at load time.

Work inside one workspace directory and keep every cache inside it. This install used
`/home/nat/DeepseekFlash/cases/1cat` and calls it `$CASE`:

```bash
export CASE=/home/nat/DeepseekFlash/cases/1cat
mkdir -p "$CASE"/{logs,cache/wheels,cache/hf-meta,src,build,evidence,weights,locks,scripts,bin,profiles}
```

Free space: the prepared weights are **342.23 GiB** on their own and the checkpoint is
**475.25 GiB** in the HF cache. The phase-0 gate required **≥ 600 G** free in the workspace, and
phase 7 re-checked the disk immediately before preparing (`logs/phase7-prepare-shards.log`:
`free before: 706873442304 bytes` = 658 GiB, `free after: 357313908736 bytes` = 333 GiB).

---

## 1. Prerequisites and the container contract

This guide does **not** install the platform. It verifies a standard Habana **1.24.2**
container. Driver downgrades are out of scope and are not a workaround for anything below.

### 1.1 The contract, with the verification command for each row

| component | expected | check (run it; do not assume) |
|---|---|---|
| OS | Ubuntu 24.04 | `cat /etc/os-release` |
| driver | 1.24.2-52a555d | `hl-smi` (header line) |
| cards | 4 × HL-225, idle | `hl-smi --query-aip=index,module_id,name,memory.used,utilization.aip --format=csv` |
| torch | 2.13.0a0+git… | `PYTHONPATH=/usr/lib/habanalabs python3 -c "import torch;print(torch.__version__)"` |
| habana userspace | 1.24.2.481 | `PYTHONPATH=/usr/lib/habanalabs python3 -c "import importlib.metadata as m;print(m.version('habana-torch-plugin'))"` |
| SDK libs | `/usr/lib/habanalabs` | `ls -la /usr/lib/habanalabs` |
| TPC compiler | `tpc-clang` 2.7.33 | `tpc-clang --version` |
| TPC tools | `tpc-llvm-objdump`, `gc_interface.h` | `which tpc-llvm-objdump; test -f /usr/lib/habanatools/include/gc_interface.h && echo present` |
| `fuser` | psmisc | `which fuser` |
| cmake | **≥ 4.0** | `cmake --version` |
| python | 3.12 | `python3 -V` |

Observed here (all from `logs/phase0-*.log`, consolidated in
`logs/phase0-phase1-gate-verification.log`): Ubuntu 24.04.5 LTS; driver `1.24.2-52a555d`;
4 × HL-225 at **768 MiB used / 0 % util**; torch `2.13.0a0+git2469d18`; habana-torch-plugin
`1.24.2.481`; `libSynapse.so` 64,776,840 B, `libhcl.so` 18,240,936 B, `libTPCFuser.so`
83,796,392 B; `tpc-clang` "TPC C/C++ compiler version 2.7.33"; `/usr/bin/tpc-llvm-objdump`
present; `/usr/lib/habanatools/include/gc_interface.h` 25,381 B present; `psmisc 23.7-1build1`;
**cmake 4.4.0** (so no cmake wheel is needed); python 3.12.3.

Two rows are not cosmetic:

* **`tpc-clang` 2.7.33 must be the container's.** The phase-5 kernel build compiles TPC
  kernels with it; a different version changes the kernel binaries and the build manifests.
* **`tpc-llvm-objdump` and `gc_interface.h` must exist** or the kernel build refuses to run.
  Both are checked in `logs/phase0-sdk-tools.log`; `gc_interface.h` lives at
  `/usr/lib/habanatools/include/`, *not* under `/usr/lib/habanalabs`.

### 1.2 The one device check that matters

```bash
PYTHONPATH=/usr/lib/habanalabs python3 -c "
import torch, habana_frameworks.torch.hpu as ht
print(torch.__version__, torch.hpu.device_count(), torch.hpu.is_available())
x = torch.randn(64,64).to('hpu'); y = (x @ x); torch.hpu.synchronize()
print('sum', float(y.sum().cpu()))"
```

Expect `4 True` and a finite sum. This is also the **acquire probe** used as a gate
everywhere later in the campaign:

```bash
"$CASE/venv/bin/python" -c "
import torch, habana_frameworks.torch
x = torch.randn(8, 8).to('hpu'); torch.hpu.synchronize(); print('    acquire OK')"
```

A failure here with `synStatus=8 [Device not found]` is **contention first, not a leak** —
check for a running job (§4.4) before blaming the card. Logs:
`logs/phase0-torch-habana.log`, and `acquire OK` lines in every
`measurements/*/logs/driver*.log`.

### 1.3 Two container warts that break naive commands

**(a) `PYTHONPATH` contains an unreadable path.** The image ships
`PYTHONPATH=/root:/usr/lib/habanalabs/`. `/root` is mode 0700 and unreadable as a normal user,
so pip's distribution scan raises `PermissionError: [Errno 13] Permission denied: '/root'`
before it contacts PyPI (`logs/phase0-pypi-requirements.log`, first attempt; the offending
`/root` entry is visible in the `sys.path` dump in `logs/phase3-venv-isolation.log`).
**Every python and pip command in this guide therefore sets `PYTHONPATH=/usr/lib/habanalabs`
explicitly.** Verify with:

```bash
PYTHONPATH=/usr/lib/habanalabs python3 -c "import sys;print(sys.path)"
```

**(b) Another workspace's `vllm_gaudi` is on the system python's path.**
`~/.local/lib/python3.12/site-packages` carries two editable finders
(`__editable__.vllm_gaudi-0.24.0…finder`, `…0.24.0rc1.dev22+g6690c4604…finder`). A bare
`python3 -c "import vllm_gaudi"` resolves into **another workspace's tree**
(`/home/nat/Llms/src/vllm-gaudi/vllm_gaudi/__init__.py`, measured in
`logs/phase3-venv-isolation.log`). §2.3's venv plus `PYTHONNOUSERSITE=1` removes the shadow;
§2.10 asserts the resolved paths rather than trusting the import to succeed.

### 1.4 `hl-smi`'s AIP index is not the module id

Observed (`logs/phase0-module-ids.log`):

```
index, module_id, name, memory.used [MiB], utilization.aip [%]
0, 2, HL-225, 768 MiB, 0 %
2, 0, HL-225, 768 MiB, 0 %
3, 3, HL-225, 768 MiB, 0 %
6, 1, HL-225, 768 MiB, 0 %
```

Reason and pin in **module ids** (`HABANA_VISIBLE_MODULES`, `HLS_MODULE_ID`), never in the
`hl-smi` index. The module-id set here is the full `{0,1,2,3}`. `hl-smi -L` floods the
terminal with log — use `--query-aip=...`.

### 1.5 The environment contract: `$CASE/env.sh`

Source this before any build or launch. Every value in it exists because something failed
without it; the file's own comments say what.

```bash
. "$CASE/env.sh"
```

It sets, and here is why each one is load-bearing:

| variable | value here | why |
|---|---|---|
| `PYTHONPATH` | `/usr/lib/habanalabs` | §1.3(a) |
| `PYTHONNOUSERSITE` | `1` | §1.3(b) — the venv is `--system-site-packages`, so the user site is otherwise live |
| `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `HF_HOME`, `VLLM_CACHE_ROOT` | under `$CASE/cache` | caches stay in the workspace; `VLLM_CACHE_ROOT` is also the thing the cache-epoch rule clears (§5.2) |
| `PT_HPU_RECIPE_CACHE_CONFIG` | `$CASE/cache/recipes,false,1024,false` | the HPU recipe cache; warm it once and a boot is ~20 min shorter |
| `PATH` | `$VENV/bin:$CASE/bin:/opt/habanalabs/bin:$PATH` | **the only `cmake` on this box is a pip console script** (`/home/nat/.local/bin/cmake`, `from cmake import cmake`) which dies under `PYTHONNOUSERSITE=1`; `$CASE/bin/cmake` is a symlink to the real binary inside the package (`logs/phase5-cmake-wrapper-trap.log`) |
| `LD_LIBRARY_PATH` | our synapse → our mme → our hcl → venv torch → `/usr/lib/habanalabs` | the bridge build dir must **not** be on this path (it would shadow the SDK) |
| `VLLM_HPU_DSV41_MHC_RRMS_POST` | `0` | its operator ships in the additive bundle this install does not build; default is `1` and the entrypoint then raises `FileNotFoundError` (§7.1) |
| `VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR` | `$CASE/build/dsv41-native/lib` | the entrypoint hash-verifies every `.so` against the build manifests before loading |

Create the cmake shim once:

```bash
ln -s /home/nat/.local/lib/python3.12/site-packages/cmake/data/bin/cmake "$CASE/bin/cmake"
"$CASE/bin/cmake" --version    # -> cmake version 4.4.0
```

---

## 2. The build

### 2.1 Clone the pinned sources, then verify each pin from its public origin

```bash
cd "$CASE/src"
export GIT_TERMINAL_PROMPT=0
git clone https://github.com/NatTuck/1Cat-vLLM-Gaudi.git 1Cat-vLLM-Gaudi
git clone https://github.com/vllm-project/vllm.git vllm
git clone https://github.com/HabanaAI/gaudi-pytorch-bridge.git gaudi-pytorch-bridge
git clone https://github.com/heyuanliu-intel/Intel_Gaudi3_Software.git synapse
git clone https://github.com/HabanaAI/HCL.git hcl

git -C vllm checkout e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
git -C gaudi-pytorch-bridge checkout 5176b1b2d6865608514e83a2b1fd66459920676e
# 1Cat, synapse and hcl are already at their pins on their default branches.
```

The pins, and the tarball hash that identifies each one deterministically
(`git archive --format=tar HEAD | sha256sum`), are rows in `manifest.json` → `sources[]`:

| dir | repo | pin | tarball sha256 |
|---|---|---|---|
| `1Cat-vLLM-Gaudi` | `NatTuck/1Cat-vLLM-Gaudi` | `e2266e0efa381e6649e0f25b789cbb61b21b0e1f` | `e1ea8ee7394214639d49353cbd10ab38eeb605098e89a611bac3201087497203` |
| `vllm` | `vllm-project/vllm` | `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` | `14755f0034d906d3c0b031cd491d93151981605e3cc6a8f77cb881960c84153a` |
| `gaudi-pytorch-bridge` | `HabanaAI/gaudi-pytorch-bridge` | `5176b1b2d6865608514e83a2b1fd66459920676e` | `6c0eb6503458e83765573212c5b25187cf87353b8517cd3ec13ebe06b7fbff1d` |
| `synapse` | `heyuanliu-intel/Intel_Gaudi3_Software` | `6de1f66a4ab658685f23dbe007bdc9eb8fe8dbdf` | `59232e109867eafb52eff443e96168369375bf48eef0f6dc5a50a5d25bd20afc` |
| `hcl` | `HabanaAI/HCL` | `11f9114b88038174d8b2c9ac09e2295b75455078` | `bd5ec47a70648c730cbde481d48eb56cb28add47f7b4bd2c7818bc1f158e7a85` |

Verify all five at execution time — never trust a pin because a note says it was checked:

```bash
for d in 1Cat-vLLM-Gaudi vllm gaudi-pytorch-bridge synapse hcl; do
  echo "$d HEAD=$(git -C $d rev-parse HEAD) dirty=[$(git -C $d status --porcelain)]"
  echo "   tar_sha256=$(git -C $d archive --format=tar HEAD | sha256sum | cut -d' ' -f1)"
done
```

`vllm` and `gaudi-pytorch-bridge` are expected to be **detached HEAD**, so assert on the
commit, not on the branch. `git ls-remote` cannot confirm a non-tip commit (`e47aa780` is one);
the GitHub commits API can. Recorded in `logs/phase1-pins.log`, `logs/phase1-tarball-hashes.log`.

**The bridge pin is v1.24.1, not v1.24.2, and that is deliberate.** The 1Cat repo's
`tools/communication/patches/native-runtime/manifest.json` pins the bridge base at `5176b1b2`
("Publish v1.24.1", tag `v1.24.1-482`) and `apply_native_runtime.py` **refuses any other base**:

```
RuntimeError: Expected 5176b1b2d6865608514e83a2b1fd66459920676e, found 27eccbcff…
```

(`logs/phase1-bridge-patch-check-at-v1242.log`.) `27eccbc` is "Publish v1.24.2" and is exactly
one commit ahead; that commit is a 53-file / ±1545-line release delta
(`logs/phase1-bridge-v1241-vs-v1242.log`). Do **not** "upgrade" any pin to match the
container's version string: the patch fingerprints and the runtime's ABI sidecar are keyed to
these exact commits. `STATUS.md` → "Phase 1 — clone the pinned sources — GATE PASS (1
deviation)" records the decision and the refusal log.

### 2.2 Apply the patches, in the order the hash chain forces

Verify every patch hash against the manifest **before** applying:

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi/tools/communication/patches/native-runtime"
python3 - <<'EOF'
import json, hashlib
man = json.load(open('manifest.json'))
for name, e in man['components'].items():
    h = hashlib.sha256(open(e['patch'],'rb').read()).hexdigest()
    print(name, e['patch'], h == e['sha256'], e['base_commit'][:12])
EOF
```

Recorded values (`logs/phase2-patch-hash-verify.log`) — all seven match:

| patch | sha256 | base | applied? |
|---|---|---|---|
| `bridge.patch` | `bc35b881c1e305eaec286dc334ffe4b73f7b5ce02401c4fc0671e306ca4c0b3d` | `5176b1b2…` | **yes** |
| `synapse.patch` | `eeeb7478d958920b…` | `6de1f66a…` | **yes** |
| `hcl.patch` | `e86f0684ca77e5e4…` | `11f9114b…` | **yes** |
| `synapse-bounded-reindex.patch` | `1860c18a827b446f…` | requires `synapse.patch` | **yes** |
| `synapse-program-window.patch` | `19dc2592c3bcfcae…` | requires bounded-reindex | **yes** |
| `synapse-tpc-fuser-state.patch` | `4317181de57c7040…` | — | **no** (§2.2.2) |
| `hcl-receive-prepost.patch` | `78a3fd1db7432f4f…` | optional overlay | **no** (§2.2.2) |

Apply:

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi"
python3 tools/communication/apply_native_runtime.py bridge  --source ../gaudi-pytorch-bridge --apply
python3 tools/communication/apply_native_runtime.py synapse --source ../synapse              --apply
python3 tools/communication/apply_native_runtime.py hcl     --source ../hcl                  --apply
python3 tools/communication/apply_bounded_reindex_runtime.py --source ../synapse --apply
python3 tools/communication/apply_native_program_window.py   --source ../synapse --apply
```

Each applier self-verifies the patch sha256, the base commit, a clean tree,
`git apply --check --whitespace=error`, then applies and re-hashes every touched file against
the manifest's `after` value. All five printed success with **zero fuzz**
(`logs/phase2-apply-native-patches.log`).

The synapse overlay order is forced by a **hash chain**, not preference:

```
synapse.patch            → stream_compute_scal.cpp 5c1f3c82…
bounded-reindex.patch    → stream_compute_scal.cpp ff20f0b9…   (its "before" is synapse's "after")
program-window.patch     → stream_compute_scal.cpp 4b42051f…   (its "before" is bounded-reindex's "after")
```

Applying program-window before bounded-reindex is refused by the per-file `before` hash check.

The vLLM delta is the repo's own pinned 18-file engine delta, not a hand patch:

```bash
python3 tools/prepare_deepseek_v41_engine.py ../vllm --check
python3 tools/prepare_deepseek_v41_engine.py ../vllm
```

#### 2.2.1 `dsv41-v2-platform.patch` is already contained in the engine delta — do not apply it

Applying it after the engine delta **fails**, and it fails because the content is already
there:

```
error: patch failed: vllm/config/vllm.py:2779
error: patch failed: vllm/platforms/interface.py:141
```

(`logs/phase2-apply-vllm-patches.log`.) The evidence is the file hashes: `vllm/config/vllm.py`
is at the patch's own `candidate_sha256` (`3f280c3f…`), and `interface.py` is at the engine
delta's candidate (`f9081180…`), a superset. `supports_v2_without_triton` is present verbatim
in both files. Diagnosis in `logs/phase2-v2-platform-diagnosis.log`; `STATUS.md` → "Phase 2 —
apply the patches" records it as a deviation from `PLAN.md`.

#### 2.2.2 What was deliberately not applied

* **`receive-prepost` (HCL + Synapse).** The native-runtime README calls it "the optional
  receive-epoch overlay", "separate from the base patches", and
  `VLLM_HPU_NATIVE_RECEIVE_PREPOST` remains disabled by default. Applying it would change
  relocation layout and force a full HCL rebuild for a feature that is off.
* **`synapse-tpc-fuser-state.patch`.** No applier script exists (unlike the other two
  overlays), the README gives no apply command for it, and its manifest qualification reads
  "experimental compiler ownership fix; device validation pending". This is the one place a
  future reader may want to add it, if concurrent-compile ownership problems appear.

Resulting diffstat vs each pin (`logs/phase2-diffstat.log`):

```
gaudi-pytorch-bridge   7 files changed,   63 insertions(+),     1 deletion(-)
synapse               80 files changed, 4211 insertions(+),   863 deletions(-)   (+10 new)
hcl                   35 files changed, 2588 insertions(+),    59 deletions(-)   (+5 new)
vllm                  17 files changed,  365 insertions(+),    68 deletions(-)   (+1 new)
1Cat-vLLM-Gaudi       unmodified (it is the tool source)
```

### 2.3 The venv — and why it is not a plain `python3 -m venv`

`python3 -m venv --system-site-packages` **fails in this container**: there is no `ensurepip`
(`python3.12-venv` is not installed), no `sudo`, no `virtualenv`, and no local pip wheel
(`logs/phase3-venv-create.log`, rc=1). Create the venv without pip and bootstrap pip from the
public origin:

```bash
export PYTHONPATH=/usr/lib/habanalabs
python3 -m venv --without-pip --system-site-packages "$CASE/venv"
curl -sSL -o "$CASE/cache/get-pip.py" -w '%{http_code}\n' https://bootstrap.pypa.io/get-pip.py
PYTHONPATH= "$CASE/venv/bin/python" "$CASE/cache/get-pip.py" --no-cache-dir
"$CASE/venv/bin/python" -m pip --version
```

Observed: `get-pip.py` HTTP 200, 2,397,294 B, sha256
`356e0d0de21cdc348c3394084d90939b85e4246c4d14c76aee6ca4f46646ea98`; `pip 26.3` installed
(`logs/phase3-venv-create.log`).

**`--system-site-packages` re-enables the user site.** A venv normally disables it; with
`--system-site-packages` it does **not** — `site.ENABLE_USER_SITE` is `True` and
`~/.local/lib/python3.12/site-packages` is back on `sys.path`, carrying the two editable
`vllm_gaudi` finders of §1.3(b). Measured (`logs/phase3-venv-isolation.log`):

| invocation | `import vllm_gaudi` resolves to |
|---|---|
| `venv/bin/python -c ...` | **`/home/nat/Llms/src/vllm-gaudi/vllm_gaudi/__init__.py`** ← another workspace |
| `PYTHONNOUSERSITE=1 venv/bin/python -c ...` | not found (correct, pre-install) |

**Set `PYTHONNOUSERSITE=1` in every invocation** (it is in `env.sh`). The container's
`dist-packages` — torch and `habana_frameworks` — stay visible, which is the point of
`--system-site-packages`. Assert it rather than assume it:

```bash
PYTHONNOUSERSITE=1 "$CASE/venv/bin/python" -c "
import site, sys, importlib.util as u
assert site.ENABLE_USER_SITE is False
assert not any('.local' in p for p in sys.path)
for m in ('vllm','vllm_gaudi'):
    s = u.find_spec(m); print(m, s.origin if s else 'NOT FOUND')"
```

### 2.4 Copy the vendor package tree in, then fix the `.upstream.so` SONAME trap

```bash
VP="$CASE/venv/lib/python3.12/site-packages"
SRC=/usr/local/lib/python3.12/dist-packages
cp -a $SRC/habana_frameworks "$VP/"
cp -a $SRC/habana_dataloader "$VP/"
cp -a $SRC/habana_*1.24.2.481.dist-info "$VP/"
```

`habana_frameworks` is a **namespace package**, so its `__path__` lists both the venv copy and
the container copy. That is harmless — `habana_frameworks.torch` is a regular package and
resolves to the first entry — but verify it (`logs/phase3-vendor-copy.log`):

```bash
PYTHONNOUSERSITE=1 "$CASE/venv/bin/python" -c "
import habana_frameworks.torch as ht, torch
print(ht.__file__)      # must be .../cases/1cat/venv/lib/python3.12/site-packages/...
print(torch.__file__)   # .../usr/local/lib/python3.12/dist-packages/torch/...  (container)
print(torch.hpu.device_count())"
```

**Now the trap.** `habana_frameworks/torch/__init__.py:63` chooses the library by whether torch
is Habana's fork:

```python
lib_to_load = "libhabana_pytorch{}_plugin{}.so".format(
    "" if is_lazy() else "2", "" if is_torch_fork else ".upstream")
```

`is_torch_fork` is **True** here, so the *primary* is loaded. But the repo's C++ extensions link
`-lhabana_pytorch_backend.upstream` and `-lhabana_pytorch2_plugin.upstream`, and in this
container those `.upstream.so` files are **real files with their own distinct SONAMEs**, not
symlinks (`logs/phase3-upstream-soname-fix.log`):

| file | before | after |
|---|---|---|
| `libhabana_pytorch_backend.upstream.so` | 57,727,656 B, SONAME `…backend.upstream.so` | symlink → `libhabana_pytorch_backend.so`, SONAME `…backend.so` |
| `libhabana_pytorch2_plugin.upstream.so` | 22,349,672 B, SONAME `…2_plugin.upstream.so` | symlink → `libhabana_pytorch2_plugin.so` |
| `libhabana_pytorch_plugin.upstream.so` | 29,752,216 B, SONAME `…plugin.upstream.so` | symlink → `libhabana_pytorch_plugin.so` |

Two real files means **two mappings of the same backend in one process**, which aborts with
`hpu::habana_bernoulli is already registered!` once a PT2 extension links the `.upstream` name
alongside the primary. Collapsing the names to one library removes the second mapping.

```bash
LIB="$VP/habana_frameworks/torch/lib"
for b in libhabana_pytorch_backend libhabana_pytorch2_plugin libhabana_pytorch_plugin; do
  rm -f "$LIB/$b.upstream.so" && ln -s "$b.so" "$LIB/$b.upstream.so"
done
objdump -p "$LIB/libhabana_pytorch_backend.upstream.so" | grep SONAME   # -> …backend.so
```

Do **not** touch `/usr/local/lib/python3.12/dist-packages/...` (the container's copy). Re-apply
this after any `habana_frameworks` reinstall — it restores real `.upstream.so` files.
`scripts/install-bridge-into-venv.sh` re-links them as part of installing the rebuilt bridge
(§2.7), which is why the two steps cannot be separated.

### 2.5 Build HCL

```bash
bash "$CASE/scripts/build-hcl.sh"
```

The script is the record; its four requirements are:

```bash
export HCL_SRC_PKG_DIR="$CASE/src/hcl"
export HCL_LIB_DIR=/usr/lib/habanalabs
export BUILD_ROOT_LATEST="$CASE/build/hcl/latest"   # MUST be writable
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake -S "$CASE/src/hcl/hcl/src" -B "$CASE/build/hcl" -DCMAKE_BUILD_TYPE=Release
make -C "$CASE/build/hcl" -j32
```

**Trap:** `hcl/hcl/src/CMakeLists.txt:19` expands `$ENV{BUILD_ROOT_LATEST}` at configure time
into a POST_BUILD `create_symlink` of `<target>.debug`. If it is unset the destination becomes
`/`, the link step fails, and `make` then **deletes the freshly linked `libhcl.so`** — which
looks like a compile or link failure and is neither.

Result: `build/hcl/lib/libhcl.so`, 16,558,488 B, sha256
`c33c0da535b175206366222aaa82d0b9c0d098598f397e48286cd05bdf7212d7`, ~70 s at `-j32`
(`logs/phase3-hcl-build.log`).

### 2.6 Build Synapse — including **fix #1, which is not in the repo**

```bash
bash "$CASE/scripts/build-synapse.sh"
```

The script does, in order:

```bash
cd "$CASE/src/synapse"
export PS1="${PS1:-}"                       # habana_set_env_val.sh does export PS1="...$PS1"
export CMAKE_POLICY_VERSION_MINIMUM=3.5     # bundled protobuf-3.9.x declares 3.1.3
source ./habana_set_env_val.sh ./ "$CASE/build/synapse"
export BUILD_ROOT_LATEST="$CASE/build/synapse/latest"
build_hl_logger -r -j 32
build_hl_gcfg   -r -j 32
cmake -S "$SYNAPSE_ROOT" -B "$CASE/build/synapse/synapse-release" \
      -DCMAKE_BUILD_TYPE=Release -DTESTS_ENABLED=OFF -DAUTO_GENERATED_TESTS=OFF \
      -DSYNAPSE_DFA_REGISTER_DUMP=OFF -DHCL_ROOT="$CASE/src/hcl/hcl"
make -C "$CASE/build/synapse/synapse-release" -j32 Synapse
```

`HCL_ROOT` is overridden to **our patched HCL**, not the unpatched `hcl/` that ships inside the
Synapse snapshot. `PS1` must be set because `habana_set_env_val.sh:49` expands it even in a
non-interactive shell.

#### Fix #1 — the CMake-4 `lemon` policy (`3rd-parties/lemon/CMakeLists.txt`)

**File:** `$CASE/src/synapse/3rd-parties/lemon/CMakeLists.txt`
**sha256 before:** `6d335a8ce6f2bfa30b73428aea73f6229f9c0bae445041e6f756c80bf755d360`
**sha256 after:** `a3ae591ebc39f57e438ed5ce783f52dfe7093979688e18ef0bf56723ec96e90b`

`CMAKE_POLICY_VERSION_MINIMUM=3.5` lowers `cmake_minimum_required` floors, but it does **not
re-enable a removed policy**. The bundled lemon sets `CMAKE_POLICY(SET CMP0048 OLD)`, and CMake
4 removed that behaviour outright:

```
CMake Error at CMakeLists.txt:3 (CMAKE_POLICY):
  Policy CMP0048 may not be set to OLD behavior because this version of CMake
  no longer supports it.
```

**Exact edit** (original lines 1–3 → patched lines 1–8):

```cmake
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# CMake >= 4 removed CMP0048's OLD behaviour; keep the original semantics on
# older CMake and skip the (now illegal) policy set on 4.x.
IF(CMAKE_VERSION VERSION_LESS 4.0)
  CMAKE_POLICY(SET CMP0048 OLD)
ENDIF()
```

**Then delete the copied tree and its stamps** — `lemon` is pulled in as an `ExternalProject`
that **DIR-copies** its source into `<build>/ThirdParty/src/lemon_project` during the download
step, so after patching the source the stale copy is reused and the same error returns:

```bash
rm -rf "$CASE/build/synapse/synapse-release/ThirdParty/src/lemon_project" \
       "$CASE/build/synapse/synapse-release/ThirdParty/src/lemon_project-build" \
       "$CASE/build/synapse/synapse-release/ThirdParty/src/lemon_project-stamp"
```

Both the before/after hashes and the "the copy still had line 3 unpatched" observation are in
`logs/phase3-lemon-cmake4-fix.log`; the fix is the only change to the synapse tree beyond the
repo's own patches (`git diff --stat` in the same log: 1 file, +5 −1).

#### The other two Synapse build facts

* **`BUILD_ROOT_LATEST` is a staging directory**, and Synapse links finished components from
  it: `make[3]: *** No rule to make target '…/build/synapse/latest/libhcl.so'`. Copy ours in:
  `cp "$CASE/build/hcl/lib/libhcl.so" "$CASE/build/synapse/latest/libhcl.so"`. Twenty-two SDK
  libraries must be staged there as well (`logs/phase3-build-fixes.log`, "FIX 1b").
* The build is **`make … Synapse`**, not `all`: GraphCompiler is the bulk of the time and
  nothing else is needed.

### 2.7 Build the Bridge — including **fix #2, which is not in the repo**

The patch manifest pins the bridge base at v1.24.1-482, but **v1.24.1 cannot build in a 1.24.2
container**, and the reason is a deleted component rather than a version string:

* v1.24.1 `CMakeLists.txt` requires `MEDIA_ROOT`, derived from
  `import habana_frameworks.mediapipe`;
* the container has **no `habana_frameworks.mediapipe`** module;
* upstream **deleted the entire `media_pyt_bridge` package in v1.24.2** (8 files → 0) and
  dropped `MEDIA_ROOT` from `REQUIRED_ENV_VARIABLES`.

Observed: `CMake Error at CMakeLists.txt:97 (message): Required environment variable MEDIA_ROOT
is not defined.` (`logs/phase3-bridge-v1241-configure-full.log`). So build from **v1.24.2-481
with the same patch**, in a separate detached worktree so the v1.24.1 tree stays intact for
comparison:

```bash
git -C "$CASE/src/gaudi-pytorch-bridge" \
    worktree add --detach ../gaudi-pytorch-bridge-1242 27eccbcffeab10842c9f29f64c6bf50d8f65d78f
P="$CASE/src/1Cat-vLLM-Gaudi/tools/communication/patches/native-runtime/bridge.patch"
git -C "$CASE/src/gaudi-pytorch-bridge-1242" apply --check --whitespace=error "$P"   # rc=0
git -C "$CASE/src/gaudi-pytorch-bridge-1242" apply --whitespace=error "$P"           # 7 files, +63 −1
```

The patch is version-agnostic **for these seven files**: `git diff` between the two tags
restricted to the seven patched paths is empty and all seven patched files are byte-identical in
both trees (`logs/phase3-bridge-rebase-1242.log`). The repo's own `manifest.json` still says
`5176b1b2`; we did not edit it (it is upstream's file and its hash is part of the contract).
`STATUS.md` → "Phase 3 … Attempt (b) — rebase the same patch onto `27eccbc` (v1.24.2-481):
CLEAN" records the decision.

Now build:

```bash
bash "$CASE/scripts/build-bridge.sh"
```

#### Fix #2 — restrict `-pie` to executables (`cmake/Targets.cmake`)

**File:** `<copy of the bridge tree>/cmake/Targets.cmake`. The fix is applied to a **copy** —
`build/bridge/bridge-src` — so the git worktree stays pristine
(`scripts/build-bridge.sh`, step [1/5]: `cp -a "$SRC_ORIG" "$ROOT/bridge-src"`, `rm -rf .git`,
then `python3 "$CASE/scripts/patch_targets_pie.py" "$ROOT/bridge-src/cmake/Targets.cmake"`).

**sha256 before:** `78d8901b6e14a210b50b98a15d1b3e537db7b479bc20309c20c3cc4bf7c9c7f1` (the
worktree file still has `-pie` on line 92)
**sha256 after:** `f9f024766ff32bf6f6169dbda289bc39fda0fb56523b2702c944c567ae4cd2aa`

Original (lines 91–93):

```cmake
  # Enable Position Independent Execution as required by SDL
  target_link_options(${target_name} PRIVATE -pie)
endfunction()
```

Replacement:

```cmake
  # Enable Position Independent Execution as required by SDL
  # ---------------------------------------------------------------------------
  # -pie must NOT be emitted on a SHARED library link: GNU ld makes -shared and -pie
  # mutually exclusive with last-one-wins, and CMake emits -shared before
  # target_link_options, so an unconditional -pie turns the .so link into a PIE
  # executable link (undefined reference to `main`, and eager resolution of every
  # symbol that a shared link would legitimately leave to runtime).
  # Restrict -pie to executables; keep the rest of the SDL hardening set unchanged.
  # ---------------------------------------------------------------------------
  target_link_options(
    ${target_name}
    PRIVATE "$<$<STREQUAL:$<TARGET_PROPERTY:${target_name},TYPE>,EXECUTABLE>:-pie>")
endfunction()
```

**Why.** `cmake/Targets.cmake` runs `target_link_options(${target_name} PRIVATE -pie)` from the
helper used by **both** `add_habana_library` and `add_habana_executable`. GNU ld treats
`-shared` and `-pie` as mutually exclusive with last-one-wins, and CMake emits `-shared` from
the SHARED_LIBRARY rule *before* `target_link_options`, so the trailing `-pie` silently
discards `-shared` and turns the shared-library link into a PIE **executable** link. ld then
demands `main` and resolves every symbol eagerly.

Measured consequence before the fix (`logs/phase3-bridge-build-narrow.log`): **105 distinct**
undefined symbols, 4,892 references — `Logger::habana_assert(...)`,
`habana_lazy::HbLazyTensor::StepMarker(...)`,
`hl_logger::v12_inline_fmt_compile::ModuleLoggerData<...>::levels`, and so on — every one of
them a symbol a shared link is entitled to leave undefined. After the fix the same log family
shows **0** undefined-reference lines and the artifacts are `Type DYN`, `entry 0x0`
(`logs/phase3-bridge-build.log`, `logs/phase3-bridge-install.log`).

`scripts/patch_targets_pie.py` refuses to run if its anchor text is missing or matches more
than once, so it cannot silently no-op on a different bridge revision.

#### What the bridge build actually used (this differs from `INSTALL.md` §5.7)

`INSTALL.md` §5.7 says the include dirs point at *our patched* synapse/HCL trees. The script
that produced the shipped artifacts does the following, and `logs/phase3-bridge-configure.log`
confirms it:

```
-- Environment variable HABANA_SOFTWARE_STACK exists: …/cases/1cat/src
-- Environment variable HCL_INCLUDE_DIR exists: /usr/include/habanalabs/
-- Environment variable SPECS_EXT_ROOT exists: /usr/include/habanalabs/
-- Environment variable SWTOOLS_SDK_ROOT exists: …/cases/1cat/src/synapse/swtools_sdk
-- Environment variable SYNAPSE_INCLUDE_DIR exists: /usr/include/habanalabs/
-- Environment variable SYNAPSE_UTILS_INCLUDE_DIR exists: /usr/include/habanalabs/
-- Environment variable BUILD_ROOT_LATEST exists: …/build/bridge/latest
```

So: **headers from the container's flattened `/usr/include/habanalabs`**, `swtools_sdk` from
our Synapse snapshot, SDK **libraries** staged into `build/bridge/latest` from
`/usr/lib/habanalabs` (22 libs, `logs/phase3-build-fixes.log` "FIX 2c"). This is fine and is
worth understanding rather than "fixing": `readelf -d` on the shipped backend shows that among
Habana libraries it links **only `libhl_logger.so`** — `libSynapse.so` and `libhcl.so` are not
`NEEDED` at all and are resolved at *runtime* from the profile's `LD_LIBRARY_PATH`, where our
builds come first (§4.2). Verify:

```bash
readelf -d "$CASE/build/bridge/build/libhabana_pytorch_backend.so" | grep -E 'NEEDED|SONAME|RUNPATH'
readelf -h "$CASE/build/bridge/build/libhabana_pytorch_backend.so" | grep -E 'Type|Entry'
```

Two further requirements, each of which cost a failed link (`logs/phase3-build-fixes.log`):

* `SWTOOLS_SDK_ROOT` must be **our snapshot's** `swtools_sdk`, because the bridge's own
  `hllog.hpp` chain expects `magic_enum-0.8.1` while the container's headers want
  `magic_enum-0.9.7`; the bridge's CMake clones `magic_enum` v0.9.7 itself
  (`Cloning into 'magic_enum-src'... HEAD is now at e046b69 v0.9.7`).
* `SPECS_EMBEDDED_ROOT` must be set even though CMake treats it as optional: omitting it
  silently drops `mem_hlml.cpp`, the backend then exports no `HlMlMemory*` symbols, and the
  result is not a drop-in for the shipped `fork_pybind/_hpu_C.so`. The gate is
  `HlMlMemory exports = 34`.

Build the four artifacts the runtime pins, not `all`:

```bash
ninja -C "$CASE/build/bridge/build" -j32 \
  libhabana_pytorch_backend.so libhabana_pytorch_plugin.so libhabana_pytorch2_plugin.so
ninja -C "$CASE/build/bridge/build" -j32 _hccl_eager_C.so
```

Then install into the venv (this also re-links the `.upstream` names of §2.4 and runs the
marker check):

```bash
bash "$CASE/scripts/install-bridge-into-venv.sh"
```

`logs/phase3-bridge-install.log`:

```
libhabana_pytorch_backend.so  57768416 B  sha256 30ad2f03…
   vendor kept: libhabana_pytorch_backend.so.vendor  57724152 B  sha256 601bda2b…
libhabana_pytorch_plugin.so   29772872 B  sha256 98733fcd…
libhabana_pytorch2_plugin.so  22382744 B  sha256 ee225cce…
fork_pybind/_hccl_eager_C.so   1631272 B  sha256 4a3a87b2…
libhabana_pytorch_backend.upstream.so -> libhabana_pytorch_backend.so  (SONAME now libhabana_pytorch_backend.so)
ours   libhabana_pytorch_backend.so:        hccl::tp2_allreduce_residual_rms_norm = 1
vendor libhabana_pytorch_backend.so.vendor: hccl::tp2_allreduce_residual_rms_norm = 0
libhabana_pytorch_backend.so: HlMlMemory exports = 34  type=DYN  entry=0x0
```

The six optional pybind extras that fail to link (`_activity_profiler_C`, `_utilization_metrics_C`,
`_git_info_C`, `_event_dispatcher_C`, `_hpex_C`, `_bridge_config_C`) are **not pinned by the
runtime profile**; the venv keeps the container's copies. Do not widen the build to `all`.

### 2.8 The TP2 fused AR/Norm bridge extension

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi"
export LD_LIBRARY_PATH=$CASE/build/synapse/synapse-release/lib:\
$CASE/build/synapse/synapse-release/lib/mme:\
$CASE/build/hcl/lib:\
/usr/local/lib/python3.12/dist-packages/torch/lib:\
/usr/lib/habanalabs
python tools/communication/build_tp2_fused_ar_norm_bridge.py \
  --bridge-source "$CASE/src/gaudi-pytorch-bridge-1242" \
  --cmake-build-directory "$CASE/build/bridge/build" \
  --build-directory "$CASE/build/tp2-ext" \
  --backend-library "$CASE/venv/lib/python3.12/site-packages/habana_frameworks/torch/lib/libhabana_pytorch_backend.so" \
  --native-hccl-library "$CASE/venv/lib/python3.12/site-packages/habana_frameworks/torch/lib/fork_pybind/_hccl_eager_C.so" \
  --synapse-library "$CASE/build/synapse/synapse-release/lib/libSynapse.so" \
  --hcl-library "$CASE/build/hcl/lib/libhcl.so"
```

**The bridge build directory must not be on `LD_LIBRARY_PATH`** (it would shadow the SDK).

Result (`logs/phase4-tp2-ext.log`): `build/tp2-ext/tp2_fused_ar_norm_bridge.so` 1,890,584 B,
sha256 `107fa4af502d9dba3db70cd0e13ce61488ab83d88889efb5cf6a6dd8495d09cd`, plus
`tp2_fused_ar_norm_bridge.abi.json`, which pins every adapter source, our `2_plugin` as
`eager_runtime`, our backend and `_hccl_eager_C` as bridge dependencies, our `libSynapse.so`
(`146e9407…`) and `libhcl.so` (`c33c0da5…`) as native dependencies, and the two patched bridge
headers (`graph_storage.h` `4b8848c5…`, `graph_execs_group.h` `644c1f31…`). **Carry the sidecar
with the `.so`** — the loader rejects a different runtime.

### 2.9 V4.1 kernels and host gather

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi"
GAUDI_PYTORCH_BRIDGE_ROOT="$CASE/src/gaudi-pytorch-bridge-1242" \
GAUDI_PYTORCH_BRIDGE_BUILD_ROOT="$CASE/build/bridge/build" \
python tools/build_deepseek_v41.py \
  --build-root "$CASE/build/dsv41-native/build" \
  --output-dir "$CASE/build/dsv41-native/lib" \
  --jobs $(nproc)
```

Outputs: `libdeepseek_v4_gaudi2_kernels.so` (8.7 MB, `9c7c2aef…`),
`hpu_dsv4_control_mme_pt2…so` (15.8 MB, `0f377af6…`), `hpu_dsv4_sparse_attn_pt2…so` (529.6 MB,
`077e6aea…`), `dsv41_host_gather…so` (5.8 MB, `4d29f52e…`), plus `deepseek_v4_build.json`
(`c2f1149c…`), `deepseek_v41_build.json` (`42e3f446…`) and `tpc_loop_audit.json`
(`logs/phase5-dsv41-native.log`, `logs/phase5-gate-verify.log`).

Gate checks, all passing here:

```bash
find "$CASE/build/dsv41-native" -name INVALID_BUILD.json | wc -l     # -> 0
python3 -c "import json;d=json.load(open('$CASE/build/dsv41-native/lib/deepseek_v41_build.json'));\
print({k:d[k] for k in ('prepared_layout_version','host_gather_abi_version','host_c1_abi_version',\
'host_gather_packed_output_version','host_gather_profiling_version')})"
# -> 2, 1, 2, 1, 1
```

**Load each `.so` in its own process.** `hpu_dsv4_control_mme_pt2` and
`hpu_dsv4_sparse_attn_pt2` cannot share one: both compile the control-MME registration unit and
the second load aborts with
`custom_op::custom_deepseek_v41_control_mme_f32_gaudi2 is already registered!` — observed, rc=134
in `logs/phase5-gate-verify.log`. All four load individually; `dsv41_host_gather` is a Python
extension and is imported by name.

### 2.10 Install the engine and the plugin

`--no-build-isolation` means the build backends must already be **in the venv**; installing
them is a required step, not an optional one. Both installs failed once with
`ModuleNotFoundError: No module named 'setuptools_rust'` / `'setuptools_scm'`
(`logs/phase6-install.log`).

```bash
export PYTHONPATH=/usr/lib/habanalabs
"$CASE/venv/bin/python" -m pip install "setuptools==80.10.2" "setuptools-scm>=8.0" \
    "setuptools-rust>=1.9.0" wheel jinja2 "packaging>=24.2"
"$CASE/venv/bin/python" -m pip install -r "$CASE/src/1Cat-vLLM-Gaudi/requirements.txt"
VLLM_TARGET_DEVICE=empty "$CASE/venv/bin/python" -m pip install --no-build-isolation -e "$CASE/src/vllm"
"$CASE/venv/bin/python" -m pip install --no-build-isolation -e "$CASE/src/1Cat-vLLM-Gaudi"
```

Installed: `vllm 0.28.1rc1.dev392+ge47aa780b.d20261109.empty`,
`vllm_gaudi 0.1.dev980+ge2266e0ef`. `numpy` becomes 2.3.5 in the venv (vLLM pins it) while the
container's is 2.5.3; the venv copy wins and that is expected.

**Then pin `transformers`.** vLLM's only constraint is `transformers >= 5.10.4`, so pip takes
the newest (5.22.2 here), and 5.18.0+ renamed `PixtralRotaryEmbedding` →
`PixtralVisionRotaryEmbedding`, which vLLM imports unconditionally:

```bash
"$CASE/venv/bin/python" -m pip install "transformers==5.16.0"
"$CASE/venv/bin/python" -c "from transformers.models.pixtral.modeling_pixtral import PixtralRotaryEmbedding; print('OK')"
```

The empirical ceiling was established by bisection: `<= 5.16.0` present, `>= 5.18.0` renamed
(`logs/phase9-fix-transformers.log`; 5.20.2 still fails). This is an **undeclared
incompatibility in the pinned vLLM commit**, not a local mistake.

Gate — assert, do not assume (`logs/phase6-gate-verify.log`):

```
site.ENABLE_USER_SITE   : False
PYTHONNOUSERSITE env    : 1
~/.local on sys.path    : False
vllm      -> /home/nat/DeepseekFlash/cases/1cat/src/vllm/vllm/__init__.py
vllm_gaudi-> /home/nat/DeepseekFlash/cases/1cat/src/1Cat-vLLM-Gaudi/vllm_gaudi/__init__.py
```

```bash
PYTHONNOUSERSITE=1 "$CASE/venv/bin/python" -c "
import vllm, vllm_gaudi, site, sys
for m in (vllm, vllm_gaudi): assert '/cases/1cat/' in m.__file__, m.__file__
assert site.ENABLE_USER_SITE is False and not any('.local' in p for p in sys.path)
print('isolation OK')"
```

The native libraries are **not** found by a `vllm_gaudi/lib` symlink (there is none); the
entrypoint reads `VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR` and, when set, verifies each `.so`'s
sha256 against `deepseek_v4_build.json` / `deepseek_v41_build.json` before loading. Keep it in
the environment and in the profile (§4.2).

### 2.11 The one local correctness patch: the TP4 decoder-halo tail

**This patch is not in the repo and is not part of the upstream install.** It is applied to the
working tree and is described in full in §7.3. Without it the engine **dies** (HTTP 500, then
process death) for any single-chunk prompt in `(4096, 16384)` tokens, which includes the 8,192
bucket. With it, 8,192 and 12,288 work.

Where it sits in the order: it was written **after** the phase-9 boot, when
`measurements/scaling1` hit the dead window (its 8,192 cell returned HTTP 500 and killed the
engine), and it was verified on device in `measurements/haloverify`. Apply it after §2.2's
patches. Both `vllm` and `vllm_gaudi` are installed **editable**, so the source edit is live and
needs no re-install — but any running server must be restarted to pick it up. Everything in §6
was measured with it in place.

---

## 3. The weights

### 3.1 Verify free space first, and never delete anything to make room

The prepared tree needs **342.23 GiB**; the checkpoint source is a read-only HF cache snapshot.
Re-run `df -h` immediately before starting and stop if it will not fit.

### 3.2 Audit, adopt, plan, prepare

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi"
# 1) audit the public checkpoint (also produces upstream-lock.json)
python tools/audit_deepseek_v41_upstream.py "$CASE/evidence/upstream-audit"
# 2) adopt the read-only HF cache snapshot (verifies all 48 shards; no re-download)
SNAP=/data/nat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
python tools/sync_deepseek_v41_checkpoint.py "$SNAP" --evidence "$CASE/evidence/upstream-audit"
# 3) plan first — it prints the budget
python tools/prepare_deepseek_v41_shards.py "$SNAP" "$CASE/weights/prepared" \
  --checkpoint-audit "$CASE/evidence/upstream-audit/complete.json" \
  --upstream-lock "$CASE/evidence/upstream-audit/upstream-lock.json" \
  --tensor-parallel-size 4 --plan-only
# 4) then for real (drop --plan-only)
```

Plan output, recorded verbatim (`logs/phase7-shard-plan.log`):

```json
{"stage": "planned", "output_payload_gib": 342.23356941342354, "temporary_limit_gib": 2.0,
 "plan_fingerprint": "2bd1ff74ed5764cd20ecf6909197f8996ec8c4a70c5acaaf7046af8e4e001990"}
```

The real run published with the same fingerprint and `prepared_layout_version=2`, TP4/PP1,
48/48 shards verified (`logs/phase7-prepare-shards.log`, `logs/phase7-gate-verify.log`).

**Trap — `--checkpoint-audit` is a FILE here and a DIRECTORY later.** The shard planner does
`json.loads(args.checkpoint_audit.read_text())`, so it wants `…/complete.json`; the serving
entrypoint (§4.3) wants the *directory*. Passing the wrong one gives `IsADirectoryError` or
`NotADirectoryError` respectively — both were observed
(`logs/phase7-shard-plan.log`, `logs/phase9-fix-checkpoint-audit.log`).

### 3.3 The three sidecars

```bash
for t in prepare_deepseek_v41_woa_fp8 prepare_deepseek_v41_dense_fp8 prepare_deepseek_v41_engram_fp8; do
  python tools/$t.py "$CASE/weights/prepared"
done
```

All three rc=0 with per-layer relative-L2 around 2–7e-06 (`logs/phase7-sidecars.log`).

### 3.4 Alias the audit records (or the runtime cannot find them)

`sync_deepseek_v41_checkpoint.py` writes `files/model-00047-of-00048.safetensors.json`. The
runtime verifies the **resolved** path, and in the HF cache that is a blob whose basename is the
LFS oid, so it asks for `files/824db488….json`
(`vllm_gaudi/ops/deepseek_v41_host.py:649`). The record *content* is already the blob's identity
(inode / mtime_ns / bytes / sha256 all match `os.stat` on the blob); only the name differs:

```bash
python3 "$CASE/scripts/alias-audit-records.py"
```

88 aliases written, 0 skipped; the specific file the runtime asked for exists and is
content-identical to the original (`logs/phase9-fix-audit-record-names.log`).

### 3.5 The hashes to check before believing the tree is right

| what | value | where |
|---|---|---|
| `weights/prepared/manifest.json` sha256 | `fba62c0bd29ebb65a09b75474dcca6fcb2364a73aebfde25a51078b4731ea6f5` | `logs/phase7-gate-verify.log` |
| plan fingerprint | `2bd1ff74ed5764cd20ecf6909197f8996ec8c4a70c5acaaf7046af8e4e001990` | `logs/phase7-shard-plan.log` |
| `prepared_layout_version` / TP / PP | `2` / `4` / `1` | `logs/phase7-gate-verify.log` |
| sidecar `source_manifest_sha256` (all three) | `fba62c0bd29ebb65…` | same |
| `attention_dense_fp8` quantization fingerprint | `67ac6228f577eb2a…` | same |
| `engram_fp8` quantization fingerprint | `0cc7a5c40711db49…` | same |
| `wo_a_fp8` quantization fingerprint | `95eb30422100fbc0…` | same |
| 12 sidecar shard hashes (3 × 4 ranks) | **12 ok, 0 bad** | same |
| sidecar shard sizes | dense 841,007,336 B ×4; engram 314,778,152 B ×4; wo_a 335,880,504 B ×4 | same |

And from the source side, `manifest.json` → `checkpoint`: revision
`dba1be0a40aa45a94ad051997016db3960a90277`, 48 shards, `total_bytes` 510,296,708,312,
per-shard `{path, size, lfs_oid}` (e.g. `model-00001-of-00048.safetensors`, 970,533,624 B,
oid `886aebdafa08cc27bbae2165ed35bdfe0de9370bf88c1411283c155c6ae4ff89`), plus the four small
files with real sha256 (`config.json` `8be45ce0…`, `tokenizer_config.json` `6ac8c8dc…`,
`model.safetensors.index.json` `74b0686a…`, `tokenizer.json` `c90dfa01…`). 48/48 shard oids and
sizes were cross-checked against the HF cache snapshot (`logs/phase0-hf-crosscheck.log`).

**The checkpoint has no chat template.** `tokenizer_config.json` is 801 B, complete, and has no
`chat_template` key; the tree has no `chat_template.jinja`. Chat formatting for this model is a
Python encoder (`encoding/encoding.py`, harmony style). Use `/v1/completions`, or supply a
template explicitly; do **not** assume `/v1/chat/completions` works out of the box. (It does
work here — the engine's own reasoning parser fills `content` — but that is the entrypoint's
doing, not the tokenizer's.)

---

## 4. The launch

### 4.1 The exact command

One command, from the repo root, with `env.sh` sourced. This is
`scripts/run-phase9.sh` — the file itself is the record:

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi"
exec "$VENV/bin/python" tools/run_deepseek_v41.py \
  --lock-dir "$CASE/locks" \
  --runtime-profile "$CASE/profiles/verified-runtime-profile.json" \
  --modules 0,1,2,3 \
  --preferred-cpus "0-127" \
  --min-host-available-gib 64 \
  "$CASE/evidence/run" -- \
  "$VENV/bin/python" -m vllm_gaudi.entrypoints.deepseek_v41 \
    "$CASE/weights/prepared" \
    --checkpoint-audit "$CASE/evidence/upstream-audit" \
    --host 127.0.0.1 \
    --port 8021 \
    --max-model-len 524288 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --block-size 128 \
    --enable-prefix-caching \
    ${EXTRA_ARGS:-}
```

`run-phase9.sh` also refuses to start if any module already shows > 2 GiB in use:

```bash
BUSY=$(hl-smi --query-aip=memory.used --format=csv,noheader | tr -d ' MiB' | awk '$1>2000' | wc -l)
[ "$BUSY" = "0" ] || { echo "REFUSING: $BUSY module(s) show >2 GiB in use before we start" >&2; exit 1; }
```

**The measurement cells did not use these three numbers.** The drivers in `measurements/`
launch either with `--max-num-seqs 32 --max-num-batched-tokens $CHUNK` and
`VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS=$CHUNK` (`$CHUNK` = 16384 for the headline cells), or —
for the `run3` batch run — by calling `run-phase9.sh` unchanged, i.e. 16 / 8192. The per-run
table in §5.4 says which. `--max-num-seqs` is a **server-start parameter**: it cannot be changed
in a running lifetime, so a 16-vs-32 comparison needs two boots
(`measurements/conformance1/run.sh:4-5`, and C3 was skipped for exactly this reason at `:135`).
If your brief fixes 16, change both numbers and re-measure; do not compare across the two.

### 4.2 The environment, variable by variable

Three layers stack: the process environment, `profile.environment`, and user config.
`INSTALL.md` §10 documents the precedence as process environment → `profile.environment` → user
config, and warns that a profile which sets `HABANA_VISIBLE_MODULES` can override a device choice
made on the command line — which is why the launcher **re-selects and locks modules after loading
the profile** and records the effective environment in `evidence/<run>/runtime-profile.json`. For
the two fusion flags the observed behaviour is that an explicit process value wins, because the
entrypoint applies its own defaults with `os.environ.setdefault` (§4.5).

**Layer 1 — process environment, set on the launch line** (verbatim from
`measurements/conformance2/run-conformance.sh`):

```bash
VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP=0 \
VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION=0 \
VLLM_HPU_DSV41_MHC_RRMS_POST=0 \
VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS=16384 \
  "$VENV/bin/python" "$REPO/tools/run_deepseek_v41.py" ...
```

`WOA_OUTPUT_ROUNDTRIP=0` and `PREFILL_OUTPUT_PROJECTION=0` are the **reporting arm**: every
number in §6 and in `REPORT.md` comes from them. They are a correctness workaround on this
build — at the entrypoint's own defaults the engine serves id `0` for every step — and §4.5
now says exactly where the defect is and why the conformant alternative is not adopted for
speed. `MHC_RRMS_POST=0` is a **build** deviation — its operator lives in the additive bundle
this install does not build (§7.1).

`PREFILL_COMPUTE_TOKENS=16384` is kept explicit for a reason that predates the flag audit: the
repo's ledger records the `0` default's inheritance from the scheduler as the defect which
eliminated the 16,384 prefill bucket. The docs say TP4 *derives* its tile capacity from
`--max-num-batched-tokens` up to `16384`
(`docs/configuration/env_variables.md`, `PREFILL_COMPUTE_TOKENS`), which would make the pin a
no-op at our launch's own `16384`. Both readings give `16384` here, so the pin is harmless
either way and is not counted as a deviation from the entrypoint's intent (the entrypoint does
not manage the key — `scripts/flag-conformance.py`, category B).

To run the **conformant** configuration instead — the repo's own intent for TP4×PP1, which is
what `scripts/flag-conformance.py` compares against — use:

```bash
VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP=1 \
VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION=1 \
VLLM_HPU_DSV41_MHC_RRMS_POST=0 \
VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS=16384 \
  "$VENV/bin/python" "$REPO/tools/run_deepseek_v41.py" ...
```

and leave `VLLM_HPU_DSV41_WOA_DECODE_HANDOFF` unset. That is
`measurements/woasplit/run-split.sh`; it boots, serves and measures 0.979x prefill with
unchanged decode, so it is not the faster arm. It requires the local split patch (§7.5).

**Layer 2 — the runtime profile** `$CASE/profiles/verified-runtime-profile.json`, generated by
`scripts/make-runtime-profile.py` (the repo ships no `make-profile.sh`;
`docs/1cat_gaudi_guide.md` gives the structure and the recipe). Ours is **12 environment keys,
11 libraries, 3 config files**, sha256
`d1de78770cd09f80f297ac013aa3f73a208f39e2b90d977ff9de9a4dba756ea9`:

```bash
python3 "$CASE/scripts/make-runtime-profile.py"     # prints the new sha256
sha256sum "$CASE/profiles/verified-runtime-profile.json"
```

Regenerate it whenever any of the 11 libraries or 3 configs changes — the profile *is* the pin.
`logs/phase8-runtime-profile.log` records an earlier 11-key version (`6b4667a4…`); the
`MHC_RRMS_POST` key was added afterwards, which is why the current file is `d1de7877…` with 12
keys. The 11 libraries are: our backend, plugin, 2_plugin, `_hccl_eager_C`, `libhcl.so`,
`libSynapse.so`, `tp2_fused_ar_norm_bridge.so`, the four native V4.1 libs. The 3 configs are the
TP2 ABI sidecar and the two native build manifests.

**Layer 3 — NUMA.** CPU pools are *not* a profile field; they go through the launcher's
`--preferred-cpus`. On this box node0 is `0-31,64-95` and node1 is `32-63,96-127`
(`logs/phase8-runtime-profile.log`), and the drivers pass `0-127` and let the launcher's
conflict policy place each rank (`evidence/run/process.json` → `modules[]` shows
main/helper CPU assignments).

**Assert the effective profile equals the source profile** — the launcher strips nothing:

```bash
cmp "$CASE/profiles/verified-runtime-profile.json" "$CASE/evidence/run/runtime-profile.json" && echo IDENTICAL
```

Verified here: both `d1de7877…`, byte-identical.

### 4.3 The entrypoint's arguments, and what the log should show

`evidence/run/process.json` records the exact argv of the last boot:

```json
[".../venv/bin/python", "-m", "vllm_gaudi.entrypoints.deepseek_v41",
 ".../weights/prepared", "--checkpoint-audit", ".../evidence/upstream-audit",
 "--host", "127.0.0.1", "--port", "8021", "--max-model-len", "524288",
 "--max-num-seqs", "32", "--max-num-batched-tokens", "16384",
 "--block-size", "128", "--enable-prefix-caching"]
```

A healthy boot, from `logs/phase9-v2-validation.log`:

```
/v1/models → id: DeepSeek-V4.1-Flash  max_model_len: 524288
/v1/chat/completions → 200, content: 'Paris', finish_reason: stop, reasoning_tokens: 18
/v1/completions → 200, ' Paris.\nAnswer: The capital of France is Paris.\nThe capital of France is'
token ids → [2581, 1309, 3287, 16, 10037, 20407, 582, 671, 6102, 294, 8760, 344, 4, 4888, 10059, 18967]
compiles after startup: 0
modules held: 88,758 MiB × 4
wall time to ready: ~11.5 min (warm recipe cache; ~21 min cold — 1265 s in the last measured boot)
```

Model id is `DeepSeek-V4.1-Flash`, **not** the HF repo path — using
`deepseek-ai/DeepSeek-V4.1-Flash` returns a 404 that reads like a missing route but is not.

`GPU KV cache size` is **not** a fixed config output. Observed across boots: 1,620,480 /
2,209,152 / 2,341,120 / 4,215,168 / 4,300,160 tokens (3.09× … 8.20× at 524,288), varying with
free device memory at boot. Do not quote it as a property of the configuration; quote the boot
it came from.

### 4.4 Launch and teardown discipline — kill the JOB, then require 768 MiB **and** a working acquire

A stopped launcher is not a stopped job. The launcher `exec`s inside the shell job's process
group, so a job left running holds all four modules at ~88.7 GiB and looks **exactly** like
leaked HBM from any other shell — and `hl-smi`'s compute-process table shows no owner, because
we run in a PID namespace and cannot see other processes anyway. This cost a false "leaked HBM /
needs a root reset" diagnosis once (`STATUS.md` → "Operational hazard", and
"Shutdown discipline, re-learned").

The three-part rule, as implemented:

1. **Stop the job, not just the launcher pid.** `shell-kill <job>` (or kill the process group),
   then wait. The drivers do: `kill -TERM` the entrypoint pid from `process.json`, poll up to
   120 s, `kill -KILL`, then `pkill -TERM -P "$LAUNCHER"`, `kill -TERM "$LAUNCHER"`, `wait`.
2. **Require the 768 MiB baseline.** The launcher itself refuses to release the module leases
   until every owned module reports `memory_mib <= 1024 and utilization == 0`, and writes
   `evidence/<run>/device-release.json` with `healthy_cards_idle`, `target_memory_mib: 768`,
   `admission_ceiling_mib: 1024`, `checked_before_unlock: true`. Its own comment is the rule:
   *"an observation timeout is not permission to unlock a still-allocated module or reset a
   shared device"* (`tools/run_deepseek_v41.py:721-732`). The measured release, drained
   module-by-module from 86,057 MiB to 768 MiB, is in `evidence/run/device-release.json`
   (13 samples, last one 768 MiB / 0 % on all four).
3. **Require a working acquire.** A plausible memory number alone is not sufficient:

```bash
hl-smi --query-aip=index,module_id,memory.used --format=csv,noheader
"$CASE/venv/bin/python" -c "
import torch, habana_frameworks.torch
x = torch.randn(8, 8).to('hpu'); torch.hpu.synchronize(); print('    acquire OK')"
```

**Never sample `hl-smi` during startup or teardown.** Immediately after a kill the modules read
a transient **98,304 MiB (full)** and an acquire fails `synStatus=8`; ~90 s later the baseline is
768 MiB and the acquire succeeds (`logs/phase9-v2-validation.log`, `STATUS.md`). If an acquire
genuinely fails after a clean release, `hl-smi -r -i <pci>` is the only recovery and it needs
root — check for a running job first.

### 4.5 The fused path: what breaks, and why the split is not the default

With `VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP=0` and `VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION=0`
the engine serves correct text on the V2 async runner **and** on the `--no-v2` fallback, with
byte-identical token ids. With the entrypoint's own defaults (both `1`) it returns id `0` for
every step, and the V2 runner additionally dies in warmup with
`Device loop repair produced a nonfinite hidden state`.

| measurement | flags at entrypoint defaults | both `0` |
|---|---|---|
| token ids | 16 × `0` (constant bos) | 16 distinct, all in range |
| `/v1/completions` | 200, `text: ""` | 200, real text |
| `/v1/chat/completions` | 200, `content: None` | 200, `content: 'Paris'`, `finish_reason: stop` |

The entrypoint forces these two with `os.environ.setdefault` (`deepseek_v41.py:126,128`), so an
explicit environment value wins. On this build the fused result is non-finite, the argmax
collapses to `0`, and id `0` is bos — which decodes to `''`. **So a serving API that was
working perfectly looked like a broken model.**

Two weak signals, worth knowing before diagnosing this again: empty text proves almost nothing
(ids 0/1/2, the whole 1,280-wide added-token range 128000–129279, and any out-of-vocab id all
decode to `''` — `logs/phase9-token-decode-table.log`); and the reasoning stream is separate
(`/v1/chat/completions` reports `reasoning_tokens` and puts the answer in `content`).

#### Where the defect actually is

`WOA_OUTPUT_ROUNDTRIP=1` reroutes **two** C1 decode chains, not one. The composed MLA op
(`custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2`) is **bit-exact against the
unfused chain** in twelve geometries (0 bf16 mismatches, `max_abs 0.0`) and is exonerated. The
chain that runs first at boot is the **ratio-0 layer handoff**
(`custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2`, `ops/deepseek_v41_paged_attention.py:1637`),
whose output is all-NaN; the first non-finite stage inside it is
`custom_deepseek_v41_woa_scale_dense_quant_gaudi2`, whose `tokens == 1` branch indexes the
product's 1024-lane channel axis instead of the `groups*8` tile owners, so it emits an
uninitialised tile and its store runs ~64x past the end of its output row. Full contract,
evidence and verdict: `notes/composed-graph.md`.

#### The split, and the negative result

A local patch (`notes/composed-graph.md` §6, applied in the working tree) gives that handoff its
own switch, `VLLM_HPU_DSV41_WOA_DECODE_HANDOFF`, default `0`, and gates `_can_fuse_mla_woa` on
it. `woa_output_roundtrip` then controls only the fused wo_a **producer** and the prefill output
projection that requires it. With it, the repo's own shipped configuration runs:

```bash
# measurements/woasplit/run-split.sh -- conformant, and NOT faster
VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP=1 VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION=1 \
VLLM_HPU_DSV41_MHC_RRMS_POST=0 VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS=16384 \
  ... # VLLM_HPU_DSV41_WOA_DECODE_HANDOFF deliberately unset -> 0
```

Measured in one cold lifetime (`measurements/woasplit/FINDINGS.md`):

| | `WOA=0` (the reporting arm) | split |
|---|---|---|
| decode ITL median, official | 9.365 ms | **9.277 ms** |
| decode, greedy 16K | 13.03 ms | **13.03 / 13.01 ms** |
| prefill, rounds 1+ | 8,818 tok/s | **8,630 tok/s (0.979x)** |
| 32-token greedy ids, identical prompt | reference | **differ, first at index 7** |
| boot / compiles / non-finite | serves | serves, **0** compiles, **0** non-finite |

So the split is **safe and conformant but buys no speed**, and it is **not fidelity-neutral**:
`WOA=1` also swaps the decode wo_a *producer* (`woa_fp8` -> `woa_fp8_roundtrip`, the group-32
BF16 codec), which changes ~94 % of that operand's lanes by about one bf16 ULP and flips
near-tie argmaxes after 40 layers. The sequences resynchronise (14 consecutive ids identical
from index 11/13), so it is quantisation noise rather than a structural break.

The earlier "~9 % of a 16K prefill" figure came from multiplying a single-recipe microbench
saving by 40 layers; it is an extrapolation and it does not survive. The step is
device-occupancy-bound, so removing work from one span frees its leave-one-out, not its own
union. **Keep the reporting arm at `WOA=0` / `PREFILL=0`.** Adopting the split would be a
conformance/fidelity decision, not a performance one; pinning the decode producer too (one
line, `project_output`) makes it id-identical, untested.

**Do not read the four-arm bisect as "there might be a flag combination that works".** It
collapses to two arms: `PREFILL_OUTPUT_PROJECTION=1` requires `WOA_OUTPUT_ROUNDTRIP=1` plus
`PREFILL_REGIONS`, `WO_A_FP8`, `PREFILL_VECTOR_QUANT` and `ATTN_DENSE_FP8`
(`ops/deepseek_v41_prefill_regions.py:70-80`, a `requirements` dict that raises), so arm C is
refused at model construction with `Compiled prefill output projection requires: wo_a output
roundtrip`. Arms B and D both die at boot in `_initialize_kv_caches`
(`measurements/fusedbisect/FINDINGS.md`). The split is the only way to run arm D.

#### The flag audit, so this is not re-derived

`scripts/flag-conformance.py` is device-free and compares three effective sets: `envs.py`'s
declared defaults, the entrypoint's own fastpath tables for TP4×PP1 / DSpark off / V2 on, and
our launch. Result, in `measurements/woasplit/flag-conformance.json`:

* **108 effective keys, 103 on, 5 off.**
* **Our launch differs from the entrypoint's intent in nothing.** The only keys we set that the
  entrypoint does not manage are `MHC_RRMS_POST=0` and `PREFILL_COMPUTE_TOKENS=16384`.
* **98 keys** where `envs.py` declares `0` and the entrypoint selects `1` — `envs.py` is not the
  configuration; the effective set is the union of the tables and `envs.py`'s own non-zero
  defaults (89 keys, 40 of them boolean on).
* Effectively off: `DSPARK`, `PACKED_PP`, `NATIVE_PP_COPY`, `PREFILL_PP_WAVEFRONT`,
  `PREFILL_MXFP4`.
* One internal conflict: `PREFILL_PP_WAVEFRONT` is `1` in the table (`deepseek_v41.py:80`) and
  forced to `0` by the `pp == 1` list (`:406`). PP-only, inert at PP1.
* The three `VLLM_HPU_TP2_*` keys are **not** a TP2 leak at TP4: `GRAPH_REPLAY` requires
  `TP2_STATIC_GROUP_PLAN and TP2_PREPARED_COMM` (`ops/deepseek_v41_config.py:196-197`) and
  `V2_SEGMENTED_PREFIX` requires `TP2_NATIVE_JOINT_PLAN` (`:82`).

Consequence for the 1.15× decode gap: **configuration is not it.** No flag we run differs from
the repo's intent, so the gap is not a deviation of ours; the search moves to the sampler branch
(§6.4), the batch-path stall and the unattributed device union of a decode step.

---

## 5. The measurement protocol

### 5.1 The client is the instrument, and its definition is fixed in its own docstring

`scripts/client-measure.py` speaks **only** the OpenAI HTTP API and computes every number from
arrival timestamps, so a disagreement with an in-repo tool is evidence about the *instrument*,
not about the model. The definition (from its docstring, agreed before any number existed):

* **prompt** — an explicit token-id list of exactly N, sent as `{"prompt": [...]}` on
  `/v1/completions`. Length is exact and never depends on re-tokenising a decoded string
  (`prompt_token_ids` alone is rejected: *"Either prompt or prompt_embeds must be provided"*).
* **nonce** — `none` (identical prompt every round), `per-round` (last 8 ids replaced),
  `distinct` (a distinct **whole** prompt of the same length, sliced from a long real corpus on
  a stride of `length`, so two rounds never overlap), `per-stream` (**withdrawn** — replacing
  the last 8 ids with arbitrary in-vocab ids made the model emit EOS after 1–5 tokens, i.e. it
  changed the *workload*, not just the cache state).
* **TTFT** = `t_first − t_send`, with `t_send` taken **before** the POST on a pre-warmed
  connection.
* **ITL** = intervals **between token events**; the first interval (send → first token) is TTFT
  and is excluded. A chunk may carry more than one token, so a chunk contributes one interval;
  a token-weighted mean is reported alongside.
* **aggregate tok/s** across N streams = `sum(completion_tokens) / (max(t_last) − min(t_send))`
  — includes TTFT.
* **per-stream steady rate** = `completion_tokens / (t_last − t_first)` — excludes TTFT. Both are
  reported; neither is presented as the other.
* **`cached_tokens`** from `usage.prompt_tokens_details.cached_tokens`, recorded **per stream**.
* **EOS-short** — `finish_reason == "stop"` with fewer tokens than `max_tokens` is flagged with
  its actual count, never averaged as if full length was produced.
* **spread** over measured rounds only: median, min, max, `(max−min)/median`, MAD.

Two client bugs are worth knowing because they produce plausible-looking wrong numbers, both
fixed and commented in the source: taking `t_send` on the first parsed chunk made TTFT
identically 0 and silently turned the aggregate into a decode-only rate; and detecting a token
event by non-empty *text* dropped every special-token chunk (they decode to `''`).

The in-repo tools are **witnesses only**, and only on cells they can run:
`tools/benchmark_deepseek_v41_throughput.py` hard-rejects any prefill length other than
8191/8192 and any decode length other than 511/512, so it measures a *different prompt length*
from our cells — agreement is evidence about calibration, never a cell value. Its prefill mode
is not used at all: `--min-cached-tokens` asserts cache **hits**, the opposite of gate 3.

### 5.2 The cache-epoch rule

`VLLM_CACHE_ROOT` is a **disk** cache under `$CASE/cache/vllm`, and
`measurements/conformance1/method.md` records that it persists across server restarts — so, on
that account, a cell can be silently warm even after a reboot. Whether restart-persistence is
what produced the fake match in our own logs is **not established** (§9); what *is* established
is that a warm cell fakes a match and that `cached_tokens` exposes it. Conformance run 1's C1
reported **8,774 tok/s** that was really a cache hit (`cached_tokens` 16,256 of 16,384); the
honest cold number for the same length is **419 tok/s**
(`measurements/conformance1/method.md` §"Cache epoch"). The rule, which is a **gate, not a
footnote**:

> **Gate 0, every uncached cell, before its number is accepted:**
> 1. the run declares a **cache epoch** — either `VLLM_CACHE_ROOT` is cleared before the boot,
>    or every cell uses a slice seed never used before (recorded in the cell JSON); **and**
> 2. `cached_tokens == 0` is **asserted** on every round of every uncached cell; a non-zero
>    value voids the cell, and the run records the epoch it was in.

Implement it exactly like this (`measurements/scaling1/run.sh:24-27`):

```bash
CACHE_EPOCH=$(date -u +%Y%m%dT%H%M%S)
rm -rf "$CASE/cache/vllm"/*
```

and rely on the assertion, which the client enforces itself:

```python
nonzero = [c for c in cached_all if c]
if args.kind == "prefill":
    cell["gate3_cached_tokens_zero"] = not nonzero
    cell["void"] = bool(nonzero)
elif args.concurrency > 1:
    cell["gate1_width_cached_tokens_zero"] = not nonzero
    cell["void"] = bool(nonzero)
```

Every `cell.json` therefore carries `cached_tokens_observed`, `void`, and the gate booleans. The
**never-used slice** half is what makes a *cold* cell repeatable: the corpus is a 1.51 M-id text
(`measurements/fusedbisect/corpus/corpus.txt`, 5,804,310 B), sliced on a stride of the prompt
length, with each cell given a disjoint base seed (3000/3100 for the two prefillcold cells;
7000/7100/7200 and 7300/7400/7500 for alignandab; 8100/8200/8300 for conformance2).

Two failure modes this rule caught in our own data, both of which would have been quoted as
results: an 11k prefill cell reporting `cached_tokens` 896 and 10,880 on two of three rounds
(rotation of one repetitive paragraph still shares most of its content — fixed by slicing a long
real corpus, `measurements/run3/SUMMARY.md`), and the 16K halo-verify cells reporting 8,064 /
12,160 / 16,256, which is why their id-identity check is **compromised and stated as such**
rather than claimed.

### 5.3 Why round 0 is reported separately

Two distinct reasons, both measured, and both about **first-request cost, not steady state**:

* **Prefill, cold recipe cache:** round 0 carries the shape compile. In the earlier campaign
  that was 35–40 s. In `measurements/prefillcold` the recipe cache was already warm from the
  bisect's 16K cells, so round 0 is only first-request overhead — **+150 ms** at chunk 16,384
  (1,963.1 vs 1,806.1 ms) and **+1,117 ms** at chunk 8,192 (4,276.0 vs 3,158.8 ms). The finding
  says plainly that round 0 "no longer contains a cold compile and must not be read as one".
* **Decode:** round 0 carries ~1.3 ms of warmup — `measurements/fusedbisect` arm A: round 0
  (discarded) **14.37 ms**, rounds 1–2 **13.02 / 13.04 ms**. That is exactly why it is separated.

So: prefill cells run `--rounds 3 --discard 0` and report round 0 in its own column; decode
cells run `--discard 1` and drop it; the conformance cell runs `--rounds 3 --discard 0` because
its round 0 is the one that would carry a compile risk, and it is reported in its own row.
Round 1 of the conformance cell produced only 66 tokens (natural EOS) and its 11.262 ms/token is
**not comparable and must not be averaged in** — a short output is dominated by per-step fixed
cost, the same reason short-output measurements mislead everywhere else in this campaign.

### 5.4 The four cells that produce the headline numbers

One boot per run directory, one lifetime, nothing restarted, all cold, all with
`--nonce distinct --corpus measurements/fusedbisect/corpus` (unless stated). **The launch
parameters differ between the runs, and each cell is only comparable to its own run:**

| run | `--max-num-seqs` | `--max-num-batched-tokens` | `VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS` |
|---|---|---|---|
| `conformance2` (H1) | 32 | 16384 | 16384 |
| `prefillcold` (H2) | 32 | `$CHUNK` (16384 or 8192) | `$CHUNK` |
| `alignandab` (H3) | 32 | 16384 | 16384 |
| `run3` (H4) | **16** | **8192** | not set — the boot is `bash scripts/run-phase9.sh` with only the two fusion flags overridden |

That last row matters: the width table in §6.2 was measured at `max-num-seqs 16` and
`max-num-batched-tokens 8192`, so it is a property of *that* boot, not of the 32/16384
configuration. Re-measure before comparing across the two.

**H1 — the conformance cell: prefill and decode at official sampling, natural EOS.**
`measurements/conformance2/cells/conformance/`, driver
`measurements/conformance2/run-conformance.sh`:

```bash
$CLIENT "$RUN/cells/conformance" --cell conformance --kind decode --url $URL --model $MODEL \
  --prompt-tokens 16384 --concurrency 1 --rounds 3 --discard 0 --max-tokens 1024 \
  --nonce distinct --slice-seed 8300 --corpus $CORPUS --suffix-text "$SUFFIX" \
  --temperature 1.0 --top-p 0.95 --seed 42 --return-token-ids
```

with `SUFFIX=$'\n\nQuestion: State in one sentence what the passage above is about.\nAnswer:'`
(a fixed 16-token instruction; the variable part is shortened by exactly 16 ids so the total is
still 16,384 — that is what lets the model reach a **natural EOS** instead of running into
`max_tokens`). Official sampling = `T=1.0, top_p=0.95, seed=42`; that is the arm the repo's
recorded baseline uses.

**H2 — cold 16K prefill, chunk 16,384, three rounds.**
`measurements/prefillcold/cells/c16384-16k/`, driver `measurements/prefillcold/run-prefill-cold.sh`:

```bash
CHUNK=16384 BASE=3000 bash run-prefill-cold.sh
# inside:
$CLIENT "$RUN/cells/c${CHUNK}-16k" --cell "c${CHUNK}-16k" --kind prefill --url $URL --model $MODEL \
  --prompt-tokens 16384 --concurrency 1 --rounds 3 --discard 0 --max-tokens 1 \
  --nonce distinct --slice-seed "$BASE" --corpus $CORPUS
```

The matching contrast cell is `CHUNK=8192 BASE=3100` in its **own lifetime** — chunk size is a
launch parameter, so the two chunk sizes cannot share a boot.

**H3 — cold decode at official sampling, independent lifetime.**
`measurements/alignandab/cells/official/`, driver `measurements/alignandab/run-align-and-ab.sh`:

```bash
$CLIENT "$RUN/cells/official" --cell official --kind decode --url $URL --model $MODEL \
  --prompt-tokens 16384 --concurrency 1 --rounds 2 --discard 1 --max-tokens 256 \
  --nonce distinct --slice-seed 7400 --corpus $CORPUS --return-token-ids \
  --temperature 1.0 --top-p 0.95 --seed 42
```

Its purpose is to close the one caveat in `measurements/samplingab` (where only the greedy arm
was cold). It reproduces H1's decode rate from a different lifetime, corpus and slices.

**H4 — the width-1 decode baseline, which is the only batch number that behaves like a
throughput number.** `measurements/run3/probe/w1/`, driver `measurements/run3/run.sh:71-73`:

```bash
$CLIENT "$RUN/probe/w1" --cell w1 --kind decode --url $URL --model $MODEL --prompt-tokens 1024 \
  --concurrency 1 --rounds 3 --discard 1 --max-tokens 128 --nonce none
```

`--nonce none` is deliberate here: this cell is warm-cache **by construction**
(`cached_tokens=896` every round after the first) and that affects its TTFT and aggregate, not
its decode rate. Its width-2..16 siblings come from the same loop (`--concurrency $w`) and from
`measurements/run3/run-util-probe.sh` for w8/w16 (`--rounds 2 --discard 1 --max-tokens 32`).

**Supporting cells** (same protocol, quoted in §6): `conformance2/cells/greedy-cold` and
`topk1-cold`/`topk1-warm`; `alignandab/cells/greedy-1`,`greedy-2`; `prefillcold/cells/c8192-16k`;
`run3/cells/c1`,`c6`,`c7`,`c8`,`c10`; `prefillbucket/cells/*`; `haloverify/cells/*`;
`scaling1/cells/*`; `fusedpath/*` (no server at all).

### 5.5 The gates a number must pass before it is quoted

From `measurements/run3/method.md` and `measurements/conformance1/method.md`, unchanged through
the campaign:

1. **0 compiles inside the measured block.** Count before and after each cell:
   `grep -c -i 'compileGraph\|Compiling' "$CASE/evidence/run/run.log"`. For the last measured
   boot the count is **0 for the entire log** (`evidence/run/run.log`), and `recipe cache miss`
   is also 0.
2. **The model is generating.** A sanity request whose token ids are not constant
   (`tools/request_deepseek_v41.py`, `return_token_ids`), so an empty-output run cannot be
   mistaken for a fast one. Passed: `ids=[2581, 1309, 3287, 16, …]`.
3. **Every uncached cell reports 0 cached tokens** (gate 0 above).
4. **Box conditions.** `hl-smi` at the 768 MiB baseline before launch; four modules held during;
   after stopping the job, baseline re-asserted **and** a fresh acquire succeeds; no other job
   in the space while the block runs.
5. **Config recorded** — every key of §4.2/§4.3 plus the effective environment the launcher
   actually delivered (`evidence/<run>/process.json`, `runtime-profile.json`).
6. **Natural EOS or an honest flag.** `finish_reason: stop` with the actual output length
   recorded; a truncated output is flagged, never averaged as if it were a full decode.

**Invalidation rules, declared before each run:** any compile inside the block; a co-tenant job
or an unstopped server from an earlier run; a cell that hit the prefix cache; an EOS-short
output averaged as a full decode; a device reading taken during startup or teardown; an
unexplained disagreement between the in-repo witness and our client.

---

## 6. The numbers to expect

### 6.1 Headline

| quantity | ours | file it comes from | recorded baseline | ratio |
|---|---|---|---|---|
| **prefill tok/s**, 16,384-token prompt, one 16,384 chunk, cold, official sampling | **8,544 / 8,813 / 8,824** (rounds 0/1/2) | `measurements/conformance2/FINDINGS.md`; raw `.../cells/conformance/cell.json` | 8,029–8,915 | **inside the band** |
| **prefill tok/s**, same, steady median over rounds 1–2 | **9,035** (median 1,813.5 ms) | `measurements/prefillcold/FINDINGS.md` | 8,029–8,915 | at/above |
| **decode ms/token**, official sampling (T=1.0, top_p=0.95, seed 42), 16K context, whole output to natural EOS | **9.478 / 9.509** (rounds 0 and 2); ITL median **9.365**, p90 10.40, n=1,701 | `measurements/conformance2/FINDINGS.md` | 8.239 | **1.15×** |
| **decode ITL median**, official sampling, independent lifetime, 256 tokens | **9.364 ms** (steady 106.61 tok/s) | `measurements/alignandab/FINDINGS.md` | 8.239 | **1.14×** |
| **decode ITL median**, greedy, 16K context, 256 tokens | **13.544** (conformance2) / **13.065**, **13.003** (alignandab) / **13.03** (fusedbisect) | three files above | 9.975 | **1.30×** |
| **prefill tok/s**, chunk 8,192 for the same 16K prompt | **5,175** (median 3,166.5 ms) | `measurements/prefillcold/FINDINGS.md` | — | 16,384 chunk is **1.75×** faster |
| **width-1 decode**, 1,024-token prompt, 128 tokens | ITL **12.897 ms**, steady **74.75 tok/s**, aggregate **57.78 tok/s** | `measurements/run3/probe/w1/cell.json` | — | — |

### 6.2 The batch table — read it as a defect, not a curve

`measurements/run3/probe/w{1,2,3,4,8,16}/cell.json` and `measurements/run3/SUMMARY.md` §"Step 2".
**Measured at `--max-num-seqs 16`, `--max-num-batched-tokens 8192`** (a `run-phase9.sh` boot with
only the two fusion flags overridden), so this table describes that configuration:

| width | aggregate tok/s | per-stream steady tok/s | ITL median | ITL p90 |
|---|---|---|---|---|
| 1 | 57.78 | 74.6 – 74.8 | **12.90 ms** | 13.6 ms |
| 2 | 10.23 | 5.2 – 5.4 | **186.10 ms** | 210.2 ms |
| 3 | 11.82 | 3.8 – 4.1 | **247.61 ms** | 270.6 ms |
| 4 | 11.83 | 3.0 – 3.1 | **329.09 ms** | 337.4 ms |
| 8 | 10.79 | 1.4 – 1.6 | **667.66 ms** | 690 ms |
| 16 | 11.46 | 0.8 – 0.8 | **1,259.12 ms** | 1,290 ms |

The aggregate is flat while per-stream ITL grows monotonically — **per-row cost, not a bucket
ladder** (a wider bucket would give flat ITL and a *rising* aggregate). The 1→2 step is a
discontinuity (+173 ms for one added row, 14×) and above 2 it is a slope (~+77 ms per added row
from w2 to w16). The device-side probe says the step is **waiting, not working**: mean
`utilization.aip` is **8.4 %** at w1 and **9.0 %** at w2 while ITL is 14× worse, with 0 compiles
during either cell and `"fallbacks": 0` in every sampling-completion summary
(`measurements/run3/SUMMARY.md`, `logs/util-sample-w{1,2}.txt`). **Do not average these into a
scaling curve, and do not quote a width>1 number as a serving result.** The repo's own caveat
rides on every one of them (`docs/deepseek_v41_concurrent_decode.md:22-24`: full-model outputs at
larger concurrencies still differ from the saved reference; batch is not qualified).

### 6.3 The prefill length anomaly — what is solid and what is not

| T (chunk 16,384) | tok/s | source |
|---|---|---|
| 8,192 | 6,465 | `measurements/prefillbucket/FINDINGS.md` |
| 10,240 | **874** (worst measured anywhere) | `measurements/alignandab/FINDINGS.md` |
| 11,000 | 2,178 / 1,722 (two lifetimes, 26 % apart) | both files |
| 11,264 | 3,478 | `measurements/alignandab/FINDINGS.md` |
| 12,288 | 5,128 | `measurements/prefillbucket/FINDINGS.md` |
| 16,384 | **8,985 / 9,035** (reproducible to 1 %) | `prefillbucket` and `prefillcold` |

Three explanations were tested and all three refuted: "a partial chunk costs a full chunk"
(16,384 in one chunk is the cheapest per token); the five-bucket token-ownership gate (12,288 is
not a bucket and is fast); and 512-alignment (10,240 is aligned and slowest). **The one solid
serving rule is: full 16,384-token transactions are reliably ~9,000 tok/s; anything shorter is
erratic.** The surviving candidate — a content-dependent cost in the learned indexer's
selected-candidate count — is a hypothesis with a falsifier, **not a measurement** (§9).

### 6.4 The decode-side decomposition (so you can tell a regression from a routing change)

Same context, same length, same cache state, one lifetime — the sampler branch is worth
3.6 ms/token (`measurements/conformance2/FINDINGS.md`, `measurements/alignandab/FINDINGS.md`):

| branch | ITL | how to select it |
|---|---|---|
| bounded nucleus | **9.36–9.4 ms** | `T=1.0, top_p=0.95, seed=42` |
| argmax / greedy | **13.0 ms** | `T=0.0` |
| unbounded inverse-CDF | 21.8 ms | `T>0, top_p=1` |
| `top_k=1` | **9.348 ms** | `T=1.0, top_k=1` — fast, but **not** a greedy replacement: ids match for 25 tokens then diverge |

The greedy penalty is **request-parameter-driven, not content-driven**: the greedy arm and the
`topk1-warm` arm send the *same prompt* in the same lifetime and measure 13.544 vs 9.348 ms. It
is also **not** the full-vocabulary fallback — the runner's audit reads `bounded_steps: 255` with
`fallbacks: 0` (occasionally 1). Inherent share: the reference's own greedy/nucleus ratio is
9.975/8.239 = 1.211; ours is 1.449, so the excess we own is ~20 %.

Also measured, and useful: **decode ITL is prefix-insensitive** — `topk1-warm` (cached 16,256)
and `topk1-cold` (cached 0) both measure **9.348 ms** to three decimals. Every earlier A/B that
assumed this is now supported by a measurement.

---

## 7. What is NOT reproducible as-is

### 7.1 The additive `deepseek_v41_unique` bundle — registration half only

`csrc/deepseek_v41_unique/` is a **second, additive** op bundle. This install does **not** build
it, and the bundle the entrypoint actually wants is three things, of which only one is buildable
from what ships:

| piece | buildable here? |
|---|---|
| `hpu_dsv41_unique_pt2…so` (the registration extension, `pytorch/setup.py`) | **yes** — a host-only `CppExtension`/`BuildExtension` compile, no device |
| `libdeepseek_v41_unique_kernels.so` (**204 TPC kernels**, `CMakeLists.txt`, `tpc-clang -march=gaudi2`, `-Wall -Werror` on both compilers) | host-only in principle, **never built here**; `-Werror` means any warning in the container's `tpc-clang 2.7.33` is a hard failure |
| `deepseek_v41_unique_build.json` (schema 1, `base_kernel_sha256`, `binaries` sha256 map) | **no from-scratch builder exists.** `tools/record_deepseek_v41_additive_build.py` *updates* an existing manifest; `tools/install_deepseek_v41_dspark_addon.py` composes from a parent that already has one. A from-scratch manifest must be hand-authored |

And the bundle is pinned to **our exact base kernel**:
`run_deepseek_v41.py:88-105` refuses to start unless every `binaries` entry hash-matches and
`GC_KERNEL_PATH` is either the base or the unique kernel database, and
`entrypoints/deepseek_v41.py:479-490` refuses unless
`addon["base_kernel_sha256"] == sha256(our libdeepseek_v4_gaudi2_kernels.so)`. **It cannot be
borrowed from another install.**

Why it matters at all: `entrypoints/deepseek_v41.py:791-795` loads the bundle **unconditionally
when any optional op flag is set**, not when an op is missing:

```python
if optional:
    from vllm_gaudi.ops.deepseek_v41_unique_experts import load_unique_expert_operators
    load_unique_expert_operators()
```

`VLLM_HPU_DSV41_MHC_RRMS_POST` defaults to **1**, so the *default* configuration tries to load a
bundle nobody built and dies with `FileNotFoundError: deepseek_v41_unique_build.json`. `env.sh`
sets it to 0 purely to get past that. The documented benefit of the flag is
**0.000290156 ms/token = 0.29 µs on a ~13 ms step (0.002 %)** — a correctness-preserving cleanup,
not a speedup. Its operator `custom_deepseek_v41_mhc_rrms_post_gaudi2` is **already registered by
our base extension** and its TPC kernel is already embedded in
`libdeepseek_v4_gaudi2_kernels.so`; the only blocker is that unconditional loader. A three-line
gate — load only if an optional op is genuinely missing — recovers it for free; the patch sketch
is in `notes/additive-bundle.md` §3. The bundle's real content is ~80 default-off DSpark fusions,
which is a build + per-flag qualification campaign of its own.

### 7.2 The 9-line fix to `tools/microbench_deepseek_v41_woa_roundtrip.py`

The tool is a **TP2 × PP2 microbench**. It hardcodes `torch.randn(tokens, 4, 4096)` — a G4
geometry — while a TP4 tree is G2, because `groups` comes from the *activation*
(`csrc/deepseek_v4/pytorch/hpu_dsv41_woa_fp8_pt2.cpp:22 groups = x.size(1)`, then
`:32-34` demands a weight of exactly `[groups,4096,1024]`). Against a TP4 sidecar it fails the
op's own `TORCH_CHECK` with *"wo_a requires prepared Gaudi2 FP8 [G,4096,1024]…"* even though the
sidecar is exactly right. It also assumes `pp_rank = args.layer // 20` (`:35`), which is only
meaningful at PP2.

**This fix is applied to the working tree of `src/1Cat-vLLM-Gaudi` and is not committed
anywhere upstream.** Nine added lines at `tools/microbench_deepseek_v41_woa_roundtrip.py:78`:

```python
    # The wo_a group count is a property of the prepared tree, not of this tool: the op's
    # meta() takes `groups` from x.size(1) (hpu_dsv41_woa_fp8_pt2.cpp:22) and then demands a
    # weight of exactly [groups,4096,1024] (:32). G = 8 // tp_size, so a TP2 tree is G4 and a
    # TP4 tree is G2. The literal 4 this line used to carry made the tool a TP2-only
    # microbench: against a TP4 sidecar it failed the op's check with
    # "wo_a requires prepared Gaudi2 FP8 [G,4096,1024] and F32 [G,1,1024] channel scales"
    # even though the sidecar was exactly right. See notes/woa-sidecar-question.md.
    groups = 8 // shard.tensor_parallel_size
    inputs = [torch.randn(args.tokens, groups, 4096).bfloat16().to("hpu") for _ in range(2)]
```

Two further invocation facts, without which the tool still fails and which are **not** in the
tool: `--woa-sidecar` takes the sidecar **directory**
(`weights/prepared/sidecars/wo_a_fp8`), not a shard file (that gives `NotADirectoryError`); and
`--library` must be `hpu_dsv4_sparse_attn_pt2…so`, the only artifact that registers the op —
`libdeepseek_v4_gaudi2_kernels.so` carries the TPC kernel objects but registers no torch ops
(`notes/woa-sidecar-question.md` §4-5, `measurements/fusedpath/run-isolated.sh`).

Re-apply it by hand (it is 9 lines) or diff the working tree:

```bash
git -C "$CASE/src/1Cat-vLLM-Gaudi" diff tools/microbench_deepseek_v41_woa_roundtrip.py
```

### 7.3 The halo patch — applied to the working tree, not committed

**Where it is:** the patch file is `$CASE/notes/halo-fix.patch` (14,398 B), and it is **applied,
uncommitted**, to the 1Cat clone. `git -C src/1Cat-vLLM-Gaudi status --porcelain` shows
`M vllm_gaudi/ops/deepseek_v41_prefill_sequence_state.py` and
`?? tests/unit_tests/ops/test_deepseek_v41_halo_tail_ownership.py` (plus a stray empty `cases/`
directory created by a tool run — harmless). Its diffstat is `278 insertions(+), 3 deletions(-)`
across those two files. Written up in `notes/halo-fix.md`.

**Re-apply (or revert) it:**

```bash
cd "$CASE/src/1Cat-vLLM-Gaudi"
git apply --check /home/nat/DeepseekFlash/cases/1cat/notes/halo-fix.patch     # before applying
git apply         /home/nat/DeepseekFlash/cases/1cat/notes/halo-fix.patch
git apply --reverse --check /home/nat/DeepseekFlash/cases/1cat/notes/halo-fix.patch   # already-applied test
```

**What it fixes.** `replicate_owned_tail`
(`vllm_gaudi/ops/deepseek_v41_prefill_sequence_state.py:59-67`) collected the decoder halo from
the **last token owner only** and guarded with `retained <= local.shape[0]`, i.e. "rank 3's
shard contains the whole 4096-row halo". That holds only when `T/4 >= 4096`, i.e. `T >= 16384`.
The halo is a **global** row interval, so for any smaller transaction it straddles token owners
(two owners at T=8192, four at T≤4096) and the call raises
`ValueError: Replicated decoder tail must fit the final TP4 token shard`
(`ops/deepseek_v41_prefill_sequence_state.py:62`). The call site is entered whenever
`cut = T − 4096 > 0`, so the dead window is **`4096 < T < 16384`**: 8,192 and 12,288 die with
HTTP 500 and then kill the engine; 16,384 sits exactly on the boundary and works; ≤ 4096 never
enters the branch. **This is a correctness/robustness defect on the long-context path, not a
performance defect.**

**The edit** (the general case, in the one function all three call sites use):

```python
    if dist.get_world_size(group) != 4 or retained < 1:
        raise ValueError("Replicated decoder tail requires a positive TP4 row count")
    if retained > local.shape[0]:
        # A short transaction splits the halo across token owners (at TP4 the
        # 4096-row halo spans two owners at T=8192 and four at T<=4096), so
        # reassemble the transaction and take its tail instead of demanding the
        # whole halo inside the final shard.
        full = gather_tokens(local.contiguous(), group=group)
        if retained > full.shape[0]:
            raise ValueError("Replicated decoder tail exceeds the full TP4 transaction")
        # Copy so every rank ends up holding exactly ``retained`` rows instead
        # of a view that keeps the whole reassembled transaction alive through
        # the decoder suffix.
        return full[-retained:].clone()
```

`retained <= shard` keeps the existing single-broadcast fast path **byte-identically**, so
T=16384 is unchanged; `retained > shard` reassembles and takes the tail — the same operation
`causal_context_tail` already performs for the DSpark draft ring one function lower. The branch
condition is a pure function of shapes identical on all four ranks, so the collective stays
matched.

**Verified on device** (`measurements/haloverify/`, cold cells, `cached_tokens == 0` asserted):
8,192 **works** (was HTTP 500 + engine death), 12,288 **works**, 16,384 works. 0 × 500s, 0 ×
`Replicated decoder tail` errors. The 16K id-identity check is **compromised and stated as
such**: both 16K cells reported non-zero `cached_tokens` (8,064 / 12,160 / 16,256), and no
pre-patch 16K id list was ever captured, so that comparison cannot be completed
retrospectively (`measurements/scaling1/FINDINGS.md` §4).

**Cost.** At T=16384 the patch saves 0 bytes (fast path unchanged). In the newly-working window
it all-gathers once per final transaction: 8192 × 40 KiB = 335.5 MB per rank at T=8192, ~0.17 ms
— against a transaction that previously returned HTTP 500. The replication at T=16384 was
already ~0.23–0.28 ms, i.e. ~0.1 % of a 263 ms/token step, so **the halo was never the decode
cost** (`notes/halo-fix.md` §6).

### 7.4 Also not reproducible as-is

* **The two chunk sizes in one lifetime.** `--max-num-batched-tokens` and
  `VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS` are launch parameters; the 16,384-vs-8,192 comparison
  is two boots (`measurements/prefillcmp`, `measurements/prefillcold`).
* **The reference baseline itself.** 8.239 ms/token and 8,029–8,915 tok/s are numbers from the
  repo's own docs, not from this install (§9).
* **The additive-bundle op counts and DSpark flag list.** Read from source; nothing was built or
  run (`notes/additive-bundle.md`).
* **The 4,300,160-token KV cache figure.** It is a per-boot observation, not a config output
  (§4.3).

### 7.5 The WOA split — applied to the working tree, not committed

A local patch that lets the repo's own shipped configuration run on this build. It adds one
environment switch and moves three decode-handoff gates onto it. **It is not needed for any
number in this document** and it is **not faster** (§4.5); it is here so the conformant
configuration is reproducible.

`vllm_gaudi/envs.py` — declare it (default `0`):

```python
    "VLLM_HPU_DSV41_WOA_DECODE_HANDOFF":
    lambda: os.environ.get("VLLM_HPU_DSV41_WOA_DECODE_HANDOFF", "0").lower() in ("1", "true"),
```

`vllm_gaudi/ops/deepseek_v41_paged_attention.py` — three call sites plus the predicate:

```python
# :1637  the C1 decode handoff that actually fires (native_rope on)
        if (decode and not prefill and output.shape[0] == 1 and self.native_rope
                and gaudi_envs.VLLM_HPU_DSV41_WOA_DENSE_HANDOFF and self.woa_fp8
                and gaudi_envs.VLLM_HPU_DSV41_WOA_DECODE_HANDOFF      # was: self.woa_output_roundtrip
                and getattr(self.weights.wo_b, "dense_fp8", False)):
# :1623  the DSPARK C2-C6 handoff, same substitution
# :1672  the non-native-RoPE C1 sibling, same substitution
# :1715  _can_fuse_mla_woa drops woa_output_roundtrip and takes the new switch
        return (
            positions.numel() == 1 and self.mla_mme and self.woa_fp8 and self.native_rope
            and gaudi_envs.VLLM_HPU_DSV41_WOA_DECODE_HANDOFF
        )
```

`woa_output_roundtrip` then controls only the fused wo_a **producer** (`project_output`,
`:806-819`) and the prefill output projection that requires it (`:1656-1669`).

**Verified on device** (`measurements/woasplit/`, one cold lifetime): boots with
`WOA=1`/`PREFILL=1`/handoff unset, serves, `max_model_len` 524,288, **0 compiles** after
startup, **0 non-finite** mentions, every round `cached_tokens == [0]`, box returned to 768 MiB
with a working acquire. Prefill 8,630 tok/s vs 8,818 (0.979x), decode ITL 9.277 vs 9.365 ms,
greedy 13.03 vs 13.03. The 32-token greedy ids **differ from index 7** because `WOA=1` also
swaps the decode wo_a producer; `producer_t1.py` measures that operand changing on ~94 % of its
lanes by about one bf16 ULP. Pinning the producer as well (one line in `project_output`) would
make decode byte-identical — **not implemented, not measured**.

---

## 8. Traps that cost us time

| # | trap | what it looks like | what to do |
|---|---|---|---|
| 1 | **The prefix cache persists across restarts and fakes a match.** (The *fake match* is measured; the restart-persistence mechanism is not — §9.2.) | A cell reports 8,774 tok/s where the honest cold number is 419 tok/s; `cached_tokens` 16,256 of 16,384. Warm cells are *indistinguishable* from fast ones in the rate column. | Gate 0 (§5.2): clear `VLLM_CACHE_ROOT`, use never-used slice seeds, and **assert** `cached_tokens == 0` on every round. The assertion is the guard — a restart alone is not. |
| 2 | **`.upstream.so` SONAME.** | `hpu::habana_bernoulli is already registered!` (or `habana_d2d_memcpy_other`) once a PT2 extension links the `.upstream` name alongside the primary. | In this container the three `.upstream.so` files are **real files with their own SONAMEs**, not symlinks. Replace them with symlinks to the primaries **in the venv copy only**, and re-do it after every bridge install (§2.4, `scripts/install-bridge-into-venv.sh`). |
| 3 | **The two fusion flags are one switch — and the flag reroutes two decode chains, not one.** | You try to keep the fused prefill projection and drop the fused wo_a roundtrip. | Not selectable: `PREFILL_OUTPUT_PROJECTION=1` *requires* `WOA_OUTPUT_ROUNDTRIP=1` plus four more flags (`ops/deepseek_v41_prefill_regions.py:70-80`), so the four-arm bisect collapses to A and D. Arms B/D die in `_initialize_kv_caches` with `nonfinite hidden state`; arm C is refused at construction. The reporting arm keeps both at `0` (§4.5). The defect is a single kernel's `tokens==1` branch, and a local split (§7.5) runs arm D safely — but it is **not faster**, so it does not replace the reporting arm. Do not assume the composed MLA op is the culprit: it is bit-exact (`notes/composed-graph.md`). |
| 4 | **The wo_a microbench is TP2/G4.** | `RuntimeError: wo_a requires prepared Gaudi2 FP8 [G,4096,1024]…` — which looks like a bad sidecar and is not. | The tool hardcodes 4 groups; a TP4 tree is G2. Fix `groups = 8 // tp_size`, pass the sidecar **directory** and the **sparse_attn** extension (§7.2). The sidecar itself passes every check `WoaFP8Sidecar` makes. |
| 5 | **`tpc-llvm-objdump` / `gc_interface.h`.** | The kernel build refuses, or a tool is "present" in a path you did not check. | `which tpc-llvm-objdump` → `/usr/bin/tpc-llvm-objdump`; `gc_interface.h` is at **`/usr/lib/habanatools/include/`**, not under `/usr/lib/habanalabs`. Both verified in phase 0 (§1.1). |
| 6 | **`PYTHONPATH=/root` container trap.** | `pip` dies with `PermissionError: [Errno 13] Permission denied: '/root'` **before** it reaches the network — reads like a network or index problem. | The image ships `PYTHONPATH=/root:/usr/lib/habanalabs/` and `/root` is 0700. Set `PYTHONPATH=/usr/lib/habanalabs` explicitly on every python and pip call (§1.3a). |
| 7 | **`~/.local` editable finders.** | `import vllm_gaudi` silently resolves into `/home/nat/Llms/src/vllm-gaudi` — another workspace — and your edits appear to have no effect. | `--system-site-packages` re-enables the user site. Set `PYTHONNOUSERSITE=1`, assert `site.ENABLE_USER_SITE is False`, no `~/.local` on `sys.path`, and both packages resolving inside your own tree (§2.3, §2.10). |
| 8 | **`max_num_seqs` is a start parameter.** | A cell table that varies it inside "one boot, one lifetime" — the cell cannot run. | It sizes the KV pool and the graph buckets at boot. A 16-vs-32 comparison needs **two boots**; `measurements/conformance1` skipped C3 for exactly this reason and said so. Same class: `--max-num-batched-tokens`, `VLLM_HPU_DSV41_PREFILL_COMPUTE_TOKENS`, chunk size. |

Five more that cost a failed attempt each, worth having in front of you: `BUILD_ROOT_LATEST`
must be writable or `make` deletes the freshly linked `libhcl.so`; the bridge pin must be
v1.24.1 *sources* rebased onto v1.24.2 (v1.24.1 cannot configure — no `habana_frameworks.mediapipe`);
`--checkpoint-audit` is a **file** for the shard planner and a **directory** for the entrypoint;
audit records are keyed by snapshot name while the runtime keys them by resolved basename
(LFS oid); and the only `cmake` on the box is a pip console script that dies under
`PYTHONNOUSERSITE=1`.

---

## 9. What is UNVERIFIED in this document

Stated plainly rather than smoothed over.

1. **The reference baseline (8.239 ms/token decode, 8,029–8,915 tok/s prefill, 9.975 ms/token
   greedy).** These are numbers from the repo's own `docs/dev_guide/` records. **We never
   reproduced them from a build of our own** — they are the yardstick, not a measurement, and
   every "×" ratio in §6 is against a number this install cannot re-derive.
2. **The mechanism behind "the prefix cache persists across restarts".** The *observation* is
   solid: a warm cell faked a 20× match and `cached_tokens` exposed it. But the fake match in
   our logs happened **within one lifetime** (after the warmup phase), and no log in this
   workspace demonstrates a `VLLM_CACHE_ROOT`-backed hit surviving a process restart. The
   procedure in §5.2 is correct either way because it asserts `cached_tokens == 0`; the
   *mechanism* as written in `measurements/conformance1/method.md` is unverified here.
3. **The cause of the prefill anomaly between 8k and 16k tokens.** "Content-dependent cost in
   the learned indexer's selected-candidate count" is a **hypothesis with a falsifier**, not a
   measurement: it was never tested (the falsifier — same length, two slices, per-span device
   time — requires a profile we did not take). Three rival explanations are *refuted*; the
   survivor is not established.

Secondary, and labelled where it appears: the additive bundle's 204-kernel library has **never
been built** here, so its `-Werror` risk against `tpc-clang 2.7.33` is unknown; the halo patch's
16K id-identity check is **compromised** by non-zero `cached_tokens` and cannot be completed
retrospectively (§7.3); the PSI gate that `measurements/conformance1/method.md` gate 4 asks for
was **not recorded** in any cell JSON; and `GPU KV cache size` varies per boot, so any single
figure for it is an observation, not a property.

---

## Appendix — where every number and artifact lives

| path | what it is |
|---|---|
| `manifest.json` | every input: `{name,url,pin,sha256,http_status,fetched_at,verified_by,local_path}`; 5 sources, 48 checkpoint shards, 4 small files, 50 wheels, `deviations_from_plan[]` |
| `INSTALL.md` | the narrative install, phases 0–9, plus its own trap table |
| `STATUS.md` | per-phase gate result, every surprise, every decision with its log |
| `PLAN.md` | the frozen instruction set (phases, constraints, gates) |
| `env.sh` | the environment contract (§1.5) |
| `profiles/verified-runtime-profile.json` | the runtime profile, `d1de7877…` |
| `scripts/build-hcl.sh`, `build-synapse.sh`, `build-bridge.sh`, `install-bridge-into-venv.sh` | the four build steps, each self-documenting |
| `scripts/patch_targets_pie.py` | fix #2, with its anchor check |
| `scripts/make-runtime-profile.py` | generates and hashes the profile |
| `scripts/client-measure.py` | the measurement client and its frozen definition |
| `scripts/alias-audit-records.py`, `decode-candidate-ids.py`, `compare-prepared-tree.py`, `build_manifest.py`, `check_hf_snapshot.py` | the small tools the gates use |
| `scripts/run-phase9.sh`, `scripts/phase9-diagnostics.sh` | the launch and the diagnosis sequence |
| `notes/halo-fix.md`, `notes/halo-fix.patch` | §7.3 |
| `notes/additive-bundle.md` | §7.1 |
| `notes/woa-sidecar-question.md` | §7.2 |
| `logs/phase*/…` | one file per step, observed output only |
| `evidence/run/` | the last boot: `process.json` (argv, modules, effective env), `runtime-profile.json`, `device-release.json`, `run.log` |
| `measurements/<run>/method.md` | the protocol frozen before that run |
| `measurements/<run>/cells/<cell>/cell.json` | per-round, per-stream raw records: `t_send_ns`, `t_first_ns`, `t_last_ns`, token counts, full interval list, `cached_tokens`, gate booleans |
| `measurements/<run>/FINDINGS.md` | what that run concluded |
| `measurements/fusedbisect/corpus/corpus.txt` | the 1.51 M-id corpus every cold cell slices |
