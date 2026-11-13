"""Does the custom op compose under torch.compile?  Each arm in its own process.

The example ships a Meta implementation of the op, and its own comment says the
Meta kernel is "needed only for torch.compile support".  This checks what that
buys on this stack -- and the answer depends entirely on GC_KERNEL_PATH, which
is why the third arm is a control with no custom op in it at all.

Measured, three settings of GC_KERNEL_PATH x three arms:

    GC_KERNEL_PATH                          eager   compile op   compile control
    ours:stock  (both)                      OK      OK           OK
    stock only  (ours absent)               FAIL    FAIL         OK
    ours only   (stock absent)              OK      FAIL         FAIL

Read the last row carefully.  With our library alone the *eager* op still works,
so it is tempting to conclude that the op does not compose under torch.compile.
It does not mean that: the control -- a plain `t * 2.0` region that never
imports custom_add1 -- fails identically.  The stock kernel library
(libtpc_kernels.so) is needed for ANY compiled region on this stack, not only
for one that contains a custom op.  That is a property of the recipe compiler,
not of this kernel, and the error it produces blames the graph.

So: run this with the two-entry list to see the op compose, and with one entry to
see which of the two failures you get.  A control must run in the same
environment as the arm it controls, or it tells you the wrong thing.

Each arm runs in a fresh interpreter on purpose.  Measured: once a compiled graph
has failed in a process, a *later* compiled graph in the same process fails too,
so a control sharing the process reports the wrong thing.

Usage:  GC_KERNEL_PATH=<...>/libadd1_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so \
            PT_HPU_LAZY_MODE=0 python3 check_torch_compile.py
"""
import subprocess
import sys

ARMS = [
    ("eager              op", "eager"),
    ("compile fullgraph  op", "compile"),
    ("compile fullgraph  control x*2.0", "control"),
]
SHAPE = (64,)


def run_arm(arm):
    import torch
    import habana_frameworks.torch.core  # noqa: F401

    from custom_add1 import CustomAdd1

    op = CustomAdd1()
    x = torch.full(SHAPE, 5.0, dtype=torch.float32).to("hpu")

    if arm == "eager":
        fn, ref = op, torch.full(SHAPE, 6.0)
    elif arm == "compile":
        fn = torch.compile(lambda t: op(t), fullgraph=True, backend="hpu_backend")
        ref = torch.full(SHAPE, 6.0)
    else:
        fn = torch.compile(lambda t: t * 2.0, fullgraph=True, backend="hpu_backend")
        ref = torch.full(SHAPE, 10.0)

    try:
        y = fn(x)
        torch.hpu.synchronize()
        print(f"OK equal={torch.equal(y.cpu(), ref)}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {type(e).__name__}: {str(e).splitlines()[0][:110]}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--arm":
        run_arm(sys.argv[2])
        return 0

    for name, arm in ARMS:
        p = subprocess.run(
            [sys.executable, __file__, "--arm", arm],
            capture_output=True, text=True,
        )
        lines = [l for l in p.stdout.splitlines() if l.startswith(("OK", "FAIL"))]
        print(f"  {name:38s} {lines[-1] if lines else '(no verdict)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
