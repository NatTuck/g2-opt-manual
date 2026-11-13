"""Does the custom op compose under torch.compile?  Each arm in its own process.

The example ships a Meta implementation of the op, and its own comment says the
Meta kernel is "needed only for torch.compile support".  This checks what that
buys on this stack:

  * the op traces under torch.compile(fullgraph=True)  -> yes (the Meta kernel
    does its job: dynamo can trace the op);
  * the compiled graph then compiles at Synapse        -> NO, on this stack;
  * a control graph with no custom op in it            -> yes, so the failure is
    about the custom op's recipe, and not about torch.compile here.

Each arm runs in a fresh interpreter on purpose.  Measured: once the custom op's
recipe has failed to compile, a *later* compiled graph in the same process fails
too, so a control sharing the process reports the wrong thing.

Usage:  GC_KERNEL_PATH=<...>/libadd1_kernels.so PT_HPU_LAZY_MODE=0 \
            python3 check_torch_compile.py
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
