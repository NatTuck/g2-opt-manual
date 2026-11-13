"""The bit-exact gate for custom_add1, with no test framework required.

`test_add1_op.py` is upstream's pytest test and is kept as upstream wrote it.
This is the same check as a standalone program, because the gate is what has to
run everywhere — and because a gate has to be shown to fail (see 09-tpc-kernels.md
§9.8: a gate that cannot fail on the wrong kernel is decoration).

    GC_KERNEL_PATH=<...>/libadd1_kernels.so PT_HPU_LAZY_MODE=0 \
        python3 gate_add1.py

Exit code 0 = every arm passed, 1 = at least one failed.

To see it fail, build a wrong kernel: change `const VECTOR one = v_mov_s((SCALAR)1.0)`
to `(SCALAR)2.0` in tpc_kernel_lib/kernels/gaudi2/add1_fwd_f32_gaudi2.c, rebuild,
and run this. It must exit 1.
"""
import sys

import torch
import habana_frameworks.torch.core  # noqa: F401

from custom_add1 import CustomAdd1

SHAPES = [(64,), (100,), (256,), (1024, 1024)]


def main():
    op = CustomAdd1()
    failures = 0

    for shape in SHAPES:
        # A value that is not 0 or 1, so a kernel that adds the wrong constant
        # (or adds nothing) cannot pass by coincidence.
        x = torch.full(shape, 5.0, dtype=torch.float32)
        expected = x + 1.0
        got = op(x.to("hpu")).cpu()
        ok = torch.equal(got, expected)
        failures += 0 if ok else 1
        print(f"  bit-exact  shape={str(shape):12s} equal={ok}")

    # The op declares Float only; anything else must be refused, not silently
    # reinterpreted.
    try:
        op(torch.zeros(64, dtype=torch.float16).to("hpu"))
        torch.hpu.synchronize()
        print("  dtype gate float16                    refused=False")
        failures += 1
    except RuntimeError as e:
        first = str(e).splitlines()[0]
        print(f"  dtype gate float16                    refused=True  ({first[:60]})")

    print(f"  {'PASS' if failures == 0 else 'FAIL'} ({failures} failing arm(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
