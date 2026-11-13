"""Discover 5 + 1 = 6, 64 times, using a custom HPU TPC kernel.

Usage:
    GC_KERNEL_PATH=/path/to/libadd1_kernels.so:/usr/lib/habanalabs/libtpc_kernels.so \
        PT_HPU_LAZY_MODE=0 \
        python add1_demo.py
"""

import torch
from custom_add1 import CustomAdd1


def main():
    # Build the input: 64 fives
    fives = torch.full((64,), 5.0, dtype=torch.float32)
    fives_hpu = fives.to("hpu")

    print("Input tensor (64 fives):")
    print(fives)
    print()

    # Apply our custom add1 TPC kernel on HPU
    custom_add1 = CustomAdd1()
    result_hpu = custom_add1(fives_hpu)
    torch.hpu.synchronize()
    result = result_hpu.cpu()

    print("After custom_add1 (5 + 1, 64 times):")
    print(result)
    print()

    # Confirm the answer is 6, 64 times
    expected = torch.full((64,), 6.0, dtype=torch.float32)
    print(f"All 64 elements equal 6.0?  {torch.equal(result, expected)}")


if __name__ == "__main__":
    main()
