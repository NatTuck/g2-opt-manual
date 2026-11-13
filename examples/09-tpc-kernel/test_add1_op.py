"""Minimal test for our custom add1 op (no autograd)."""

import torch
import os
import pytest
from custom_add1 import CustomAdd1


def test_custom_add1_op_function():
    print(torch.ops.custom_op.custom_add1)

    input = torch.full((64,), 5.0, dtype=torch.float32)
    input_hpu = input.to("hpu").detach()

    expected = (input + 1.0).clone()
    custom_add1 = CustomAdd1()
    output_hpu = custom_add1(input_hpu)
    torch.hpu.synchronize()

    assert torch.equal(output_hpu.cpu(), expected), f"Add1 mismatch: got {output_hpu.cpu()}, expected {expected}"
    print("Add1 forward: PASSED")


if __name__ == "__main__":
    test_custom_add1_op_function()
