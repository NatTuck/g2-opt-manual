"""Python wrapper for our custom add1 TPC kernel (no autograd)."""

import torch
import os
import habana_frameworks.torch.core
from pathlib import Path

my_dir = os.path.dirname(os.path.realpath(__file__))
custom_op_lib_path = str(
    next(
        Path(
            next(Path(os.path.join(my_dir, "build")).glob("lib.linux-x86_64-*"))
        ).glob("hpu_custom_add1.cpython-*-x86_64-linux-gnu.so")
    )
)
torch.ops.load_library(custom_op_lib_path)


class CustomAdd1(torch.nn.Module):
    def __init__(self):
        super(CustomAdd1, self).__init__()

    def forward(self, input):
        return torch.ops.custom_op.custom_add1(input)

    def extra_repr(self):
        return 'CustomAdd1 for float32 only'
