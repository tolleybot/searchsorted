# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

# defining the problem dimensions
nrows_a = 50000
nrows_v = 50000
nsorted_values = 300
nvalues = 1000
device = None


def test(device):
    a = torch.randn(nrows_a, nsorted_values, device=device)
    a = torch.sort(a, dim=1)[0]
    v = torch.randn(nrows_v, nvalues, device=a.device)
    side = torch.tensor([0], dtype=torch.int64)  #  'right'

    # Convert to numpy for validation
    a_np = a.cpu().detach().numpy()
    v_np = v.cpu().detach().numpy()

    out = torch.ops.mynamespace.searchsorted(a, v, side)

    side_str = "right" if side.item() == 0 else "left"

    # Expected results using NumPy's searchsorted
    expected_out_np = np.array(
        [np.searchsorted(a_np[i], v_np[i], side=side_str) for i in range(nrows_a)]
    )

    # Convert expected output to torch tensor for comparison
    expected_out = torch.from_numpy(expected_out_np)
    out_cpu = out.cpu().detach()

    # Validate the output (considering the output and expected output are of the same shape)
    if torch.equal(out_cpu, expected_out.int()):
        print("Test passed: Output matches expected results.")
    else:
        print("Test failed: Output does not match expected results.")

    # Optional: Print differences if needed
    differences = torch.abs(out_cpu.int() - expected_out.int())
    print("Differences:", differences)

    print(out)


def register_custom_op():
    def mysearchsorted(g, a, v, side_left):
        return g.op("mydomain::searchsorted", a, v, side_left)

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("mynamespace::searchsorted", mysearchsorted, 9)


def export_custom_op(device):
    class CustomModel(torch.nn.Module):
        def forward(self, a, v, side_left):
            return torch.ops.mynamespace.searchsorted(a, v, side_left)

    side_left = torch.tensor([0], dtype=torch.int64, device=device)  #  'right'
    # generate a matrix with sorted rows
    a = torch.randn(nrows_a, nsorted_values, device=device)
    a = torch.sort(a, dim=1)[0]
    # generate a matrix of values to searchsort
    v = torch.randn(nrows_v, nvalues, device=a.device)

    inputs = (a, v, side_left)

    model_file = "./model.onnx"
    torch.onnx.export(
        CustomModel(),
        inputs,
        model_file,
        opset_version=9,
        input_names=["a", "v", "side_left"],
        output_names=["Y"],
        custom_opsets={"mydomain": 1},
    )


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using CUDA.")
    torch.ops.load_library(
        "/src/searchsorted/torch/searchsorted_cuda.cpython-310-x86_64-linux-gnu.so"
    )
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")
    torch.ops.load_library("searchsorted_cpu.cpython-310-x86_64-linux-gnu.so")

# test(device)
register_custom_op()
export_custom_op(device)
