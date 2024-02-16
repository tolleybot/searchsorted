# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import pytest
import tempfile
import os
import onnxruntime as ort

# defining the problem dimensions
nrows_a = 50000
nrows_v = 50000
nsorted_values = 300
nvalues = 1000


def register_custom_op():
    def mysearchsorted(g, a, v, side_left):
        return g.op("mydomain::searchsorted", a, v, side_left)

    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("mynamespace::searchsorted", mysearchsorted, 9)


def export_custom_op(model_file: str, device):
    class CustomModel(torch.nn.Module):
        def forward(self, a, v, side_left):
            return torch.ops.mynamespace.searchsorted(a, v, side_left)

    side_left = torch.tensor([0], dtype=torch.int64, device=device)  # 'right'
    a = torch.randn(nrows_a, nsorted_values, device=device)
    a = torch.sort(a, dim=1)[0]
    v = torch.randn(nrows_v, nvalues, device=a.device)
    inputs = (a, v, side_left)

    torch.onnx.export(
        CustomModel(),
        inputs,
        model_file,
        opset_version=9,
        input_names=["a", "v", "side_left"],
        output_names=["Y"],
        custom_opsets={"mydomain": 1},
    )


@pytest.mark.cpu
def test_searchsorted_cpu():
    device = torch.device("cpu")
    _test_searchsorted(device)


@pytest.mark.gpu
def test_searchsorted_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _test_searchsorted(device)
    else:
        pytest.skip("CUDA is not available, skipping GPU test.")


def _test_searchsorted(device):
    a = torch.randn(nrows_a, nsorted_values, device=device)
    a = torch.sort(a, dim=1)[0]
    v = torch.randn(nrows_v, nvalues, device=device)
    side = torch.tensor([0], dtype=torch.int64, device=device)  #  'right'

    out = torch.ops.mynamespace.searchsorted(a, v, side)

    # Convert to numpy for validation
    a_np = a.cpu().numpy()
    v_np = v.cpu().numpy()
    side_str = "right" if side.item() == 0 else "left"

    # Expected results using NumPy's searchsorted
    expected_out_np = np.array(
        [np.searchsorted(a_np[i], v_np[i], side=side_str) for i in range(nrows_a)]
    )
    expected_out = torch.from_numpy(expected_out_np).to(device)

    # Validate the output
    assert torch.equal(
        out.int(), expected_out.int()
    ), "Output does not match expected results."


def test_searchsorted_onnxruntime_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping ONNX Runtime GPU test.")

    device = torch.device("cuda")
    # Setup temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_file = os.path.join(tmpdirname, "model.onnx")

        # Export the custom op model to the temporary directory
        export_custom_op(model_file, device)

        # Setup ONNX Runtime session
        if "CUDAExecutionProvider" in ort.get_available_providers():
            sess_options = ort.SessionOptions()
            sess = ort.InferenceSession(model_file, sess_options)
            sess.set_providers(["CUDAExecutionProvider"])  # Use CUDA

            # Prepare inputs
            a = torch.randn(nrows_a, nsorted_values).sort(dim=1)[0].numpy()
            v = torch.randn(nrows_v, nvalues).numpy()
            side_left = np.array([0], dtype=np.int64)

            # Prepare the input dictionary
            input_dict = {"a": a, "v": v, "side_left": side_left}
            # Run the model
            out_onnx = sess.run(None, input_dict)

            # Validation step (assuming the expected validation code is similar to the PyTorch test's validation)
            expected_out_np = np.array(
                [np.searchsorted(a[i], v[i], side="right") for i in range(nrows_a)]
            )

            # Validate the output
            np.testing.assert_array_equal(
                out_onnx[0],
                expected_out_np,
                err_msg="ONNX Runtime GPU output does not match expected results.",
            )
