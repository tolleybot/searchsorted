# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import onnxruntime as ort
import time
import tempfile
import os

# Benchmark settings
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


def prepare_data(device):
    """Prepare test data."""
    a = torch.randn(nrows_a, nsorted_values, device=device).sort(dim=1)[0]
    v = torch.randn(nrows_v, nvalues, device=device)
    side_left = torch.tensor([1], dtype=torch.int64, device=device)  # 'left side'
    return a, v, side_left


def benchmark_customop_pytorch(device):
    """Benchmark the PyTorch implementation."""
    a, v, side_left = prepare_data(device)

    start_time = time.time()
    out = torch.ops.mynamespace.searchsorted(a, v, side_left)
    torch.cuda.synchronize()  # Ensure completion on GPU
    elapsed_time = time.time() - start_time

    print(f"PyTorch ({device}): {elapsed_time:.3f} seconds")


def benchmark_onnxruntime(device, model_file):
    """Benchmark the ONNX Runtime implementation."""
    a, v, side_left = prepare_data("cpu")
    a_np = a.numpy()
    v_np = v.numpy()
    side_left_np = side_left.numpy()

    sess_options = ort.SessionOptions()
    if device.type == "cuda":
        sess = ort.InferenceSession(
            model_file, sess_options, providers=["CPUExecutionProvider"]
        )
    else:
        sess = ort.InferenceSession(
            model_file, sess_options, providers=["CUDAExecutionProvider"]
        )

    start_time = time.time()
    out_onnx = sess.run(None, {"a": a_np, "v": v_np, "side_left": side_left_np})
    elapsed_time = time.time() - start_time

    print(f"ONNX Runtime ({device}): {elapsed_time:.3f} seconds")


def run_benchmarks():

    if torch.cuda.is_available():
        print("GPU is available. Using CUDA.")
        device = torch.device("cuda")
        torch.ops.load_library("searchsorted_cuda.cpython-310-x86_64-linux-gnu.so")
    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")
        torch.ops.load_library("searchsorted_cpu.cpython-310-x86_64-linux-gnu.so")

    register_custom_op()

    if torch.cuda.is_available():
        benchmark_customop_pytorch("cuda")

    print(f"\nBenchmarking on {device}...")

    # Export model for ONNX Runtime
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_file = os.path.join(tmpdirname, "model.onnx")
        export_custom_op(model_file, device)

        # ONNX Runtime benchmark
        benchmark_onnxruntime(device, model_file)


if __name__ == "__main__":
    run_benchmarks()
