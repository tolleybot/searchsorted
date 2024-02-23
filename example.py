import os
import onnxruntime as ort
import numpy as np


def numpy_searchsorted(a: np.ndarray, v: np.ndarray, side="left"):
    """Numpy version of searchsorted that works batch-wise on pytorch tensors"""
    nrows_a = a.shape[0]
    (nrows_v, ncols_v) = v.shape
    nrows_out = max(nrows_a, nrows_v)
    out = np.empty((nrows_out, ncols_v), dtype=np.int64)

    def sel(data, row):
        return data[0] if data.shape[0] == 1 else data[row]

    for row in range(nrows_out):
        out[row] = np.searchsorted(sel(a, row), sel(v, row), side=side)
    return out


# Construct the path to your custom operator library
# Assuming the current script is in the same directory as the "build" folder
library_path = os.path.join(os.getcwd(), "build", "libortsearchsortedop.so")

# Initialize session options
so = ort.SessionOptions()
so.log_severity_level = 0  # Enable verbose logging

# Register your custom operator library with ONNX Runtime
so.register_custom_ops_library(library_path)

if not "CUDAExecutionProvider" in ort.get_available_providers():
    print("CUDAExecutionProvider not available, using CPU.")

# Load your ONNX model that uses the custom operator
# Replace 'path_to_your_model.onnx' with the actual path to your ONNX model file
model_path = "searchsorted_custom_op.onnx"
sess = ort.InferenceSession(model_path, so, providers=["CUDAExecutionProvider"])

# make some dummy data
a_dummy = np.random.randn(10, 10).astype(
    np.float32
)  # Adjust dtype as per your model's requirement
a_dummy = np.sort(a_dummy, axis=1)

v_dummy = np.random.randn(10, 1).astype(np.float32)
side_left_dummy = np.array([True], dtype=np.bool_)  # Example value

print("Input a sorted: ", a_dummy)
print("Input v: ", v_dummy)

# Matching input names from your model
input_data = {"a": a_dummy, "v": v_dummy, "side_left": side_left_dummy}

# Run inference
outputs = sess.run(None, input_data)

indices = numpy_searchsorted(a_dummy, v_dummy, side="left")

# Process the outputs as needed
print("Ouput indices: ", outputs[0])
print("Expected indices: ", indices)

# Compare outputs with indices
if np.array_equal(outputs[0], indices):
    print("Outputs and indices match!")
else:
    print("Outputs and indices do not match.")

    # Find the mismatched indices
    mismatched_indices = np.where(outputs[0] != indices)[0]

    # Print the mismatched indices
    print("Mismatched indices: ", mismatched_indices)
