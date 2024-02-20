import os
import onnxruntime as ort
import numpy as np

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
a_dummy = np.random.randn(50000, 300).astype(
    np.float32
)  # Adjust dtype as per your model's requirement
v_dummy = np.random.randn(50000, 1000).astype(np.float32)
side_left_dummy = np.array([True], dtype=np.bool_)  # Example value

# Matching input names from your model
input_data = {"a": a_dummy, "v": v_dummy, "side_left": side_left_dummy}

# Run inference
outputs = sess.run(None, input_data)

# Process the outputs as needed
print(outputs)
