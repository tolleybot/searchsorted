import os
import onnxruntime as ort

# Construct the path to your custom operator library
# Assuming the current script is in the same directory as the "build" folder
library_path = os.path.join(os.getcwd(), "build", "libortsearchsortedop.so")

# Initialize session options
so = ort.SessionOptions()

# Register your custom operator library with ONNX Runtime
so.register_custom_ops_library(library_path)

# Load your ONNX model that uses the custom operator
# Replace 'path_to_your_model.onnx' with the actual path to your ONNX model file
sess = ort.InferenceSession("path_to_your_model.onnx", so)

# Prepare your input data according to the model's requirements
input_data = {"input_name": ...}  # Replace 'input_name' and ... with actual values

# Run inference
outputs = sess.run(None, input_data)

# Process the outputs as needed
print(outputs)
