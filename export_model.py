from onnx import helper, TensorProto, checker, ModelProto

# Define the inputs and output tensor info
a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None, None])
v = helper.make_tensor_value_info("v", TensorProto.FLOAT, [None, None])
side_left = helper.make_tensor_value_info(
    "side_left", TensorProto.BOOL, [1]
)  # Assuming side_left is a single boolean value

output_tensor = helper.make_tensor_value_info(
    "output", TensorProto.INT64, [None, None]
)  # Adjust the shape as necessary

# Create the custom node
# Please adjust the domain and op_type to match your actual custom op details
searchsorted_node = helper.make_node(
    "SearchSortedOp",  # op_type
    ["a", "v", "side_left"],  # inputs
    ["output"],  # outputs
    domain="mydomain",  # Adjust the domain as necessary,
    name="SearchSortedOp",
)

# Create the graph
graph = helper.make_graph(
    nodes=[searchsorted_node],
    name="SearchSortedGraph",
    inputs=[a, v, side_left],
    outputs=[output_tensor],
)

# Create the model (don't forget to set the opset_imports for your custom domain)
model = helper.make_model(
    graph,
    producer_name="searchsorted_example",
    opset_imports=[
        helper.make_opsetid("", 9),
        helper.make_opsetid("mydomain", 1),  # Adjust the version as necessary
    ],
)

# Check the model
checker.check_model(model)

# Save the model
with open("searchsorted_custom_op.onnx", "wb") as f:
    f.write(model.SerializeToString())

print(
    "Model with custom 'searchsorted' operator has been saved to 'searchsorted_custom_op.onnx'"
)
