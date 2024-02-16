# SPDX-License-Identifier: Apache-2.0

import torch

# defining the problem dimensions
nrows_a = 50000
nrows_v = 50000
nsorted_values = 300
nvalues = 1000

def test():
    a = torch.randn(nrows_a, nsorted_values, device='cpu')
    a = torch.sort(a, dim=1)[0]
    v = torch.randn(nrows_v, nvalues, device=a.device)
    side = torch.tensor([0], dtype=torch.int64) #  'right'
    out = torch.empty((max(a.shape[0], v.shape[0]), v.shape[1]), device=v.device, dtype=torch.int64)
    torch.ops.mynamespace.searchsorted(a, v, out, side)
    print(out)
     
def register_custom_op():
    def mysearchsorted(g,  a, v, out, left_side):
        return g.op("mydomain::testsearchsorted", a, v, out, left_side)

    from torch.onnx import register_custom_op_symbolic
                                
    register_custom_op_symbolic("mynamespace::searchsorted", mysearchsorted, 9)

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, a, v, out, left_side):
            return torch.ops.mynamespace.searchsorted(a, v, out, left_side)                                    

    side = torch.tensor([0], dtype=torch.int64) #  'right'
    # generate a matrix with sorted rows
    a = torch.randn(nrows_a, nsorted_values, device='cpu')
    a = torch.sort(a, dim=1)[0]
    # generate a matrix of values to searchsort
    v = torch.randn(nrows_v, nvalues, device=a.device)
    result_shape = (max(a.shape[0], v.shape[0]), v.shape[1])
    output = torch.empty(result_shape, device=v.device, dtype=torch.int64)

    inputs = (a, v, output , side)   


    model_file = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, model_file,
                    opset_version=9,                     
                    input_names=["a", "v", "out", "left_side"],
                    output_names=["Y"],
                    custom_opsets={"mydomain": 1})



# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU is available. Using CUDA.")
#     torch.ops.load_library(
#         "build/lib.linux-x86_64-3.10/searchsorted/cuda.cpython-310-x86_64-linux-gnu.so")
#else:
device = torch.device("cpu")
print("GPU is not available. Using CPU.")
# torch.ops.load_library(
#     "/src/searchsorted/src/build/lib.linux-x86_64-3.10/custom_group_norm.cpython-310-x86_64-linux-gnu.so")     

torch.ops.load_library(
    "searchsorted_cpu.cpython-310-x86_64-linux-gnu.so")     

register_custom_op()
export_custom_op()