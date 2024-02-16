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
    out = torch.ops.mynamespace.searchsorted(a, v, side)
    print(out)
     
def register_custom_op():
    def mysearchsorted(g,  a, v, side_left):
        return g.op("mydomain::testsearchsorted", a, v, side_left)

    from torch.onnx import register_custom_op_symbolic
                                
    register_custom_op_symbolic("mynamespace::searchsorted", mysearchsorted, 9)

def export_custom_op():
    class CustomModel(torch.nn.Module):
        def forward(self, a, v, side_left):
            return torch.ops.mynamespace.searchsorted(a, v, side_left)                                    

    side_left = torch.tensor([0], dtype=torch.int64) #  'right'
    # generate a matrix with sorted rows
    a = torch.randn(nrows_a, nsorted_values, device='cpu')
    a = torch.sort(a, dim=1)[0]
    # generate a matrix of values to searchsort
    v = torch.randn(nrows_v, nvalues, device=a.device)

    inputs = (a, v, side_left)   


    model_file = './model.onnx'
    torch.onnx.export(CustomModel(), inputs, model_file,
                    opset_version=9,                     
                    input_names=["a", "v", "side_left"],
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