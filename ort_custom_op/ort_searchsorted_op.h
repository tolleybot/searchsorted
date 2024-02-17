#include "SearchSortedOp.h"
#include "searchsorted_cuda_kernel.h" // Include your CUDA kernel header

void* SearchSortedOp::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
    return new Kernel(api, info);
}

const char* SearchSortedOp::GetName() const {    
    return "mydomain::searchsorted";
}

size_t SearchSortedOp::GetInputTypeCount() const {
    return 3; // a, v, side_left
}

ONNXTensorElementDataType SearchSortedOp::GetInputType(size_t index) const {
    // Simplification: assuming float for a and v, and int64 for side_left
    if (index < 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

size_t SearchSortedOp::GetOutputTypeCount() const {
    return 1; // Only one output
}

ONNXTensorElementDataType SearchSortedOp::GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64; // Assuming output is int64
}

// Kernel class implementation
class Kernel : public Ort::KernelBase {
public:
    Kernel(OrtApi api, const OrtKernelInfo* info) : Ort::KernelBase(api, info) {}

   void Compute(OrtKernelContext* context) override {
    // Obtain input tensors from the ORT context
    const OrtValue* input_a = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_v = ort_.KernelContext_GetInput(context, 1);
    const OrtValue* input_side_left = ort_.KernelContext_GetInput(context, 2);

    // Convert ORT tensors to raw pointers for CUDA kernel
    auto* a_data = reinterpret_cast<float*>(ort_.GetTensorMutableData<float>(input_a));
    auto* v_data = reinterpret_cast<float*>(ort_.GetTensorMutableData<float>(input_v));
    auto* side_left_data = reinterpret_cast<bool*>(ort_.GetTensorMutableData<bool>(input_side_left));

    // Assuming you have already obtained the dimensions similar to your PyTorch implementation
    // For simplicity, let's assume nrow_a, nrow_v, ncol_a, and ncol_v are already defined

    // Prepare the output tensor
    OrtTensorDimensions dimensions_a(ort_, input_a);
    OrtTensorDimensions dimensions_v(ort_, input_v);
    auto nrow_a = dimensions_a[0];
    auto ncol_a = dimensions_a[1];
    auto nrow_v = dimensions_v[0];
    auto ncol_v = dimensions_v[1];
    auto nrow_res = std::max(nrow_a, nrow_v);

    std::vector<int64_t> output_dims = {nrow_res, ncol_v};
    OrtValue* output_tensor = nullptr;
    ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size(), &output_tensor);
    auto* output_data = ort_.GetTensorMutableData<int64_t>(output_tensor);

    // Configure kernel dimensions as in PyTorch
    dim3 threads(ncol_v, nrow_res);
    dim3 blocks(1, 1);
    if (nrow_res * ncol_v > 1024) {
        threads.x = static_cast<int>(fmin(1024.0, static_cast<double>(ncol_v)));
        threads.y = static_cast<int>(floor(1024 / threads.x));
        blocks.x = static_cast<int>(ceil(static_cast<double>(ncol_v) / threads.x));
        blocks.y = static_cast<int>(ceil(static_cast<double>(nrow_res) / threads.y));
    }

    // Launch the CUDA kernel
    bool side_left = *side_left_data; // Simplification: assuming side_left is a single bool value
    searchsorted_kernel<<<blocks, threads>>>(output_data, a_data, v_data, nrow_res, nrow_a, nrow_v, ncol_a, ncol_v, side_left);

    // Synchronize after kernel execution
    cudaDeviceSynchronize();
}

};

// Here you would include the logic to obtain input tensors from OrtKernelContext,
// prepare them for the CUDA kernel, and launch your existing CUDA kernel.
// After computing, you'd write the results to the output tensor also obtained from OrtKernelContext.
