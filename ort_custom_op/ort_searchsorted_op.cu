
#include "ort_searchsorted_op.h"
#include "searchsorted_cuda_kernel.h"


// Implementation of the Compute function for SearchSortedKernel
template <typename T>
void SearchSortedKernel<T>::Compute(OrtKernelContext* context) {

    // Obtain input tensors from the ORT context
    const OrtValue* input_a = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_v = ort_.KernelContext_GetInput(context, 1);
    const OrtValue* input_side_left = ort_.KernelContext_GetInput(context, 2);

    // Convert ORT tensors to raw pointers for CUDA kernel
    T* a_data = reinterpret_cast<T*>(ort_.GetTensorMutableData<T>(input_a));
    T* v_data = reinterpret_cast<T*>(ort_.GetTensorMutableData<T>(input_v));
    T* side_left_data = reinterpret_cast<T*>(ort_.GetTensorMutableData<T>(input_side_left));

    // Prepare the output tensor
    OrtTensorDimensions dimensions_a(ort_, input_a);
    OrtTensorDimensions dimensions_v(ort_, input_v);
    auto nrow_a = dimensions_a[0];
    auto ncol_a = dimensions_a[1];
    auto nrow_v = dimensions_v[0];
    auto ncol_v = dimensions_v[1];
    auto nrow_res = std::max(nrow_a, nrow_v);

    std::vector<int64_t> output_dims = {nrow_res, ncol_v};
    OrtValue* output_tensor = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());

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
