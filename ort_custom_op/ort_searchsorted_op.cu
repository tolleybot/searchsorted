
#include "ort_searchsorted_op.h"
#include "searchsorted_cuda_kernel.h"
#include <iostream>

const int MAX_THREADS_PER_BLOCK = 1024;


__global__
void test_kernel(int64_t *res, int64_t nrow_res, int64_t ncol_v) {
    // Calculate global row and column indices for the current thread
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check whether we are within the bounds to be computed
    if (row < nrow_res && col < ncol_v) {
        // Write a dummy value to the result tensor
        // For example, just combining row and column indices
        res[row * ncol_v + col] = row * 100 + col; // Example dummy operation
    }
}

__global__ void supersimple_kernel(/*int64_t *res*/) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //res[0] = 12345;
    }
}



// Implementation of the Compute function for SearchSortedKernel
template <typename T>
void SearchSortedKernel<T>::Compute(OrtKernelContext* context) {

    Ort::KernelContext ctx(context);
    auto v_a = ctx.GetInput(0);
    auto v_v = ctx.GetInput(1);
    auto v_side_left = ctx.GetInput(2); 
   
    const T* a_data = v_a.GetTensorData<T>();
    const T* v_data = v_v.GetTensorData<T>();
    const bool* side_left_data = v_side_left.GetTensorData<bool>();
    
    auto dimensions_a = v_a.GetTensorTypeAndShapeInfo().GetShape();
    auto dimensions_v = v_v.GetTensorTypeAndShapeInfo().GetShape();
    
    // Prepare the output tensor    
    auto nrow_a = dimensions_a[0];
    auto ncol_a = dimensions_a[1];
    auto nrow_v = dimensions_v[0];
    auto ncol_v = dimensions_v[1];
    auto nrow_res = std::max(nrow_a, nrow_v);

    std::vector<int64_t> output_dims = {nrow_res, ncol_v};
    auto output_tensor = ctx.GetOutput(0, output_dims);

    int64_t* output_data = output_tensor.GetTensorMutableData<int64_t>();

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
    bool side_left_bool;
    cudaMemcpy(&side_left_bool, side_left_data, sizeof(bool), cudaMemcpyDeviceToHost);

   // bool side_left_bool = *side_left_data; // Simplification: assuming side_left is a single bool value
    searchsorted_kernel_float(output_data, a_data, v_data, nrow_res, nrow_a, nrow_v, ncol_a, ncol_v, side_left_bool, blocks, threads);
        // Optional: Check for errors after kernel launch
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // Handle error
        std::cerr << "CUDA error in kernel launch: " << cudaGetErrorString(error) << std::endl;
    }

    // Optional: Synchronize device to wait for kernel completion and catch errors
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        // Handle error
        std::cerr << "CUDA error on synchronize: " << cudaGetErrorString(error) << std::endl;
    }

   

}

template class SearchSortedKernel<float>;
