// searchsorted_cuda.h
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void searchsorted_kernel(
    int64_t *res, scalar_t *a, scalar_t *v,
    int64_t nrow_res, int64_t nrow_a, int64_t nrow_v, 
    int64_t ncol_a, int64_t ncol_v, bool side_left);


