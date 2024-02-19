#include <cuda_runtime.h>

/**
 * CUDA kernel for performing a searchsorted operation.
 *
 * This kernel assumes that 'a' is sorted along each row. For each element in 'v',
 * it finds the insertion point in 'a' to maintain sorted order, considering 'side_left'
 * to choose between two valid insertion points.
 *
 * @param res The output tensor where the result indices are stored.
 * @param a The input sorted tensor to search in.
 * @param v The values tensor to search for.
 * @param nrow_res The number of rows in the result tensor.
 * @param nrow_a The number of rows in the 'a' tensor.
 * @param nrow_v The number of rows in the 'v' tensor.
 * @param ncol_a The number of columns in the 'a' tensor.
 * @param ncol_v The number of columns in the 'v' tensor.
 * @param side_left If true, will return the first suitable location found. If false, returns the last.
 */
template <typename scalar_t>
__global__ void searchsorted_kernel(
    int64_t *res, scalar_t *a, scalar_t *v,
    int64_t nrow_res, int64_t nrow_a, int64_t nrow_v, 
    int64_t ncol_a, int64_t ncol_v, bool side_left);

/**
 * Wrapper function to call the searchsorted CUDA kernel with float data.
 *
 * This function configures and launches the CUDA kernel 'searchsorted_kernel' with
 * float type parameters and synchronizes after execution.
 *
 * @param res The output tensor where the result indices are stored.
 * @param a The input sorted tensor to search in, with float data.
 * @param v The values tensor to search for, with float data.
 * @param nrow_res The number of rows in the result tensor.
 * @param nrow_a The number of rows in the 'a' tensor.
 * @param nrow_v The number of rows in the 'v' tensor.
 * @param ncol_a The number of columns in the 'a' tensor.
 * @param ncol_v The number of columns in the 'v' tensor.
 * @param side_left If true, will return the first suitable location found. If false, returns the last.
 * @param blocks The grid dimension for kernel launch.
 * @param threads The block dimension for kernel launch.
 */
void searchsorted_kernel_float(
  int64_t *res,
  float *a,
  float *v,
  int64_t nrow_res, int64_t nrow_a, int64_t nrow_v, int64_t ncol_a, int64_t ncol_v, bool side_left,
  dim3 blocks, dim3 threads);
