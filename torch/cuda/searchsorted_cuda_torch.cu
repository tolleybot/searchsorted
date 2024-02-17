#include "searchsorted_cuda_kernel.h"
#include "searchsorted_cuda_torch.h"
#include <algorithm> 

torch::Tensor searchsorted_cuda(
  torch::Tensor a,
  torch::Tensor v,
  bool side_left){

      // Get the dimensions
      auto nrow_a = a.size(/*dim=*/0);
      auto nrow_v = v.size(/*dim=*/0);
      auto ncol_a = a.size(/*dim=*/1);
      auto ncol_v = v.size(/*dim=*/1);

      auto nrow_res = std::max(nrow_a, nrow_v);

      // Allocate the result tensor. Assuming the result is a 2D tensor of int64_t.
      auto options = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
      torch::Tensor res = torch::empty({nrow_res, ncol_v}, options);

      // prepare the kernel configuration
      dim3 threads(ncol_v, nrow_res);
      dim3 blocks(1, 1);
      if (nrow_res*ncol_v > 1024){
         threads.x = int(fmin(double(1024), double(ncol_v)));
         threads.y = floor(1024/threads.x);
         blocks.x = ceil(double(ncol_v)/double(threads.x));
         blocks.y = ceil(double(nrow_res)/double(threads.y));
      }

      AT_DISPATCH_ALL_TYPES(a.scalar_type(), "searchsorted cuda", ([&] {
        searchsorted_kernel<scalar_t><<<blocks, threads>>>(
          res.data_ptr<int64_t>(),
          a.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(),
          nrow_res, nrow_a, nrow_v, ncol_a, ncol_v, side_left);
      }));

      // Ensure the kernel is done executing and data is synced back
      cudaDeviceSynchronize();

      return res;

  }

