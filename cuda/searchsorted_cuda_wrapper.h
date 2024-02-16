#ifndef _SEARCHSORTED_CUDA_WRAPPER
#define _SEARCHSORTED_CUDA_WRAPPER

#include "searchsorted_cuda_kernel.h"

void searchsorted_cuda_wrapper(
    torch::Tensor a,
    torch::Tensor v,
    torch::Tensor res,
    bool side_left);

#endif
