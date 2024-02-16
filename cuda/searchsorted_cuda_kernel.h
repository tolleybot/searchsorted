#ifndef _SEARCHSORTED_CUDA_KERNEL
#define _SEARCHSORTED_CUDA_KERNEL

//#include <torch/extension.h>
#include <torch/script.h>

void searchsorted_cuda(
    torch::Tensor a,
    torch::Tensor v,
    torch::Tensor res,
    bool side_left);

#endif