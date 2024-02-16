#ifndef _SEARCHSORTED_CUDA_KERNEL
#define _SEARCHSORTED_CUDA_KERNEL

//#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor searchsorted_cuda(
    torch::Tensor a,
    torch::Tensor v,
    bool side_left);

#endif
