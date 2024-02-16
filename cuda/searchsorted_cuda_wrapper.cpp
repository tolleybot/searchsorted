#include "searchsorted_cuda_wrapper.h"

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor searchsorted_cuda_wrapper(torch::Tensor a, torch::Tensor v, bool side_left)
{
  CHECK_INPUT(a);
  CHECK_INPUT(v);

  return searchsorted_cuda(a, v, side_left);
}

  static auto registry =
  torch::RegisterOperators("mynamespace::searchsorted", &searchsorted_cuda_wrapper);

