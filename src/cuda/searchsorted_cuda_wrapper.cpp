#include "searchsorted_cuda_wrapper.h"

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void searchsorted_cuda_wrapper(torch::Tensor a, torch::Tensor v, torch::Tensor res, bool side_left)
{
  CHECK_INPUT(a);
  CHECK_INPUT(v);
  CHECK_INPUT(res);

  searchsorted_cuda(a, v, res, side_left);
}

  static auto registry =
  torch::RegisterOperators("mynamespace::searchsorted", &searchsorted_cuda_wrapper);

