#include "searchsorted_cuda_torch.h"

torch::Tensor searchsorted_cuda_wrapper(
    torch::Tensor a,
    torch::Tensor v,
    bool side_left);

