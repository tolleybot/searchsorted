#include <torch/script.h>

torch::Tensor searchsorted_cuda(
    torch::Tensor a,
    torch::Tensor v,
    bool side_left);


