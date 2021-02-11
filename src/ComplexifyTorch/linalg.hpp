#ifndef LINALG_HPP
#define LINALG_HPP
#include<torch/torch.h>

// Defining operations following
// https://pytorch.org/tutorials/advanced/dispatcher.html#defining-schema-and-backend-implementations

// src/ComplexifyTorch/cmatmul.cpp
torch::Tensor cmatmul_cpu(torch::Tensor & input, torch::Tensor & other);
// src/ComplexifyTorch/cadd.cpp
torch::Tensor cadd_cpu(torch::Tensor & input, torch::Tensor & other);

TORCH_LIBRARY(complexifyTorch_cpp, m){
    m.def("cmatmul(Tensor input, Tensor other) -> Tensor");
    m.def("cmm(Tensor input, Tensor other) -> Tensor");
    m.def("cadd(Tensor input, Tensor other) -> Tensor");
}

#endif //LINALG_HPP
