#include "linalg.hpp"

torch::Tensor cmatmul_cpu(torch::Tensor & input, torch::Tensor & other){
    // return real matrix multiplication if tensor is not complex
    if(! input.is_complex() || ! other.is_complex()){
        return at::matmul(input,other);
    }
    // otherwise perform complex multiplication
    return at::complex(
        // real part
        at::matmul(at::real(input),at::real(other))-at::matmul(at::imag(input),at::imag(other)),
        // imag part
        at::matmul(at::real(input),at::imag(other))+at::matmul(at::imag(input),at::real(other))
    );
}

TORCH_LIBRARY_IMPL(complexifyTorch_cpp, CPU, m) {
    m.impl("cmatmul", cmatmul_cpu);
}
