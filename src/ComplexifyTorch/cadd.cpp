#include <torch/torch.h>

torch::Tensor cadd_cpu(torch::Tensor & input, torch::Tensor & other){
    // return real matrix multiplication if tensor is not complex
    if( ! input.is_complex() || ! other.is_complex() ){
        return at::add(input,other);
    }
    // otherwise perform complex multiplication
    return at::complex(
        // real part
        at::add(at::real(input),at::real(other)),
        // imag part
        at::add(at::imag(input),at::imag(other))
    );
}

TORCH_LIBRARY_IMPL(complexifyTorch_cpp, CPU, m) {
    m.impl("cadd", cadd_cpu);
}
