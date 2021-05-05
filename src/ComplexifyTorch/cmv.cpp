#include <torch/torch.h>

torch::Tensor cmv_cpu(torch::Tensor & input, torch::Tensor & other){
    // return real matrix multiplication if tensor is not complex
    if(! input.is_complex() || ! other.is_complex()){
        return at::mv(input,other);
    }
    // otherwise perform complex multiplication
    return at::complex(
        // real part
        at::mv(at::real(input),at::real(other))-at::mv(at::imag(input),at::imag(other)),
        // imag part
        at::mv(at::real(input),at::imag(other))+at::mv(at::imag(input),at::real(other))
    );
}


TORCH_LIBRARY_IMPL(complexifyTorch_cpp, CPU, m) {
    m.impl("cmv", cmv_cpu);
}
