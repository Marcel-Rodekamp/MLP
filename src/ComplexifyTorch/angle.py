import torch

class AngleFunction(torch.autograd.Function):
    r"""

    """
    @staticmethod
    def forward(self,input):
        # Todo we negect tha cases Re(x) < 0, Im(x) = 0 => Pi
        #                      and Re(x) = Im(x)        => 0
        input_mask = input.imag < 0

        self.save_for_backward(input,input_mask)

        abs = torch.sqrt(input.real**2 + input.imag**2)

        output = torch.acos(input.real / abs)

        output[input_mask] *= -1

        return output

    @staticmethod
    def backward(self,grad_output):
        input,input_mask = self.saved_tensors

        abs_sq = input.real**2 + input.imag**2

        output = grad_output * (input.real*input.imag/(abs_sq)).conj() \
               + grad_output.conj() * (input.real - input.imag)/abs_sq

        output[input_mask] *= -1

        return output

cangle = AngleFunction.apply
