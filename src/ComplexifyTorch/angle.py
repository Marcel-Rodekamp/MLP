import torch

class AngleFunction(torch.autograd.Function):
    r"""

    """
    @staticmethod
    def forward(self,input):
        # Todo we negect tha cases Re(x) < 0, Im(x) = 0 => Pi
        #                      and Re(x) = Im(x)        => 0
        self.save_for_backward(input)
        abs = torch.sqrt(input.real**2 + input.imag**2)
        return 2 * torch.atan(input.imag/(abs+input.real))

    @staticmethod
    def backward(self,grad_output):
        input, = self.saved_tensors

        return 0.5*(grad_output.conj() * (1j/input.conj()) - grad_output * (1j/input).conj())

cangle = AngleFunction.apply
