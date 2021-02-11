import torch

class ComplexExponentialFuntion(torch.autograd.Function):
    @staticmethod
    def forward(self,input):
        out = input.exp()
        self.save_for_backward(out)
        return out

    @staticmethod
    def backward(self,grad_output):
        out, = self.saved_tensors

        return grad_output * out.conj()

cexp = ComplexExponentialFuntion.apply

class ComplexExponential(torch.nn.Module):
    def __init__(self):
        super(ComplexExponential,self).__init__()

    def forward(self,input):
        return cexp(input)
