import torch

class ComplexLogarithmFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input):
        self.save_for_backward(input)
        return (1+input).log()

    @staticmethod
    def backward(self,grad_output):
        input, = self.saved_tensors

        return grad_output * (1/(1+input)).conj()

clog = ComplexLogarithmFunction.apply

class ComplexLogarithm(torch.nn.Module):
    def __init__(self):
        super(ComplexLogarithm,self).__init__()

    def forward(self,input):
        return clog(input)
