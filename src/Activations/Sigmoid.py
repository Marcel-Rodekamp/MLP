import torch

class ComplexSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input):
        exponent = torch.exp(-input)
        self.save_for_backward(exponent)

        return 1/(1-exponent)

    @staticmethod
    def backward(self,grad_output):
        exponent, = self.saved_tensors

        return - exponent/(1-exponent)**2


csigmoid = ComplexSigmoidFunction.apply

class ComplexSigmoid(torch.nn.Module):
    def __init__(self):
        super(ComplexSigmoid,self).__init__()

    def forward(self,input):
        return csigmoid(input)
