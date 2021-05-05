import torch

class ComplexSofplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input,beta,threshold):
        self.beta = beta
        self.threshold = threshold
        argument = beta*input
        self.save_for_backward(argument)

        output = torch.zeros_like(argument)

        output[torch.abs(argument)>threshold] = argument[torch.abs(argument)>threshold]
        output[torch.abs(argument)<=threshold] = 1/beta * (1+(argument[torch.abs(argument)<=threshold]).exp()).log()

        return output

    @staticmethod
    def backward(self,grad_output):
        argument, = self.saved_tensors

        grad = torch.zeros_like(argument)
        grad[torch.abs(argument)>self.threshold] = self.beta
        grad[torch.abs(argument)<=self.threshold] = ((argument[torch.abs(argument)<=self.threshold].exp())/(1+argument[torch.abs(argument)<=self.threshold].exp()))

        return grad_output * grad.conj(),None,None

csoftplus = ComplexSofplusFunction.apply

class ComplexSoftplus(torch.nn.Module):
    def __init__(self,beta=1,threshold=20):
        super(ComplexSoftplus,self).__init__()

        self.beta = beta
        self.threshold = threshold

    def forward(self,input):
        return csoftplus(input,self.beta,self.threshold)
