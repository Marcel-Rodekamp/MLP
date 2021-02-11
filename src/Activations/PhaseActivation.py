import torch
from .Exponential import cexp
from ..complexifyTorch import cangle

class PhaseActivation(torch.nn.Module):
    def __init__(self,activation):
        super(PhaseActivation,self).__init__()

        self.activation = activation

    def forward(self,input):
        abs = self.activation(torch.sqrt( input.real**2 + input.imag**2 ))
        ang = self.activation(cangle(input))

        return abs * cexp(1j * ang)
