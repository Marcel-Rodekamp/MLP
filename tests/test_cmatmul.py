import torch
from MLP.complexifyTorch import cmatmul

def test_forward():
    x = (2+1j)*torch.eye(2,2,dtype = torch.cdouble)
    y = (2+2j)*torch.eye(2,2,dtype = torch.cdouble)

    print(x)
    print(cmatmul(x,y))

test_forward()
