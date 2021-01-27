import torch
from .impl import implements

@implements(torch.matmul)
def cmatmul(input,other,*args,**kwargs):
    try:
        # complex tensor multiplication
                            # real part
        return torch.complex(torch.matmul(input.real, other.real, *args, **kwargs)
                            -torch.matmul(input.imag, other.imag, *args, **kwargs),
                            # imag part
                             torch.matmul(input.real, other.imag, *args, **kwargs)
                            +torch.matmul(input.imag, other.real, *args, **kwargs))
    except RuntimeError: # torch throws a RuntimeError if tensor is real and one
                         # trys to access the real/imag part.
        return torch.matmul(input,other,*args,**kwargs)

@implements(torch.add)
def cadd(input,other,*args,**kwargs):
    try:
        # complex tensor add
        return torch.complex( torch.add(input.real,other.real,*args,**kwargs),
                              torch.add(input.imag,other.imag,*args,**kwargs))
    except RuntimeError: # torch throws a RuntimeError if tensor is real and one
                         # trys to access the real/imag part.
        return torch.add(input,other,*args,**kwargs)

@implements(torch.addmm)
def caddmm(input,mat1,mat2,*args,beta=1,alpha=1,**kwargs):
    return add(beta*input, alpha*matmul(mat1,mat2))
