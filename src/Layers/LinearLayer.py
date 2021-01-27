import torch

from ..complexifyTorch import cmatmul,cadd

class CLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input,weight,bias=None):
        self.save_for_backward(input,weight,bias)

        output = cmatmul(input,weight.t())

        if bias is not None:
            output = cadd(output,bias)

        return output

    @staticmethod
    def backward(self,grad_output):
        input,weight,bias = self.saved_tensors

        input_out = None
        weight_out = None
        bias_out = None

        # df(x,w,b)/dx = w^T
        if self.needs_input_grad[0]:
            input_out = cmatmul(grad_output,weight.t())
            # print(f"input_out.size() = {input_out.size()}")

        # df(x,w,b)/dw = x
        if self.needs_input_grad[1]:
            weight_out = cmatmul(grad_output.t(),input)
            # print(f"weight_out.size() = {weight_out.size()}")

        # df(x,w,b)/db = 1
        if self.needs_input_grad[2] and bias is not None:
            bias_out = grad_output.sum(0)
            # print(f"bias_out.size() = {bias_out.size()}")

        return input_out,weight_out,bias_out

# alias the class application
clinear =  CLinearFunction.apply

class CLinearLayer(torch.nn.Module):
    def __init__(self,in_features,out_features,dtype,bias = True):
        r"""
            in_features: int
                Input dimension
            out_features: int
                Output dimension
            dtype: torch.dtype
                Type of parameters
            bias: bool, default: True
                Usage of a additive bias term.

            This is a replica of the PyTorch linear layer allowing for a general
            dtype of the weight and bias parameters.

        """
        super(CLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.zeros(out_features,in_features,dtype=dtype))
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_features,dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # this is not as in PyTorchs implementation. But it suits more for our usecase.
        torch.nn.init.uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias)

    def forward(self,input):
        return clinear(input,self.weight,self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, dtype={self.dtype}"
