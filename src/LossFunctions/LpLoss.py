import torch

class LpLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input,target,p):
        self.mb_size,self.dim = input.size()
        # get difference
        output = input - target
        self.save_for_backward(output)
        self.p = p

        # get the element wise norm and raise it to p-power
        output = torch.pow( torch.sqrt(torch.square(output.real)+torch.square(output.imag)) , self.p)

        # sum all elements of the vector and take p-root
        output = torch.pow(torch.sum(output,dim=-1),1/self.p)

        # average over batch and return
        return torch.sum(output)/self.mb_size

    @staticmethod
    def backward(self,grad_output):
        difference, = self.saved_tensors
        mb_size,dim = difference.size()

        # single element norm
        diff_elem_norm = torch.sqrt(torch.square(difference.real)+torch.square(difference.imag))

        # get the first factor size=(mb_size,dim)
        # |x-y|^(p-2) * (x-y)
        output_grad = torch.pow( diff_elem_norm, self.p-2 ) * difference

        # multiply second factor size=(mb_size) as scalar to each minibatch mb
        # [ sum_i=0^dim |x_i - y_i|^p ]^((1-p)/p)
        output_grad *= (
            torch.pow(torch.sum(torch.pow(diff_elem_norm,self.p),dim=-1),(1-self.p)/self.p)
        ).view(mb_size,1).expand_as(output_grad)

        # the wirtinger derivative is the complex conjugation of the conjugate wirtinger derivative
        # i.e. (dL/dx^*)^* = dL/dx
        # apply the wirtinger gradien descent
        #           grad_output * (dL/dx)^* + grad_output^* * dL/dx^*   | grad_output = 1
        #        Re(grad_output)* (dL/dx)^*                             | grad_output = 1
        # assuming dim = 0 is the minibatch dimension
        return 0.5*output_grad/self.mb_size,None,None

LpLossFunc = LpLossFunction.apply

class LpLoss(torch.nn.Module):
    def __init__(self,p):
        r"""
            p: int or float
                Order of Lp loss function

            Lp Loss function takes an input and target vector, element of the
            corresponding Lp-space, and computes the the Lp-norm on the difference
            of those.
                LpLoss(input=x,target=y) = (sum_i (x_i-y_i)^p )^(1/p)
            If input/target are two dimensional a average is taken over the first
            dimension.
            The backward corresponds to the wirtinger calculus.
        """

        super(LpLoss, self).__init__()
        if p == 0:
            raise ValueError(f"Lp norm for p = 0 not implemented!")

        self.p = p

    def forward(self, input, target):
        return LpLossFunc(input,target,self.p)
