import torch

class ActionLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input,target,action,force = None):
        self.mb_size,self.dim = input.size()

        # save the force for backward
        self.force = force

        # get action difference
        output = torch.tensor( [action(input[i_mb,:].numpy()) - action(target[i_mb,:].numpy()) for i_mb in range(self.mb_size)], dtype = input.dtype )

        self.save_for_backward(input,output)

        # get the element wise norm and raise it to p-power
        output = torch.square(output.real)+torch.square(output.imag)

        # average over batch and return
        return torch.sum(output)/self.mb_size

    @staticmethod
    def backward(self,grad_output):
        input,difference_conj = self.saved_tensors

        difference_conj = difference_conj.conj()

        # get derivative of the Action
        # force = - dS/dPhi hence the factor of -1
        grad = -1./self.mb_size * torch.tensor( [self.force(input[i_mb,:].numpy()) * difference_conj[i_mb] for i_mb in range(self.mb_size)] )

        return grad,None,None,None

ActionLossFunc = ActionLossFunction.apply

class ActionLoss(torch.nn.Module):
    def __init__(self,action):
        super(ActionLoss, self).__init__()
        self.action = action.eval
        self.force = action.force

    def forward(self, input, target):
        return ActionLossFunc(input,target,self.action,self.force)
