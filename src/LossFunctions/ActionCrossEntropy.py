import torch

class ActionCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input,target,action,force = None):
        self.mb_size,self.dim = input.size()

        # save the force for backward
        self.force = force


        # get action difference
        action_input = torch.tensor( [action(input[i_mb,:].numpy()) for i_mb in range(self.mb_size)], dtype = input.dtype )
        action_target = torch.tensor( [action(target[i_mb,:].numpy()) for i_mb in range(self.mb_size)], dtype = input.dtype )

        output = action_target * action_input.log() + (1-action_target) * (1-action_input).log()

        self.save_for_backward(input,action_input,action_target,output)

        output = torch.sqrt(output.real**2+output.imag**2)

        # average over batch and return
        output = torch.mean(output)

        return output

    @staticmethod
    def backward(self,grad_output):
        input,action_input,action_target,cross_entropy = self.saved_tensors

        action_grad_input = -torch.tensor( [self.force(input[i_mb,:].numpy()) for i_mb in range(self.mb_size)], dtype = input.dtype )

        grad = torch.unsqueeze( cross_entropy.conj()*((action_target)/(action_input) + (1-action_target)/(1-action_input)), -1 ) * action_grad_input
        return grad.conj(),None,None,None

ActionCrossEntropyFunc = ActionCrossEntropyFunction.apply

class ActionCrossEntropy(torch.nn.Module):
    def __init__(self,action):
        super(ActionCrossEntropy, self).__init__()
        self.action = action.eval
        self.force = action.force

    def forward(self, input, target):
        return ActionCrossEntropyFunc(input,target,self.action,self.force)
