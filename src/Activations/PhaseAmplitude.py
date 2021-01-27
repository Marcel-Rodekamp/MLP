import torch

class PhaseAmplitudeFunction(torch.autograd.Function):
    @staticmethod
    def forward(self,input,p,q):
        self.p = p
        self.q = q

        # elementwise absolute value
        abs_in = torch.sqrt(input.real**2 + input.imag**2)

        self.save_for_backward(input)

        return input/(self.p+torch.pow(abs_in,self.q))

    @staticmethod
    def backward(self,grad_output):
        input, = self.saved_tensors

        # elementwise absolute value
        abs_in = torch.sqrt(input.real**2 + input.imag**2)

        # wirtinger derivative
        # Note: Leave out the denominator and devide it in the final result
        dwirt =self.p+torch.pow(abs_in,self.q)*(1-self.q)

        # conjugate wirtinger derivative
        # Note: Leave out the denominator and devide it in the final result
        dcwirt = torch.pow(input,2)*self.q*torch.pow(abs_in,self.q-2)

        # dwirt is real so we leave out the conjugate here
        # return None for the p,q gradients
        return (grad_output.conj()*dcwirt+grad_output*dwirt)/(self.p+torch.pow(abs_in,self.q)),None,None

PhaseAmplitudeFunc = PhaseAmplitudeFunction.apply

class PhaseAmplitude(torch.nn.Module):
    def __init__(self, p=1,q=1,*args,**kwargs):
        r"""!
            p: float
                additive parameter
            q: float
                power parameter

            This activation function can be used for complex valued neural networks.
            It computes
                f(x) = x/(p + |x|^q)
            and is based on the work by
            G. M. Georgiou and C. Koutsougeras,
            “Complex domain backpropagation”,
            IEEE Transactions on Circuits and Systems II: Analog and Digital SignalProcessing,
            vol. 39, no. 5, pp. 330–334, 1992
        """
        super(PhaseAmplitude,self).__init__(*args,**kwargs)
        self.p = p
        self.q = q

    def forward(self,input):
        return PhaseAmplitudeFunc(input,self.p,self.q)
