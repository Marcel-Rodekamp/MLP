import torch

class SplitActivation(torch.nn.Module):
    def __init__(self,activation,*args,**kwargs):
        r"""
            ativation: torch.nn.Module
                A activation function from the torch.nn.Module implementation.
            *args:
                Arguments passed to the torch.nn.Module class
            *kwargs:
                Keyworded arguments passed to the torch.nn.Module class

            The easiest way to make a activation available for complex weights and
            complex inputs is to split its real imag part. Let f(x) be a activation
            function. Then this class provides the interface for
                actFct = SplitActivation(f)
                actFct(x) = f(x.real) + 1j* f(x.imag)
        """
        super(SplitActivation,self).__init__(*args,**kwargs)

        self.activation = activation

    def forward(self,input):
        return torch.complex(self.activation(input.real),self.activation(input.imag))
