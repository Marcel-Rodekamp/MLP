import torch

class RandomAffineCouplingLayer(torch.nn.Module):
    def __init__(self,in_dim,am_mul,am_add,*args,**kwargs):
        r"""
            in_dim: int
                input dimension
            am_mul: torch.nn.Module
                A function representing the multiplicatvie part of the affine transformation
                f(x_B). Must handle half dim input vectors
            am_add: torch.nn.Module
                A function representing the additive part of the affine transformation
                g(x_B). Must handle half dim input vectors
            *args:
                Arguments passed to the torch.nn.Module class
            *kwargs:
                Keyworded arguments passed to the torch.nn.Module class

            The Random Affine Coupling Layer draws two random integer partitions
            A,B of same size, such that (A U B) equals the index interval [1,dim].
            Then the input vector, x, can be partitioned into two subsets
            x_A,x_B. With that one defines the Affine Coupling Layer as
                       | x_A
                f(x) = |
                       | f(x_B) * x_A + g(x_B)
            where f,g are arbitrary functions. In prticular, we apply all sorts
            of neuronal networks as f,g (Sequence of Linear layers here). f,g are
            called affine members here.
        """
        super(RandomAffineCouplingLayer,self).__init__(*args,**kwargs)
        self.dim = in_dim

        # Create the affine members (am): f(x) = am_mul(x) + am_add
        # Note: For simplicity these are stacks of linear layers our application
        #       has a possibility to change the network architecture in here.
        self.am_mul = am_mul
        self.am_add = am_add

        # create (boolean) masks describing the integer partitions A,B
        self.mask_A = torch.zeros(size=(in_dim,), dtype=torch.uint8)
        self.mask_B = torch.ones(size=(in_dim,), dtype=torch.uint8)
        # Randomly draw which element indices, do not allow for doubling here!
        self.indices = torch.randperm(in_dim)[::2]
        self.mask_A[self.indices] = 1 # True
        self.mask_B[self.indices] = 0 # False
        # register the masks as parameter but do not change train them.
        self.mask_A = torch.nn.parameter.Parameter( self.mask_A,requires_grad=False )
        self.mask_B = torch.nn.parameter.Parameter( self.mask_B,requires_grad=False )

    def forward(self,input):
        return (input*self.mask_A) * self.am_mul(input*self.mask_B) \
                                   + self.am_add(input*self.mask_B)
