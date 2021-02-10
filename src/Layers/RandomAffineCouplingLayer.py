import torch

class SlicingFunction(torch.autograd.Function):
    r"""
        The Random Affine Coupling layer requires a slicing operation when e.g. a
        particular partition of the input is passed to the affine members.
        PyTorch has no autograd support for .index yet therefore we implement it
        our selfs.

        Note, tensor.index[indices] is a linear operation on a finite vector space
        hence there is a matrix representing this. In particular, if the input
        vector is of size N a matrix of size N/2 x N can represent the partitioning,
        assuming both partitions have the same size.
        The vector indices is spread along this N/2 x N matrix such that each row
        has exactly one element which is non zero (i.e. only one one) and the
        particular column is defined by the element in the vector indices.
        Example:
            N = 4; indices = [1,3]

            x = (x0,x1,x2,x3)^T

            A = ( 0,1,0,0 )
                ( 0,0,0,1 )

            => A x = (x1,x3)^T

        This representation allows to compute the wirtinger derivative by
            d/dx Ax = A
            d/dx^* Ax = 0
        which, going back to the slicing operation is just ouputing the grad_output
        at the allowed indices.
    """
    @staticmethod
    def forward(self,input,mask):
        r"""
            input: torch.tensor
                Input tensor which is sliced in the last dimension
            mask: torch.tensor; dtype=bool
                Mask with the same shape as input.size()[-1] corresponing containing
                True for every element which should be passed and False else.
        """
        self.save_for_backward(mask)
        self.input_size = input.size()

        return input[...,mask]

    @staticmethod
    def backward(self,grad_output):
        r"""
            grad_output: torch.tensor
                Output from previous backpropagation steps
        """
        mask, = self.saved_tensors
        out = torch.zeros(size=self.input_size,dtype=grad_output.dtype)
        out[:,mask] = grad_output
        return out,None

slicingFunc = SlicingFunction.apply

class RandomAffineCouplingFunction(torch.autograd.Function):
    r"""
        Autograd building block for the random affine coupling layer. Let f be the
        multiplicative affine member and g be the additive one. Further, let A,B
        be integer partitions of equal kardianlity forming the index space of
        the input vector x. i.e. if (x_i) i = 0,1,...N-1 then A U B = {0,1,...,N-1}

        f,g: C^(N/2) --> C^(N/2)

        The forward pass is defined by

        z: C^N x C^(N/2) x C^(N/2) --> C^N:
        z(x,f(h_B(x)),g(h_B(x))) =  (             h_A(x)             )
                                    ( f(h_B(x)) * h_A(x) + g(h_B(x)) )

        where h_{A,B} are the slicing operations defined above.
    """

    @staticmethod
    def forward(self,input,mul_res,add_res,indices,mask_A,mask_B,masked_inputs = None):
        r"""
            input: torch.Tensor
                Input tensor may have any dimension, the last dimension is assumed
                to be the physical one being processed by the slicing operation
                h_A,h_B.
            mum_res: torch.Tensor
                Result of the multiplicative affine coupling member f(h_B(x)).
                Must have the same shape as the input tensor apart from the last
                dimension which has half of the size as the input tensor.
            add_res: torch.Tensor
                Result of the additive affine coupling member g(h_B(x)).
                Must have the same shape as the input tensor apart from the last
                dimension which has half of the size as the input tensor.
            indices: torch.Tensor
                Indices belonging to the B partition, used to access the output
                vector in the B components
            mask_A: torch.Tensor; dtype=torch.bool
                Mask for the A partition indices
            mask_B: torch.Tensor; dtype=torch.bool
                Mask for the B partition indices
            masked_inputs: (torch.Tensor,torch.Tensor); default = None
                Collects the A masked input and B masked input. This is required
                at training stage as the gradients in respect to the input reguire
                the gradients for f,g which can be compute only if these arguments
                are apparent.
                At inference stage this might be neglected.
        """
        output = input.clone()
        if masked_inputs is None:
            # ToDo, do we really need this?
            self.save_for_backward(input,mul_res,add_res,indices,mask_A,mask_B)
        else:
            self.save_for_backward(input,mul_res,add_res,indices,mask_A,mask_B,*masked_inputs)

        output[:,indices] = mul_res * masked_inputs[0] + add_res

        return output

    @staticmethod
    def backward(self,grad_output):
        r"""
            grad_output: torch.tensor
                Output from previous backpropagation steps
        """

        input,mul_res,add_res,indices,mask_A,mask_B,A_masked_input,B_masked_input = self.saved_tensors

        grad_input = grad_mul = grad_add = None

        #print(self.needs_input_grad)

        # ======================================================================
        # Grad by input: d(RACL)/dx
        # ======================================================================
        if self.needs_input_grad[0]:
            # determine the gradients of the affine members in respect to the B masked input
            mul_grad, = torch.autograd.grad(mul_res,B_masked_input, torch.ones_like(B_masked_input),retain_graph=True)
            add_grad, = torch.autograd.grad(add_res,B_masked_input, torch.ones_like(B_masked_input),retain_graph=True)

            # taking the derivative of the A components is 1 in the wirtinger and 0 for the conj wirtinger
            grad_input = torch.ones_like(input)

            # taking the derivative of the B components is f' * x_a + f + g' for the wirtinger and 0 for the conj wirtinger
            # again do not toch the A components using the indices
            grad_input[...,indices] = grad_output[...,indices] * (mul_grad * A_masked_input + mul_res + add_grad).conj()

        # ======================================================================
        # Grad by multiplicative affine member: d(RACL)/d(mul_res)
        # ======================================================================
        if self.needs_input_grad[1]:
            grad_mul = grad_output[...,indices]*input[...,indices].conj()

        # ======================================================================
        # Grad by additive affine member: d(RACL)/d(add_res)
        # ======================================================================
        if self.needs_input_grad[2]:
            grad_add = grad_output[...,indices]

        # Only the first three components (input, am_mul,am_add ) are supposed to
        # require grad. Thus return None otherwise
        return grad_input,grad_mul,grad_add,None,None,None,None

RandomAffineCouplingFunc = RandomAffineCouplingFunction.apply

class RandomAffineCouplingLayer(torch.nn.Module):
    def __init__(self,in_dim,am_mul,am_add):
        r"""
            in_dim: int
                input dimension
            am_mul: torch.nn.Module
                A function representing the multiplicatvie part of the affine transformation
                f(h_B(x)). Must handle half dim input vectors
            am_add: torch.nn.Module
                A function representing the additive part of the affine transformation
                g(h_B(x)). Must handle half dim input vectors

                This is a Module wrapper around the `RandomAffineCouplingFunction`.
        """
        super(RandomAffineCouplingLayer,self).__init__()
        self.dim = in_dim

        # Create the affine members (am): f(x) = am_mul(x) + am_add
        # Note: For simplicity these are stacks of linear layers our application
        #       has a possibility to change the network architecture in here.
        #self.am_mul = am_mul
        #self.am_add = am_add
        self.add_module("am_mul", am_mul)
        self.add_module("am_add", am_add)

        # To obtain the integer partitions A,B we draw a random permutation of the
        # total integer set i.e. 0,1,2,... in_dim -> sigma(0),sigma(1),... sigma(in_dim)
        # for a given permutation sigma.
        # As A,B should make up the entire index set only half of this permuted
        # set is taken i.e. sigma(0),sigma(2),sigma(4),...
        indices = torch.randperm(in_dim)[::2]
        # These indices will now acount for partition B, where the affine transformation
        # is set.
        # To ensure that A,B are orthogonal (i.e. i \in A => i \not\in B) create
        # a boolean mask A. (indexing: tensor[True] is returned; tensor[False] is not returned)
        # Recall: 0 <=> False; 1 <=> True
        mask_A = torch.ones(size=(in_dim,),dtype=torch.bool) # True,True,...
        # This mask is now FALSE for all indices sigma(0),sigma(2),sigma(4),...
        # i.e. it is TRUE for all indices sigma(1),sigma(3),sigma(5),...
        mask_A[indices] = False
        # By logical negation we obtain the mask for the B partition
        # This mask is now TRUE for all indices sigma(0),sigma(2),sigma(4),...
        # i.e. it is FALSE for all indices sigma(1),sigma(3),sigma(5),...
        mask_B = ~ mask_A

        # Register the indices and masks.
        self.register_buffer("indices",indices)
        self.register_buffer("mask_A" ,mask_A)
        self.register_buffer("mask_B" ,mask_B)

    def forward(self,input):
        # determine the masked input i.e.
        # x_i if i in index partition A: h_A(x)
        A_masked_input = slicingFunc(input,self.mask_A)
        # x_i if i in index partition B: h_B(x)
        B_masked_input = slicingFunc(input,self.mask_B)

        # compute the multiplicative affine member f(h_B(x))
        mul_res = self.am_mul(B_masked_input)
        # compute the additive affine member g(h_B(x))
        add_res = self.am_add(B_masked_input)

        # pass the results in the autograd building block
        # ToDo: How to check for inference (we can then pass A_masked_input,B_masked_input as None)
        #       effectively saving some memory 
        return RandomAffineCouplingFunc(input,mul_res,add_res,self.indices,self.mask_A,self.mask_B,(A_masked_input,B_masked_input))
