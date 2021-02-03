import torch
import MLP
import matplotlib.pyplot as plt
import itertools

# PARAMS = {
dim = 6
train_size = 100
valid_size = 100
infer_size = 100
lr = 1e-1
wd = 10
num_epochs = 100
device = torch.device("cpu")
#}
torch.autograd.detect_anomaly()

# ==============================================================================
# The training data for this test case is only artifically random drawn data of
# the size given in PARAMS
# ==============================================================================
# tuple of data,label
train_data = (torch.rand(train_size,dim,dtype=torch.cdouble,device=device),torch.rand(train_size,dim,dtype=torch.cdouble,device=device))
valid_data = (torch.rand(valid_size,dim,dtype=torch.cdouble,device=device),torch.rand(valid_size,dim,dtype=torch.cdouble,device=device))
infer_data = (torch.rand(infer_size,dim,dtype=torch.cdouble,device=device),torch.rand(infer_size,dim,dtype=torch.cdouble,device=device))

# put everything in the dataLoader interface, suppressing minibatches (data size to small in our cases)
train_dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*train_data),batch_size=train_size)
valid_dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*valid_data),batch_size=valid_size)
infer_dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*infer_data),batch_size=infer_size)

@torch.no_grad()
def inference(model,loss_function=torch.nn.L1Loss()):
    for data,label in infer_dataLoader:
        loss = loss_function(model(data),label).item()

    return loss

# ==============================================================================
# Start with a simple linear layer
# ==============================================================================
def test_sll():
    model = torch.nn.Sequential(
        MLP.layer.CLinearLayer(in_features=dim,out_features=dim,dtype=torch.cdouble,bias=True),
#        MLP.activation.SplitActivation(torch.nn.Tanh()),
        MLP.activation.PhaseAmplitude(p=1,q=1),
    ).to(device)

    loss_fct = MLP.loss.LpLoss(p=1)

    # Note GPU training requires SGD as Adam uses addcmul_cuda
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)

    loss_train, loss_valid = MLP.train( train_dataLoader=train_dataLoader,
                                        valid_dataLoader=valid_dataLoader,
                                        model=model,
                                        optimizer=optimizer,
                                        loss_function=loss_fct,
                                        num_epochs=num_epochs)

    print(f"Average Inference Loss = {inference(model,loss_fct):.4e}")
    abscissa = torch.arange(num_epochs)
    plt.plot(abscissa,loss_train,'.-',c='b',label="Loss Train")
    plt.plot(abscissa,loss_valid,'.-',c='g',label="Loss Valid")
    plt.title(model.extra_repr())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
    plt.clf()

# ==============================================================================
# Affine coupling layer
# ==============================================================================
def test_acl():
    affiineMem_mul = [MLP.layer.CLinearLayer(in_features=dim,out_features=dim,dtype=torch.cdouble,bias=True)]*3
    affiineMem_add = [MLP.layer.CLinearLayer(in_features=dim,out_features=dim,dtype=torch.cdouble,bias=True)]*3

    model = torch.nn.Sequential(
        MLP.layer.RandomAffineCouplingLayer(in_dim=dim,am_mul=affiineMem_mul[0],am_add=affiineMem_add[0]),
        MLP.activation.PhaseAmplitude(p=1,q=1),
        MLP.layer.RandomAffineCouplingLayer(in_dim=dim,am_mul=affiineMem_mul[1],am_add=affiineMem_add[1]),
        MLP.activation.PhaseAmplitude(p=1,q=1),
        MLP.layer.RandomAffineCouplingLayer(in_dim=dim,am_mul=affiineMem_mul[2],am_add=affiineMem_add[2]),
        MLP.activation.PhaseAmplitude(p=1,q=1),
    ).to(device)

    loss_fct = MLP.loss.LpLoss(p=1)

    # Note GPU training requires SGD as Adam uses addcmul_cuda
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)

    loss_train, loss_valid = MLP.train( train_dataLoader=train_dataLoader,
                                        valid_dataLoader=valid_dataLoader,
                                        model=model,
                                        optimizer=optimizer,
                                        loss_function=loss_fct,
                                        num_epochs=num_epochs)

    print(f"Average Inference Loss = {inference(model,loss_fct):.4e}")
    abscissa = torch.arange(num_epochs)
    plt.plot(abscissa,loss_train,'.-',c='b',label="Loss Train")
    plt.plot(abscissa,loss_valid,'.-',c='g',label="Loss Valid")
    plt.title(model.extra_repr())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
    plt.clf()

# ==============================================================================
# Call the test functions
# ==============================================================================
#test_sll() # simple linear layer + activation
test_acl() # 3 Affine Coupling layers filled with linear layers
