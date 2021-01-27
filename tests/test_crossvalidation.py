import torch
import MLP
import matplotlib.pyplot as plt

# PARAMS = {
# model
dim = 6
train_size = 10000
lr = 1e-1
wd = 0
# pyTorch
device = torch.device("cpu")
# Cross validation
CV_params = {
    "epochs": 100,
    "folds": 5,
    "minibatches": 1,
    "keep last training": False,
}
#}
# ==============================================================================
# The training data for this test case is only artifically random drawn data of
# the size given in PARAMS
# ==============================================================================
# tuple of data,label
train_data = (torch.rand(train_size,dim,dtype=torch.cdouble,device=device),torch.rand(train_size,dim,dtype=torch.cdouble,device=device))

train_data = torch.utils.data.TensorDataset(*train_data)

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
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=wd)

    loss_train_avg,loss_train_var,loss_valid_avg,loss_valid_var = MLP.CrossValidation(
        data=train_data,
        model=model,
        optimizer=optimizer,
        loss_function=loss_fct,
        CV_params=CV_params
    )

    abscissa = torch.arange(CV_params["epochs"])
    plt.plot(abscissa,loss_train_avg,'o-',c='b',label="Loss Train")
    plt.plot(abscissa,loss_train_avg-loss_train_var.sqrt(),'-.',c='b')
    plt.plot(abscissa,loss_train_avg+loss_train_var.sqrt(),'-.',c='b')
    plt.fill_between(abscissa, loss_train_avg-loss_train_var.sqrt(),loss_train_avg+loss_train_var.sqrt(),color='b',alpha = 0.5)


    plt.plot(abscissa,loss_valid_avg,'o-',c='g',label="Loss Valid")
    plt.plot(abscissa,loss_valid_avg-loss_valid_var.sqrt(),'-.',c='g')
    plt.plot(abscissa,loss_valid_avg+loss_valid_var.sqrt(),'-.',c='g')
    plt.fill_between(abscissa, loss_valid_avg-loss_valid_var.sqrt(),loss_valid_avg+loss_valid_var.sqrt(),color='g',alpha = 0.5)

    # In case you like error bars more ;)
    #plt.errorbar(abscissa,loss_train_avg,yerr=loss_train_var.sqrt(),fmt='o',markersize=2,capsize=2,elinewidth=1,c='b',ecolor='b',label="Loss Train")
    #plt.errorbar(abscissa,loss_valid_avg,yerr=loss_valid_var.sqrt(),fmt='o',markersize=2,capsize=2,elinewidth=1,c='g',ecolor='g',label="Loss Valid")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross Validation Result: Linear Layer Learning Random Data")
    plt.grid()
    plt.legend()
    plt.show()
    plt.clf()


# ==============================================================================
# Call the test functions
# ==============================================================================
test_sll() # simple linear layer + ReLU
