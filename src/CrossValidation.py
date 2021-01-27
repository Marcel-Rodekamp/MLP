
import torch
from time import time_ns

from .train import train

def CrossValidation(data,model,optimizer,loss_function,CV_params):
    r"""
        data: torch.util.data.Dataset, torch.util.data.Subset
            Dataset which gets devided into folds to perform the cross validation.
        model: torch.nn.Module
            The model which should be trained
        loss_function: torch.nn.Module
            The loss function used for the training
        CV_params: dict
            Dictionary of allowed parameters used for the cross validation.
            Must contain
                * "epochs": int
                * "folds": int
                * "minibatches": int
                * "keep last training": bool
        Returns: (torch.tensor,torch.tensor,torch.tensor,torch.tensor): size = (number of epochs)
            Returns the loss of the training process and validation respectively
            (train loss average,train loss variance,valid loss average, valid loss variance)

        This method performes a cross validation for one given model. The procedure
        is as follows. Let N be the total data size
            1. Compute the validation data size. Nv = N // num_folds
            2. Identify

    """
    N = len(data)

    # 1. Compute validation data size Nv = N//num folds
    if(N % CV_params["folds"] != 0):
        raise ValueError(f"ERROR: number of folds ({CV_params['folds']}) is not a divisor of training data set size ({N}).")
    Nv = N // CV_params["folds"]

    # 2. Compute the minibatch size (N-Nv)//num_minibatches
    Nmb = (N - Nv)//CV_params["minibatches"]
    if (N-Nv) % Nmb != 0:
        raise ValueError(f"ERROR: number of minibatches ({CV_params['minibatches']}) is not a devisor of reduced training data set size ({N-Nv}).")

    # 3. prepare output arrays: train_,valid_loss and train_,valid_variance
    train_loss = torch.zeros(CV_params["epochs"],CV_params["folds"])
    valid_loss = torch.zeros(CV_params["epochs"],CV_params["folds"])

    # 4. Store the current model state
    model_state_backup = model.state_dict()

    for fold_id in range(CV_params["folds"]):
        print(f"Training Model on Fold ({fold_id}/{CV_params['folds']}) for {CV_params['epochs']} epochs.")
        # 5. Identify validation data  and prepare DataLoaders
        train_index_list = list(range(0,fold_id*Nv))+list(range((fold_id+1)*Nv,N))
        valid_index_list = list(range(fold_id*Nv,(fold_id+1)*Nv))

        # 5.1 pepare the data loader, no minibatches are taken at vailidation
        train_dataLoader = torch.utils.data.DataLoader(data,batch_size=Nmb,sampler=train_index_list)
        valid_dataLoader = torch.utils.data.DataLoader(data,batch_size=Nv,sampler=valid_index_list)

        # 6. train the model with identified validation data set
        # Note: train(...) is defined in src/train.py
        train_loss[:,fold_id],valid_loss[:,fold_id] = train(
            train_dataLoader=train_dataLoader,
            valid_dataLoader=valid_dataLoader,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            num_epochs=CV_params["epochs"]
        )

        # 7. Resetting the model
        # Optional: if CV_params["keep last training"] is True, model will not be
        #           resetted after last fold
        if(CV_params["keep last training"] and i_fold == num_folds-1):
            pass
        else:
            model.load_state_dict(model_state_backup)

    return train_loss.mean(dim=-1),torch.var(train_loss,dim=-1),valid_loss.mean(dim=-1),torch.var(valid_loss,dim=-1)
