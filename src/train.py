import torch
from time import time_ns

def train(train_dataLoader,valid_dataLoader,model,optimizer,loss_function,num_epochs=100,stop_at_nan = True):
    r"""
        train_dataLoader: torch.util.data.DataLoader
            Dataloader for the training data set, expecting a get of (data,label)
        valid_dataLoader: torch.util.data.DataLoader
            Dataloader for the validation data set, expecting a get of (data,label)
        model: torch.nn.Module
            The model which should be trained
        optimizer: torch.optim.Optimizer
            The optimizer which should be apllied for training
        loss_function: torch.nn.Module
            The loss function used for the training
        num_epochs: int
            Number of training epochs, i.e. how often the network shall see the
            entire data set.
        Returns: (torch.tensor,torch.tensor)
            Returns the loss of the training process and validation respectively
            (train loss,valid loss)

        This method trains a given model on a given data set.
    """

    train_loss = torch.zeros(num_epochs)
    valid_loss = torch.zeros(num_epochs)

    # train for the num_epochs
    e_train = 0
    e_valid = 0
    for epoch in range(num_epochs):
        # ===========================
        # Train
        # ===========================
        s_train = time_ns() * 1e-9
        for i_mb,(mb_data,mb_label) in enumerate(train_dataLoader):
            # zero the gradient data
            optimizer.zero_grad()
            # make a prediction
            pred = model(mb_data)
            # compute the loss
            loss = loss_function(pred,mb_label)
            if torch.isnan(loss) and stop_at_nan:
                # if nan is found leave optimizer in normal state and return infinities for the loss
                optimizer.zero_grad()
                return torch.Tensor([float("Inf")]*num_epochs),torch.Tensor([float("Inf")]*num_epochs)
            # backpropagate
            loss.backward()
            # update parameters
            optimizer.step()
        train_loss[epoch] = loss.item()
        #print(f"Current Loss [{epoch}] = {loss.item():.3e}")
        e_train += time_ns() * 1e-9 - s_train

        # ===========================
        # Valid
        # ===========================
        s_valid = time_ns() * 1e-9
        with torch.no_grad():
            for data,label in valid_dataLoader:
                # make a prediction and compute loss to unseen data
                valid_loss[epoch] = loss_function(model(data),label).item()
        e_valid += time_ns() * 1e-9 - s_valid

        # output the train and validation time any 10th epoch
        if epoch % 10 == 9:
            print(f"Training epochs {epoch-9:02d}-{epoch:02d}|    "
                + f"Train: "
                + f"Time = {e_train:.2f}s, "
                + f"Loss = {train_loss[epoch]:.4e}"
                +  "    |    "
                + f"Valid: Time = {e_valid:.2f}s, "
                + f"Loss = {valid_loss[epoch]:.4e}"
            )
            # reset the timers
            e_train = 0
            e_valid = 0

    return train_loss,valid_loss
