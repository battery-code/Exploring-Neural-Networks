# code credit: Udacity DL Nanodegree program

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from tqdm import tqdm


def get_train_val_data_loaders(batch_size, valid_size, transforms, num_workers):
    """
    Purpose: Helper function for setting up the data pipeline, specifically creating separate data loaders for training and validation.
    Need? 1) Explicitly separates train and validation data and used for all epochs so no data leakage occurs
    2) Additional features like randomization in selection where possible 3) manage parallel processing
    Input parameters:  
    -batch_size: how many images to be processed each time, 
    -valid_size: what fraction to be set aside for validation
    -transforms: Image transformations to be applied to the dataset
    -num_workers: Number of parallel processes for loading the data
    """
    
    # Downloads CIFAR10 dataset if not already present
    train_data = datasets.CIFAR10("data", train=True, download=True, transform=transforms)

    # We will now split into train and validation
    # Compute how many items we will reserve for the validation set
    n_tot = len(train_data)
    split = int(np.floor(valid_size * n_tot))

    # Splitting indices of data between training and validation randomly using randperm()
    shuffled_indices = torch.randperm(n_tot)
    train_idx, valid_idx = shuffled_indices[split:], shuffled_indices[:split]

    # define samplers so indices of data are picked randomly for training and validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # create pytorch DataLoader for training and validation
    # DataLoader picks the data using the sampler, batches data, coordinate parallel processing, provides iterator for data
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    return train_loader, valid_loader


def get_test_data_loader(batch_size, transforms, num_workers):
    """
    Purpose: Helper function for creating the data loader for testing.
    Need? Why not included above? Practically downloads/loads the test data and need not be clubbed with training/validation which uses the same dataset 
    """
    # We use the entire test dataset in the test dataloader
    test_data = datasets.CIFAR10("data", train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )

    return test_loader


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses()
    else:
        liveloss = None

    # Loop over the epochs and keep track of the minimum of the validation loss
    valid_loss_min = None
    logs = {}

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)  # -

            valid_loss_min = valid_loss

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss

            liveloss.update(logs)
            liveloss.send()

            
def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # we do not need the gradients
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()  # -

        # if the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model = model.cuda()

        # Loop over test dataset
        # We also accumulate predictions and targets so we can return them
        preds = []
        actuals = []
        
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)  # =
            # 2. calculate the loss
            loss_value = loss(logits, target).detach()  # =

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # NOTE: the predicted class is the index of the max of the logits
            pred = logits.data.max(1, keepdim=True)[1]  # =

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)
            
            preds.extend(pred.data.cpu().numpy().squeeze())
            actuals.extend(target.data.view_as(pred).cpu().numpy().squeeze())

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss, preds, actuals


def plot_confusion_matrix(pred, truth, classes):

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)
    confusion_matrix.index = classes
    confusion_matrix.columns = classes
    
    fig, sub = plt.subplots()
    with sns.plotting_context("notebook"):

        ax = sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d',
            ax=sub, 
            linewidths=0.5, 
            linecolor='lightgray', 
            cbar=False
        )
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")
 
    return confusion_matrix


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one epoch of training
    """

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()  # -

    # Set the model in training mode
    # (so all layers that behave differently between training and evaluation,
    # like batchnorm and dropout, will select their training behavior)
    model.train()  # -

    # Loop over the training data
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()  # -
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)  # =
        # 3. calculate the loss
        loss_value = loss(output, target)  # =
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()  # -
        # 5. perform a single optimization step (parameter update)
        optimizer.step()  # -

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    # During validation we don't need to accumulate gradients
    with torch.no_grad():

        # set the model to evaluation mode
        # (so all layers that behave differently between training and evaluation,
        # like batchnorm and dropout, will select their evaluation behavior)
        model.eval()  # -

        # If the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model.cuda()

        # Loop over the validation dataset and accumulate the loss
        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)  # =
            # 2. calculate the loss
            loss_value = loss(output, target)  # =

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss

