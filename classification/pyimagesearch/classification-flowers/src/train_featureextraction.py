import torch
import config
from torchvision import transforms
from dataset import get_dataloader
from torchvision.models import resnet50
from torch import nn
from torch import optim
import time
from tqdm import tqdm
import utils
import sys
import numpy as np

def train(train_dataset, train_dataloader):
    # load up resnet50 model
    base_model = resnet50(pretrained=True)

    # since we are using the ResNet50 model as a feature extractor,and therefore no actual “learning” needs to take place in the body of the network, 
    # we freeze all layers in the body of the network by setting its parameters to non-trainable (by default they are trainable)
    for param in base_model.parameters():
        param.requires_grad = False
    
    # remove existing fc layer and append a new classification top to our feature extractor
    model_output_features = base_model.fc.in_features
    # we create a new FC layer head that consists of a single FC layer.
    base_model.fc = nn.Linear(model_output_features, len(train_dataset.classes))
    # move the model to device
    model = base_model.to(config.DEVICE)

    # Initialize loss function, optimizer
    loss_func = nn.CrossEntropyLoss()
    # notice that we are only providing the parameters of the classification top to our optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LR)

    # Calculate steps per epoch for training and validation set
    train_steps = len(train_dataset) // config.FEATURE_EXTRACTION_BATCH_SIZE
    val_steps = len(val_dataset) // config.FEATURE_EXTRACTION_BATCH_SIZE

    # Initialize a dictionary to store training history
    H = {'train_loss': [], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    # Loop over epochs
    print(f'[INFO] Training the network...')
    start_time = time.time()

    for epoch in tqdm(range(config.EPOCHS)):
        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        # initialize the number of correct predictions in the training and validation steps
        train_correct = 0
        val_correct = 0

        # loop over the training set
        for (i, (x,y)) in enumerate(train_dataloader):
            # send the input to device
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)

            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = loss_func(pred, y)

            # calculate gradients
            loss.backward()

            # check if we are updating the model parameters and if so update them, and zero out the previously accumulated gradients
            if (i+2) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # add loss to total training loss and calculate the number of correct predictions
            total_train_loss += loss.item()
            train_correct += (pred.argmax(1)==y).type(torch.float).sum().item()

        # switchoff autograd
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            # loop over validation set
            for (x,y) in val_dataloader:
                # send the input to device
                x,y = x.to(config.DEVICE), y.to(config.DEVICE)

                # make the predicitons and calculate the validation loss
                pred = model(x)
                loss = loss_func(pred,y)

                # add loss to total validation loss and calculate the number of correct predictions
                total_val_loss += loss.item()
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate average training and validation loss
        avg_train_loss = total_train_loss/train_steps
        avg_val_loss = total_val_loss/val_steps

        # calculate the training and validation accuracy
        train_accuracy = train_correct/len(train_dataset)
        val_accuracy = val_correct/len(val_dataset)

        # update the training history
        H['train_loss'].append(avg_train_loss)
        H['train_acc'].append(train_accuracy)
        H['val_loss'].append(avg_val_loss)
        H['val_acc'].append(val_accuracy)

        # print the model training and validation information
        print(f"[INFO] EPOCH: {epoch+1}/{config.EPOCHS}")
        print(f"Train loss: {avg_train_loss}, Train Accuracy: {train_accuracy}")
        print(f"Val loss: {avg_val_loss}, Val Accuracy: {val_accuracy}")

    end_time = time.time()
    print(f'[INFO] Total time taken to train model: {end_time-start_time}')
    return model, H

if __name__=='__main__':
    # Set reproducibility
    utils.torch_seed(43)

    # Define augumentation pipelines
    train_transform, test_transform = utils.augumentation_pipeline()

    # Create data loaders
    train_dataset, train_dataloader = get_dataloader(config.TRAIN, transforms=train_transform, batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE)
    val_dataset, val_dataloader = get_dataloader(config.VAL, transforms=train_transform, batch_size=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)

    # train model
    model, H = train(train_dataset, train_dataloader)

    # plot and save figure 
    utils.plot_save_fig(H, config.WARMUP_PLOT_PATH)

    # serialize the model state to disk
    torch.save(model.state_dict(), config.WARMUP_MODEL_PATH)