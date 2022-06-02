from sklearn.metrics import classification_report
import torch
import config
from torchvision import transforms
from dataset import get_dataloader
from torchvision.models import densenet121
from torch import nn
from model import FoodClassifier
from torch import optim
import time
from tqdm import tqdm
import matplotlib.pyplot as plt 
import utils
import sys
import numpy as np

# Train the network
def train(train_dataset, train_dataloader, val_dataset, val_dataloader, NUM_GPU, GLOBAL_BATCH_SIZE):
    # Load up the densenet121 model
    base_model = densenet121(pretrained=True)

    # Loop over modules of the model and if the module is batch norm, set it to non-trainable
    for module, param in zip(base_model.modules(), base_model.parameters()):
        if isinstance(module, nn.BatchNorm2d):
            param.requires_grad = False

    # Initialize our custom model and flash it to the current device
    model = FoodClassifier(base_model, len(train_dataset.classes))
    model = model.to(config.DEVICE)

    # If we have more than 1 GPU then parallalize the model
    if NUM_GPU>1:
        model = nn.DataParallel(model)

    # Initialize loss function, optimizer, and gradient scaler
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR*NUM_GPU)
    scaler = torch.cuda.amp.GradScaler(enabled=True) # The Gradient scaler is a very helpful tool that will help bring mixed precision into the gradient calculations.

    # Find optimized learning rate
    if config.FIND_LR:
        print(f'[INFO] Finding optimized learning rate...')
        lrs, losses =  utils.find_lr(model, train_dataloader, loss_func, optimizer)
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig(config.LR_PLOT_PATH)
        sys.exit(f"LR Plot has been plotted and saved to {config.LR_PLOT_PATH}") 

    # Initialize a learning rate scheduler to decay it by a factor of 0.1 after every 10 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Calculate steps per epoch for training and validation set
    train_steps = len(train_dataset) // GLOBAL_BATCH_SIZE
    val_steps = len(val_dataset) // GLOBAL_BATCH_SIZE

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
        for (x,y) in train_dataloader:
            # enable mixed precisions
            with torch.cuda.amp.autocast(enabled=True):
                # send the input to device
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)

                # perform a forward pass and calculate the training loss
                pred = model(x)
                loss = loss_func(pred, y)

            # calculate gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # add loss to total training loss and calculate the number of correct predictions
            total_train_loss += loss.item()
            train_correct += (pred.argmax(1)==y).type(torch.float).sum().item()

        # update lr scheduler
        lr_scheduler.step()

        # switchoff autograd
        with torch.no_grad():
            # set the model to evaluation mode
            model.eval()

            # loop over validation set
            for (x,y) in val_dataloader:
                # enable mixed precisions
                with torch.cuda.amp.autocast(enabled=True):
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

# Evaluate the network
def evaluate(model, test_dataloader):
    print(f'[INFO] Evaluating the network...')

    # switch off autograd
    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        # initialize the list to store predictions
        preds = []

        # loop over test set
        for (x, _) in test_dataloader:
            # send the input to device
            x = x.to(config.DEVICE)

            # make the predictions and add item to list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    return preds

def plot_save_fig(H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)

if __name__=='__main__':
    # Set reproducibility
    utils.torch_seed(43)

    # Determine the number of GPUs we have
    NUM_GPU = torch.cuda.device_count()
    print(f'[INFO] Number of GPUs found: {NUM_GPU}')

    # Determine batch size based on the number of GPUs
    GLOBAL_BATCH_SIZE = config.LOCAL_BATCH_SIZE*NUM_GPU
    print(f'[INFO] Using a batch size of {GLOBAL_BATCH_SIZE}')

    # Define augumentation pipelines
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # Create data loaders
    train_dataset, train_dataloader = get_dataloader(config.TRAIN, transforms=train_transform, batch_size=GLOBAL_BATCH_SIZE)
    val_dataset, val_dataloader = get_dataloader(config.VAL, transforms=train_transform, batch_size=GLOBAL_BATCH_SIZE, shuffle=False)
    test_dataset, test_dataloader = get_dataloader(config.TEST, transforms=test_transform, batch_size=GLOBAL_BATCH_SIZE, shuffle=False)

    # train model
    model, H = train(train_dataset, train_dataloader, val_dataset, val_dataloader, NUM_GPU, GLOBAL_BATCH_SIZE)

    # evaluate model
    preds = evaluate(model, test_dataloader)

    # generate classification report
    print(classification_report(test_dataset.targets, preds, target_names=test_dataset.classes))

    # plot and save figure 
    plot_save_fig(H)

    # serialize the model state to disk
    torch.save(model.state_dict(), config.MODEL_PATH)