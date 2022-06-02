import config
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from torchvision import transforms

def find_lr(model, train_dataloader, loss_func, optimizer, init_value=1e-8, final_value=10.0):
    '''
        Method to find optimized learning rate
    '''
    # number of batches excluding the first batch
    batches_count = len(train_dataloader)-1
    update_step = (final_value/init_value)**(1/batches_count)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_dataloader:
        batch_num += 1
        inputs, labels = data
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # crash out if loss explodes
        if batch_num>1 and loss>4*best_loss:
            if(len(log_lrs) > 20):
                return log_lrs[10:-5], losses[10:-5]
            else:
                return log_lrs, losses
        
        # record the best loss
        if loss<best_loss or batch_num==1:
            best_loss=loss
        
        # store the values
        losses.append(loss.item())
        log_lrs.append((lr))

        # do the backward pass and optimize
        loss.backward()
        optimizer.step()

        # update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]['lr'] = lr
    
    if(len(log_lrs) > 20):
        return log_lrs[10:-5], losses[10:-5]
    else:
        return log_lrs, losses


def torch_seed(random_seed):
    '''
        Controlling sources of randomness
    '''
    # Sets the seed for generating random numbers
    torch.manual_seed(random_seed)

    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Numpy random ness
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

# Define augumentation pipelines
def augumentation_pipeline():
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

    return train_transform, test_transform

def plot_save_fig(H, plot_path):
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
    plt.savefig(plot_path)