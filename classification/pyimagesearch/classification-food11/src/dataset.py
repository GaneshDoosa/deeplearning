# Dataset Download Link : https://www.kaggle.com/datasets/trolukovich/food11-image-dataset
# Import necessary libraries
# Pytorch DataLoaders
# Pytorch Datasets
import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

def get_dataloader(root_dir, transforms, batch_size, shuffle=True):
    # Create a dataset and use it to create dataloader
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE=='cuda' else False)

    return (dataset, dataloader)
