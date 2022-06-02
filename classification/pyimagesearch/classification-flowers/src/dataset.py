# Import necessary libraries
# Pytorch DataLoaders
# Pytorch Datasets
import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import preprocess
import utils
from imutils import paths
import numpy as np

def get_dataloader(root_dir, transforms, batch_size, shuffle=True):
    # Create a dataset and use it to create dataloader
    dataset = datasets.ImageFolder(root=root_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE=='cuda' else False)

    return (dataset, dataloader)

def build_dataset():
    root_dir = config.DATA_PATH
    utils.torch_seed(43)

     # Get the list of all images present in directory
    imagepaths =  list(paths.list_images(root_dir))
    print(f'[INFO] Total Images Found in {root_dir}: {len(imagepaths)}')
    np.random.shuffle(imagepaths)

    # Generate training and validation splits
    validation_size = int(len(imagepaths) * config.VAL_SPLIT)
    training_size = len(imagepaths) - validation_size
    training_imagepaths = imagepaths[:training_size]
    validation_imagepaths = imagepaths[training_size:]

    preprocess.copy_images(training_imagepaths, config.TRAIN)
    preprocess.copy_images(validation_imagepaths, config.VAL)

if __name__=='__main__':
    # Copy the images to respective directories
    print('[INFO] Building Dataset...')
    build_dataset()
