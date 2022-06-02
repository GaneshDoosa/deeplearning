# Import necessary packages
import os
import torch

# Path to original dataset
DATA_PATH = '../dataset/original'

# Define basepath to store modified dataset
BASE_PATH = '../dataset/'

# Define output path
OUTPUT_PATH = '../output/'

# Define the paths to separate train, validation and test splits
TRAIN = os.path.join(BASE_PATH, 'training')
VAL = os.path.join(BASE_PATH, 'validation')
TEST = os.path.join(BASE_PATH, 'evaluation')

# Initialize the list of class label names
CLASSES = ["Bread", "Dairy_product", "Dessert", "Egg", "Fried_food", "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]

# Imagenet Input Mean, Standard Deviation, Image Size 
MEAN = [0.485, 0.456, 0.406] # channel-wise, height-wise, width-wise
STD = [0.229, 0.224, 0.225] # channel-wise, height-wise, width-wise
IMAGE_SIZE = 224 # 224x224

# Set the device to be used for training and evaluation
DEVICE = torch.device('cuda')

# Specify training hyperparameters
LOCAL_BATCH_SIZE = 64 # Batch size during training
PRED_BATCH_SIZE = 4 # Batch size during inference
EPOCHS = 20
LR = 0.001

# Define paths to store training plot and training model
PLOT_PATH = os.path.join(OUTPUT_PATH, 'model_training.png')
MODEL_PATH = os.path.join(OUTPUT_PATH, 'food_classifier.pth')

# Extra Params
FIND_LR = False
LR_PLOT_PATH = os.path.join(OUTPUT_PATH, 'lr_plot.png')