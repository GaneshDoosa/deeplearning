import config
from imutils import paths
import shutil
import os
import numpy as np

# Prepare hierarchical dataset structure
# Original Structure: dataset_type/X_Y.jpg (X=> LabelNumber, Y=> ImageNumber)
# Modified Structure: dataset_type/LABEL_NAME/IMAGENUMBER.jpg
def copy_images(imagepaths, dest_dir):
    print(f'[INFO] Copying images to {dest_dir}')
    for imagepath in imagepaths:
        # Extract class label from filename
        filename = imagepath.split(os.path.sep)[-1]
        label = imagepath.split(os.path.sep)[-2].strip()

        # Construct path to output directory
        dir_path = os.path.sep.join([dest_dir, label])

        # If the output directory doesn't exist create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Construct the path to output image file and copy it
        p = os.path.sep.join([dir_path, filename])
        shutil.copy2(imagepath, p)
    
    # Calculate the total number of images in destination directory
    current_total = list(paths.list_images(dest_dir))
    print(f'[INFO] Total number of images copied to {dest_dir}: {len(current_total)}')
