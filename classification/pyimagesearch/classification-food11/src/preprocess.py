"""
 @ deprecated
 This function is no longer required, as we had already downloaded the processed dataset from kaggle
"""
import config
from imutils import paths
import shutil
import os

# Prepare hierarchical dataset structure
# Original Structure: dataset_type/X_Y.jpg (X=> LabelNumber, Y=> ImageNumber)
# Modified Structure: dataset_type/LABEL_NAME/IMAGENUMBER.jpg
def copy_images(root_dir, dest_dir):
    # Get the list of all images present in directory
    imagepaths =  list(paths.list_images(root_dir))
    print(f'[INFO] Total Images Found in {root_dir}: {len(imagepaths)}')

    for imagepath in imagepaths:
        # Extract class label from filename
        filename = imagepath.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split('_')[0])].strip()

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


if __name__=='__main__':
    # Copy the images to respective directories
    print('[INFO] Copying Images...')
    copy_images(os.path.join(config.DATA_PATH, 'training'), config.TRAIN)
    copy_images(os.path.join(config.DATA_PATH, 'validation'), config.VAL)
    copy_images(os.path.join(config.DATA_PATH, 'evaluation'), config.TEST)