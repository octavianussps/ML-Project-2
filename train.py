from helpers import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ConvNN import ConvNN


root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
gt_dir = root_dir + "groundtruth/"
filepath_for_weights = root_dir + "weightsCNN.h5"

num_images = 900
num_epochs = 100


def main():
    
    
    # load images and their groundtruths
    imgs, gt_imgs = load_images(image_dir, gt_dir, num_images)
    
    # Instanciate the model
    CNN = ConvNN()
    
    history = CNN.train(imgs, gt_imgs, num_epochs)
    
    # save the updated weights
    CNN.save(filepath=filepath_for_weights, 
             overwrite=True, 
             include_optimizer=True,
             save_format=None, 
             options=None)
        
        
if __name__ == '__main__':
    main()
