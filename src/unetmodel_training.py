import os
import matplotlib.pyplot as plt
from unet import *
from helpers import *

######################################
# PARAMETERS
######################################
TRAIN_DATA = "../data/training/"
MODEL_PATH = "../models/unetBestWeights.h5"
MODEL_PATH_LRELU = '../models/unetLReLU.h5'

BATCH = 16
EPOCHS = 500

SIZE_IMAGES = 400
NUM_IMAGES = 900
NUM_FILTERS = 32


def main():
    image_dir = TRAIN_DATA + "images/"
    gt_dir = TRAIN_DATA + "groundtruth/"
    
    files = os.listdir(image_dir)
    n = NUM_IMAGES
    
    
    print("Loading " + str(n) + " images")
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    print("Loading " + str(n) + " images")
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    
    imgs = np.asarray(imgs)
    gt_imgs = np.expand_dims(np.asarray(gt_imgs), axis=3)
    
    #instantiate the model
    unet = unet_model(SIZE_IMAGES, NUM_FILTERS)
    unet.summary()
    unet = keras.models.load_model(MODEL_PATH_LRELU)

    
    # train the model
    unet = trainUNET(unet, imgs, gt_imgs, BATCH, EPOCHS)
    # save the weights
    unet.save(MODEL_PATH_LRELU)
    
if __name__== "__main__" :
    main()
