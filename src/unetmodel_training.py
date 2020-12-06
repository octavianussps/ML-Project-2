import os
import matplotlib.pyplot as plt
from unetmodel import *
from helpers import *

TRAIN_DATA = "../data/training/"
MODEL_PATH = "../models/unetweights.h5"

BATCH = 16
EPOCHS = 100

SIZE_IMAGES = 400
NUM_IMAGES = 10
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
    
    
    # train the model
    unet = train(unet, imgs, gt_imgs, BATCH, EPOCHS)
    # save the weights
    model.save(MODEL_PATH)
    
if __name__== "__main__" :
    main()
