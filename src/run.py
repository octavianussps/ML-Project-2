import tensorflow as tf
from helpers import *
from cnn_model import CnnModel
import tensorflow.keras as keras
import sys

PATH_WEIGHTS = '../models/weightsFinal.h5'
PATH_REG = '../models/weightsReg.h5'
PATH_UNET = '../models/unet.h5'
PATH_PATCH = '../models/weightsPatch4.h5'
PATH_TEST_DATA = '../data/test_set_images/'
PATH_PREDICTION_DIR = '../data/predictions/'
PATH_SUBMISSION = '../out/final_submission.csv'
TEST_SIZE = 50



def main():
    # We add all test images to an array, used later for generating a submission
    image_filenames_test = [PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png' for i
                            in range(TEST_SIZE)]
    image_filenames_predict = [PATH_PREDICTION_DIR + 'pred_' + str(i + 1) + '_unet.png' for i in range(TEST_SIZE)]
    print('Loading Model...')
    # Run the normal CNN model
    model = CnnModel()
    #model.built=True
    model.load_weights(PATH_WEIGHTS)
    # Generates the submission
    #modelType = 2
    #generate_submission(model, PATH_SUBMISSION, modelType, *image_filenames_test)
    # IF SMALLER PATCHES CHOSEN, COMPUTE THE MEAN OVER SOME OF THEM
    # IN ORDER TO SUBMIT ON AICROWD
    modelType = 2
    generate_submission(model, PATH_SUBMISSION, modelType, *image_filenames_test)
            
    
if __name__ == '__main__':
    main()
