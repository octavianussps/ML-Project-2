import tensorflow as tf
from helpers import *
from unetmodel import *
import tensorflow.keras as keras
import sys




PATH_LRELU = '../models/unetLReLU.h5'
PATH_TEST_DATA = '../data/test_set_images/'
PATH_PREDICTION_DIR = '../data/predictions/'
PATH_SUBMISSION = '../out/final_UNET_submission.csv'
TEST_SIZE = 50



def main():

        image_filenames_test = [PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png' for i
                                in range(TEST_SIZE)]
        image_filenames_predict = [PATH_PREDICTION_DIR + 'pred_' + str(i + 1) + '_unet.png' for i in range(TEST_SIZE)]
        print('Loading Model...')


        model_unet = keras.models.load_model(PATH_LRELU)
        gen_image_predictions_unet(model_unet, PATH_PREDICTION_DIR, *image_filenames_test)
        print("images predicted!")

        # Generates the submission
        generate_submission(model_unet, PATH_SUBMISSION, 1, *image_filenames_predict)
        print("csv file generated!")


if __name__ == '__main__':
    main()
