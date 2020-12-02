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



def main(argv):
    # We add all test images to an array, used later for generating a submission
    image_filenames_test = [PATH_TEST_DATA + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png' for i
                            in range(TEST_SIZE)]
    image_filenames_predict = [PATH_PREDICTION_DIR + 'pred_' + str(i + 1) + '_unet.png' for i in range(TEST_SIZE)]
    print('Loading Model...')
    

    
    if len(argv) != 2:
        print(len(argv))
        raise Exception('Please, pass only one argument to the script')
    else:
        if argv[1] == '-unet':

            # Run the UNET model
            model_unet = keras.models.load_model(PATH_UNET)
            gen_image_predictions_unet(model_unet, PATH_PREDICTION_DIR, *image_filenames_test)

            # Generates the submission
            modelType = 1
            generate_submission(model_unet, PATH_SUBMISSION, modelType, *image_filenames_predict)

        elif argv[1] == '-normal':
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
            
                
        elif argv[1] == '-reg':
            #run the regression model
            model = cnnRegression()
            model.load_weights(PATH_REG)
            ### fix!!!
            
            generate_submission(model, PATH_SUBMISSION, False, *image_filenames_test)

        
        else:
            raise Exception('Please pass only "unet" or "normal" as argument to the script')


if __name__ == '__main__':
    main(sys.argv)
