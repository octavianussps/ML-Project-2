from helpers import *
from cnn import CnnModel
#from utils.metrics import f1_scores

data_dir = '../data/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
PATH_WEIGHTS = '../models/weightsFinal.h5'
PATH_WEIGHTS_PATCH4 = '../models/weightsPatch4.h5'

num_images = 900
nb_epochs = 100



def main():
    # Instanciate the model
    cnn = CnnModel()
    
    PATH_WEIGHTS = PATH_WEIGHTS_PATCH4
    
    # Load data
    tr_imgs, gt_imgs = load_images(train_data_filename, train_labels_filename, num_images)

    # Train the model
    history = cnn.train_CNN_model(gt_imgs, tr_imgs, nb_epochs)
    
    # Save the weights
    cnn.save_weights(PATH_WEIGHTS)



if __name__ == '__main__':
    main()
