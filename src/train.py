from helpers import *
from cnn_model import CnnModel
#from utils.metrics import f1_scores

data_dir = '../data/training/'
train_data_filename = data_dir + 'images/'
train_labels_filename = data_dir + 'groundtruth/'
PATH_WEIGHTS_try = '../models/weights.h5'
PATH_WEIGHTS_PATCH4 = '../models/weightsPatch4.h5'

num_images = 20
nb_epochs = 2



def main():
    # Instanciate the model
    cnn = CnnModel()
    
    #cnn.built=True
    PATH_WEIGHTS = PATH_WEIGHTS_PATCH4
    #cnn.load_weights(PATH_WEIGHTS_PATCH4)
    
    # Load data
    tr_imgs, gt_imgs = load_images(train_data_filename, train_labels_filename, num_images)
    #print("trimgshape",tr_imgs.shape)
    #print("gtimgshape",gt_imgs.shape)
    # Train the model
    history = cnn.train_model(gt_imgs, tr_imgs, nb_epochs)
    # Save the weights
    cnn.save_weights(PATH_WEIGHTS)

    # Generate plots
   # if history is not None:
#        plot_metric_history(f1_scores=f1_scores(history))


if __name__ == '__main__':
    main()
