import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from helpers import *



class CnnModel(keras.Model):

    def __init__(self):
    	"""
    	Define the parameters of class CNN Models
    	Variables:
    	  batch_size = batch size 
    	  patch_size = patch size
    	  windows_size = window size for context of the patch
    	  channels = channel size w.r.t color image mode
    	  nb_classes = black and whita labelling
    	  alpha = parameter for leaky relu
    	  dropout_prob = probability for dropout
    	  regularization_value = value for regularization
    	  pool = polling dimension
    	  stride = stride dimension
    	"""
    	super(CnnModel, self).__init__()
    	keras.backend.set_image_data_format('channels_last')
    	self.batch_size = 256
    	self.patch_size = 16
    	self.window_size = 72
    	self.channels = 3 
    	self.nb_classes = 2  # black and white labeling
    	self.alpha = 0.1  # leaky relu parameter
    	self.dropout_prob = 0.25  # random dropout parameter
    	self.regularization_value = 1e-6  # regularization value
    	self.pool = (2,2)
    	self.stride = (1,1)
    	self.create_model()

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        """
        Save the trained model's weight 
        Inputs:
    	    filepath = path to save the weight
    	    overwrite = boolean operation, overwrite the best weight
    	    include_optimizer = boolean operation, utilize the optimizer
    	    save_format = specific format to save
    	    signatures = signature of the model
    	    options = option
        """
        self.model.save_weights(filepath)
       

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        
        """
        Predict the images
        Inputs:
            x = input image
            batch_size = batch size
            verbose = verbose
            steps = step
            callback = callback
            max_queue_size = maximum queue size
            workers = workers
            use_multiprocessing = boolean operation, utilize multiprocessing
    
        Output:
        group_pathes = rebuilt predicted image
        """
        
        #generate the patches for input
        X_patches = gen_patches(x, self.window_size,
                                self.patch_size)
        Y_pred = self.model.predict(X_patches)
        Y_pred = (Y_pred[:, 0] < Y_pred[:, 1]) * 1
        return group_patches(Y_pred, x.shape[0])

    def create_model(self):
        """Define the layer for CNN Model architecture composed of convulation layers, dense layer and apply activation function, optimizer, and loss
           Inputs: 
              model = the model of the layer
              input shape = shape of input of layer w.r.t input image
              optimizer = the optimizer used for the training
        """
        self.model = keras.Sequential()

        # Initialization
        input_shape = (self.window_size, self.window_size, self.channels)
        self.model.add(layers.InputLayer(input_shape))

        # conv layer, 64 filters, each 5x5
        self.model.add(layers.Conv2D(filters = 64, kernel_size = 5, strides = self.stride, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size = self.pool, padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        # conv layer, 128 filters, each 3x3 
        self.model.add(layers.Conv2D(filters = 128, kernel_size = 3, strides = self.stride, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size = self.pool, padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        # conv layer, 256 filters, each 3x3 
        self.model.add(layers.Conv2D(filters = 256, kernel_size = 3, strides = self.stride, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size = self.pool, padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))
        self.model.add(layers.Flatten()) # flatten all the layer into one 

        # fully connected layer, 128 nodes
        self.model.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(self.regularization_value)))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.Dropout(self.dropout_prob * 2))

        # activation layer
        self.model.add(
            layers.Dense(self.nb_classes, kernel_regularizer=keras.regularizers.l2(self.regularization_value),
                         activation='softmax'))

        # optimizer : the Adam one
        optimizer = keras.optimizers.Adam()

        # define loss function
        self.model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy,
                           metrics=['accuracy'])
       
       
        self.model.summary()


    def train_CNN_model(self, x_train, y_train, nb_epochs=100):
        """
        Train the CNN model
        Inputs:
           x_train = training images
           y_train = ground truth images
           nb_epoch = number of epoch for training
        
        Output:
           history = trained model
        """
        def minibatch(self,x_train, y_train, nb_images):
            """
            Generates training data by cropping windows of the training images and computing their
            corresponding groundtruth window
            inputs : 
                x_train = training images
                y_train = groundtruth images
                nb_images
            """
            while True:
                # Generate one minibatch
                x_batch = np.empty((self.batch_size, self.window_size, self.window_size, 3))
                y_batch = np.empty((self.batch_size, 2))

                for i in range(self.batch_size):
                    
                    index = np.random.choice(nb_images)
                    x_image = x_train[index]
                    y_image = y_train[index]
                    shape = x_image.shape

                    # Sample a random window from the image, center is the pixel in the center
                    center = np.random.randint(self.window_size // 2, shape[0] - self.window_size // 2, 2)
                    
                    # x range = center +/- half window
                    # y range analogously
                    window = x_image[center[0] - self.window_size // 2:center[0] + self.window_size // 2,
                            center[1] - self.window_size // 2:center[1] + self.window_size // 2]
                    
                    # Find the corresponding ground truth patch: 16x16 pixels
                    groundtruth_patch = y_image[center[0] - self.patch_size // 2:center[0] + self.patch_size // 2,
                            center[1] - self.patch_size // 2:center[1] + self.patch_size // 2]
                
                    # x_batch is the input
                    x_batch[i] = window

                    # convert groundtruth images into new groundtruth, since we transformed our problem into classification case
                    y_batch[i] = to_categorical(patch_to_label(groundtruth_patch), self.nb_classes)
                    
                yield x_batch, y_batch

        np.random.seed(1234)  # for simulation, to set same number random value.
        nb_images = x_train.shape[0]

        padding_size = (self.window_size - self.patch_size) // 2
        
        # pad images to be able to apply sliding window approach to pixels clos to the border!
        x_train, y_train = pad_images(x_train, padding_size), pad_images(y_train, padding_size)

        # Number of windows fed to the model per epoch
        samples_per_epoch = x_train.shape[0] * x_train.shape[1] * x_train.shape[2] // (
                self.patch_size ** 2 * self.batch_size)
       
        history = None
        try:
            minibatch_generation = minibatch(self, x_train, y_train, nb_images)
            history = self.model.fit_generator(minibatch_generation,
                                               steps_per_epoch=samples_per_epoch,
                                               epochs=nb_epochs,
                                               verbose=1,
                                               workers=1)
        except KeyboardInterrupt:
            pass

        print('Trained')
        return history

    
