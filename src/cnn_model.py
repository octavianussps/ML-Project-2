import tensorflow.keras as keras
from tensorflow.keras import layers
from helpers import *
import matplotlib.pyplot as plt


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

    def create_model(self):
        """Define the layer for CNN Model architecture composed of convulation layers, dense layer and apply activation function, optimizer, and loss
           Inputs: 
              model = the model of the layer
              input shape = shape of input of layer w.r.t input image
              optimizer = the optimizer used for the training
        """
        self.model = keras.Sequential()

        # Input layer
        input_shape = (self.window_size, self.window_size, self.channels)
        self.model.add(layers.InputLayer(input_shape))

        # First convolution layer : 5x5 filter, depth 64
        self.model.add(layers.Conv2D(filters = 64, kernel_size = 5, strides = self.stride, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size = self.pool, padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        # Second convolution layer : 3x3 filter, depth 128
        self.model.add(layers.Conv2D(filters = 128, kernel_size = 3, strides = self.stride, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size = self.pool, padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        # Third convolution layer : 3x3 filter, depth 256
        self.model.add(layers.Conv2D(filters = 256, kernel_size = 3, strides = self.stride, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size = self.pool, padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))
        self.model.add(layers.Flatten()) # flatten all the layer into one 

        # Fourth fully connected layer : 128 node
        self.model.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(self.regularization_value)))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.Dropout(self.dropout_prob * 2))

        # Softmax activation function
        self.model.add(
            layers.Dense(self.nb_classes, kernel_regularizer=keras.regularizers.l2(self.regularization_value),
                         activation='softmax'))

        # Adam optimizer
        optimizer = keras.optimizers.Adam()

        # Binary cross_entropy loss
        self.model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy,
                           metrics=['accuracy'])
        # Prints a summary for the model
        self.model.summary()

    def train_model(self, gt_imgs, tr_imgs, nb_epochs=100):
        """
        Train the CNN model
        Inputs:
           gt_imgs = ground truth images
           tr_imgs = training images
           nb_epoch = number of epoch for training
        
        Output:
           history = trained model
        """

        np.random.seed(1234)  # for simulation, to set same number random value.
        nb_images = tr_imgs.shape[0]

        padding_size = (self.window_size - self.patch_size) // 2
        
        # pad images to b aple to apply sliding window approach to pixels clos to the border!
        tr_imgs, gt_imgs = pad_images(tr_imgs, padding_size), pad_images(gt_imgs, padding_size)
        #print("trimgshape",tr_imgs.shape)
        #print("gtimgshape",gt_imgs.shape)
        def generate_minibatch():
            """
            Procedure to generate real-time minibatch , preparing cropping random windows and corresponding patches
            This runs in a parallel thread while the model is being trained.
            """
            while True:
                # Generate one minibatch
                x_batch = np.empty((self.batch_size, self.window_size, self.window_size, 3))
                y_batch = np.empty((self.batch_size, 2))

                for i in range(self.batch_size):
                    # Select a random image
                    idx = np.random.choice(nb_images)
                    tr_img = tr_imgs[idx]
                    gt_img = gt_imgs[idx]

                    shape = tr_img.shape

                    # Sample a random window from the image, center is the pixel in the center
                    center = np.random.randint(self.window_size // 2, shape[0] - self.window_size // 2, 2)
                    
                    # x range = center +/- half window
                    # y range analogously
                    window = tr_img[center[0] - self.window_size // 2:center[0] + self.window_size // 2,
                             center[1] - self.window_size // 2:center[1] + self.window_size // 2]
                    
                    # Find the corresponding ground truth patch: 16x16 pixels
                    gt_patch = gt_img[center[0] - self.patch_size // 2:center[0] + self.patch_size // 2,
                               center[1] - self.patch_size // 2:center[1] + self.patch_size // 2]
                   
                    # x_batch is the input
                    x_batch[i] = window
                    # convert groundtruth images into new groundtruth, since we transformed our problem into classification case
                    
                    y_batch[i] = to_categorical(patch_to_label(gt_patch), self.nb_classes)
                    
                yield x_batch, y_batch

        # Number of windows fed to the model per epoch
        samples_per_epoch = tr_imgs.shape[0] * tr_imgs.shape[1] * tr_imgs.shape[2] // (
                self.patch_size ** 2 * self.batch_size)

        print("samples per epoch %d" % samples_per_epoch)

        #callback: reduces the learning rate when the training accuracy does not improve any more
        start_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
                                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        # Stops the training process upon convergence
        stop_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=2, verbose=1,
                                                      mode='auto')
        history = None
        try:
            history = self.model.fit_generator(generate_minibatch(),
                                               steps_per_epoch=samples_per_epoch,
                                               epochs=nb_epochs,
                                               verbose=1,
                                               callbacks=[start_callback, stop_callback], workers=1)
        except KeyboardInterrupt:
            pass

        print('Training completed')
        return history

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
        print("finish saving model...")

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
        print('Predicting images...')
        #generate the patches for input
        X_patches = gen_patches(x, self.window_size,
                                self.patch_size)
        Y_pred = self.model.predict(X_patches)
        Y_pred = (Y_pred[:, 0] < Y_pred[:, 1]) * 1
        return group_patches(Y_pred, x.shape[0])
