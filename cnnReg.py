import tensorflow.keras as keras
from tensorflow.keras import layers
from utils.helpers import *


class cnnRegression(keras.Model):

    def __init__(self):
        super(cnnRegression, self).__init__()
        keras.backend.set_image_data_format('channels_last')

        self.image_size = 400
        self.window_size = 32  # size of the window acting as context for the patch
        self.nb_channels = 3  # color images in RGB mode
        self.leak_alpha = 0.1  # leaky_relu parameter
        self.dropout_prob = 0.25  # random dropout parameter
        self.regularization_value = 1e-6  # regularization value
        self.nb_classes = 2  # black and white labeling
        self.batch_size = 20
        self.create_model()

    def create_model(self):
        """Creates and compile the CNN model """
        self.model = keras.Sequential()

        # Input layer
        input_shape = (self.window_size, self.window_size, self.nb_channels)
        self.model.add(layers.InputLayer(input_shape))

        # First convolution layer : 5x5 filter, depth 64
        self.model.add(layers.Conv2D(16, 5, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.MaxPool2D(padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))

        # Second convolution layer : 3x3 filter, depth 128
        self.model.add(layers.Conv2D(32, 9, padding='same'))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.MaxPool2D(padding='same'))
        self.model.add(layers.Dropout(self.dropout_prob))
        
        # Second convolution layer : 3x3 filter, depth 128
        #self.model.add(layers.Conv2D(64, 18, padding='same'))
        #self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        #self.model.add(layers.MaxPool2D(padding='same'))
        #self.model.add(layers.Dropout(self.dropout_prob))
        
        self.model.add(layers.Flatten())


        # fully connected layer : 64 node
        self.model.add(layers.Dense(128, kernel_regularizer=keras.regularizers.l2(self.regularization_value)))
        self.model.add(layers.LeakyReLU(alpha=self.leak_alpha))
        self.model.add(layers.Dropout(self.dropout_prob * 2))

        # Softmax activation function, 36*36 output nodes, one per pixel of the window in the groundtruth
        # softmax because we want 0 or 1 (black and white binary image)
        self.model.add(
            layers.Dense(self.window_size*self.window_size, kernel_regularizer=keras.regularizers.l2(self.regularization_value),
                         activation='softmax'))

        # Adam optimizer
        optimizer = keras.optimizers.Adam()

        # Binary cross_entropy loss
        self.model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy,
                           metrics=['accuracy'])
        # Prints a summary for the model
        self.model.summary()

    def train_model(self, gt_imgs, tr_imgs, nb_epochs=100):
        """Trains the CNN model """

        np.random.seed(1)  # for reproducibility
        nb_images = tr_imgs.shape[0]

        
        def generate_minibatch():
            """
            Procedure for real-time minibatch creation, preparing cropping random windows and corresponding patches
            This runs in a parallel thread while the model is being trained.
            """
            while True:
                # Generate one minibatch
                x_batch = np.empty((self.batch_size, self.window_size, self.window_size, 3))
                y_batch = np.empty((self.batch_size, self.window_size*self.window_size))


                for i in range(self.batch_size):
                    # Select a random image
                    idx = np.random.choice(nb_images)
                    tr_img = tr_imgs[idx]
                    gt_img = gt_imgs[idx]

                    shape = tr_img.shape

                    # Sample a random window from the image, top_left will be the top left pixel of the window
                    top_left = np.random.randint(0,self.image_size-self.window_size,size=2)

                    window = tr_img[top_left[0]:top_left[0] + self.window_size,
                             top_left[1]:top_left[1] + self.window_size]

                    # Find the corresponding ground truth patch
                    gt_patch = gt_img[top_left[0]:top_left[0] + self.window_size,
                             top_left[1]:top_left[1] + self.window_size]

                    x_batch[i] = window
                    y_batch[i] = np.ndarray.flatten(gt_patch)

                
                yield x_batch, y_batch

        # Number of windows fed to the model per epoch
        #samples_per_epoch = tr_imgs.shape[0] // self.batch_size
        samples_per_epoch = 20
        print("samples per epoch %d" % samples_per_epoch)
        
        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
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
                                               callbacks=[lr_callback, stop_callback], workers=1)
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
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
        self.model.save_weights(filepath)
        print("model saved !")

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        print('Predicting images..')
        
        # crop image into patches
        X_patches = gen_windows_for_reg(x, self.window_size)
        
        Y_pred = self.model.predict(X_patches)
        Y_pred = (Y_pred[:, 0] < Y_pred[:, 1]) * 1
        return group_patches(Y_pred, x.shape[0])
