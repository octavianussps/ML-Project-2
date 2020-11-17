from helpers import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




class ConvNN(keras.Model):
    
    def __init__(self):
        super(ConvNN, self).__init__()

        self.image_size = 400 # images have 400 width, 400 height
        self.num_channels = 3 # RGB channels
        self.alpha = 0.15 # parameter for activation function Leaky ReLu
        self.dropout_probability = 0.8 # dropout to avoid overfitting
        self.num_classes = 2 # groundtruth is black and white image (label 0 and 1)
        self.batch_size = 256
        self.reg_parameter = 1e-6 # parameter for regularization 
        
        self.build_model()
        
        
        
        
    def build_model(self):
        
        self.model = keras.Sequential()
        
        # input layer
        input_shape = (self.image_size, self.image_size, self.num_channels)
        self.model.add(layers.InputLayer(input_shape))
        
        # first conv layer
        self.model.add(layers.Conv2D(filters=64,kernel_size=5,strides=(1,1),padding="same"))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
        self.model.add(layers.Dropout(rate=self.dropout_probability))
        
        # second conv layer
        self.model.add(layers.Conv2D(filters=128,kernel_size=3,strides=(1,1),padding="same"))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
        self.model.add(layers.Dropout(rate=self.dropout_probability))
        
        # third conv layer
        self.model.add(layers.Conv2D(filters=256,kernel_size=3,strides=(1,1),padding="same"))
        self.model.add(layers.LeakyReLU(alpha=self.alpha))
        self.model.add(layers.MaxPool2D(pool_size=(2, 2), padding="same"))
        self.model.add(layers.Dropout(rate=self.dropout_probability))
        self.model.add(layers.Flatten())
        
        # fully connected layer
        self.model.add(layers.Dense(units=64,activation='relu'))
        self.model.add(layers.Dropout(rate=self.dropout_probability))
        
        self.model.add(layers.Dense(units=self.num_classes,activation='softmax'))
        
        
        # prints summary of the model
        self.model.summary()
        
        # configures the model for training
        self.model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy,
                           metrics=['accuracy'])
        
        
        
        
    def train(self, imgs, gt_imgs, num_epochs):
        
        np.random.seed(0)
              
        # This callback reduces the learning rate when the training accuracy does not improve any more
        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2,
                                                        verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

        # Stops the training process upon convergence
        stop_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=2, verbose=1,
                                                      mode='auto')
        history = None
        try:
            history = self.model.fit(x=imgs,
                                    y=keras.utils.to_categorical(gt_imgs, num_classes = self.num_classes),
                                    batch_size=16,
                                    epochs=num_epochs,
                                    verbose=1,
                                     callbacks=[lr_callback, stop_callback], 
                                     validation_split=0.2,
                                     shuffle=True,
                                      workers=1)
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        print('Training completed')
        return history
        
        
        
