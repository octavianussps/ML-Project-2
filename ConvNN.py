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
        self.dropout_probability = 0.3 # dropout to avoid overfitting
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
        
        self.model.add(layers.Dense(units=self.num_classes,activation='relu'))
        
        
        self.model.summary()
