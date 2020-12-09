import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from helpers import *
import matplotlib.pyplot as plt

NUM_CHANNELS =3

        
     
def double_conv(data, out_channel):
    # convolution block, 2 conv layers with relu each
    conv = layers.Conv2D(filters=out_channel, kernel_size=3, padding='same')(data)
    #conv = layers.ReLU()(conv)
    conv = layers.LeakyReLU(alpha=0.3)(conv)
    conv = layers.Conv2D(filters=out_channel, kernel_size=3, padding='same')(conv)
    #conv = layers.ReLU()(conv)
    conv = layers.LeakyReLU(alpha=0.3)(conv)
    return conv
    
    
def max_pool(data):        
    max_pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same")(data)
    return max_pooled


def crop_img(tensor, target_tensor):
    # helper function if one chooses padding = 'valid' in the max pool layers
    target_size = target_tensor.get_shape()[2]
    tensor_size = tensor.get_shape()[2]
    delta = tensor_size-target_size
    delta = delta // 2
    #print("tensor",tensor.get_shape())
    #print("target",target_tensor.get_shape())
    tensor = tensor[:,delta:tensor_size-delta, delta:tensor_size-delta,:]
    #print(tensor)
    return tensor

def up_conv(data, out_channels):
    upconv = layers.Conv2DTranspose(filters=out_channels, kernel_size=2, strides=(2,2))(data)
    return upconv    
        
def unet(data, num_filters):
    
    #encoder
    x1 = double_conv(data, num_filters)
    x2 = max_pool(x1)
    x3 = double_conv(x2, num_filters*2) #
    x4 = max_pool(x3)
    x5 = double_conv(x4, num_filters*4) #
    x6 = max_pool(x5)
    x7 = double_conv(x6, num_filters*8) #
    x8 = max_pool(x7)
    x9 = double_conv(x8, num_filters*16)
    
    
    #decoder
    x10 = up_conv(x9, num_filters*8)
    #x7 = crop_img(x7,x10)
    x11 = layers.concatenate([x10,x7], axis=3)
    x11 = double_conv(x11,num_filters*8)
    
    x12 = up_conv(x11, num_filters*4)
    #x5 = crop_img(x5,x12)
    x13 = layers.concatenate([x12,x5], axis=3)
    x13 = double_conv(x13,num_filters*4)
    
    x14 = up_conv(x13, num_filters*2)
    #x3 = crop_img(x3,x14)
    x15 = layers.concatenate([x14,x3], axis=3)
    x15 = double_conv(x15,num_filters*2)
    
    x16 = up_conv(x15, num_filters)
    #x1 = crop_img(x1,x16)
    x17 = layers.concatenate([x16,x1], axis=3)
    x17 = double_conv(x17,num_filters)
    
    out = layers.Conv2D(filters=1, kernel_size = 1, activation='sigmoid')(x17)
    print(out)
    return out 
    
    
    

def unet_model(img_size, num_filters):
    #instantiate and compile the model
    
    inputs = layers.Input((img_size, img_size, NUM_CHANNELS))
    outputs = unet(inputs, num_filters)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model




def train(model, x_train, y_train, batch_size, num_epochs):
    """Optimize the model and return the model with the training F1-Score"""
    try:
       #checkpoint = ModelCheckpoint("../models/unetBestWeights.h5",monitor='loss',verbose =1,save_best_only = True,mode = 'auto',period = 5)
        checkpoint = ModelCheckpoint("../models/unetLReLU.h5",monitor='loss',verbose =1,save_best_only = True,mode = 'auto',period = 5)
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[checkpoint], validation_split=0.1)
    except KeyboardInterrupt:
            pass
    print("model trained!")
    return model

    
    
    
    
    
    
    
    
    
