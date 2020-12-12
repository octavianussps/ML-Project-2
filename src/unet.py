import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from helpers import *
import matplotlib.pyplot as plt


######################################
# PARAMETERS
######################################
NUM_CHANNELS =3
        
     
def double_conv(data, out_channel):
    """
    Generate 2 convolutional layer for the model
    Inputs:
    data = embedded image (tensor)
    out_channel = the size of the output channel

    Outputs:
    conv = convolution layers for the model
    """
    # convolution block, 2 conv layers with relu each
    conv = layers.Conv2D(filters=out_channel, kernel_size=3, padding='same')(data)
    conv = layers.LeakyReLU(alpha=0.3)(conv)
    conv = layers.Conv2D(filters=out_channel, kernel_size=3, padding='same')(conv)
    conv = layers.LeakyReLU(alpha=0.3)(conv)
    return conv
    
    
def max_pool(data):
    """
    Generate maximum pooling layer
    Inputs:
    data = embedded image (tensor)

    Outputs:
    max_pooled = pooling layer
    """       
    max_pooled = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same")(data)
    max_pooled = layers.Dropout(0.5)(max_pooled)
    return max_pooled


def up_conv(data, out_channels):
    """
    Generate up convolutional layer for the model
    Inputs:
    data = embedded image (tensor)
    out_channel = the size of the output channel

    Outputs:
    upconv = up convolution layers for the model
    """
    upconv = layers.Conv2DTranspose(filters=out_channels, kernel_size=2, strides=(2,2))(data)
    upconv = layers.Dropout(0.5)(upconv)
    return upconv    
        
def unet(data, num_filters):
    """
    Generate unet layer
    Inputs:
    data = embedded image (tensor)
    num_filters = number of filters

    Outputs:
    out = unet layer
    """
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
    """
    Generate unet model
    Inputs:
    img_size  = image size
    num_filters = number of filters

    Outputs:
    model = unet model
    """
    #instantiate and compile the model
    
    inputs = layers.Input((img_size, img_size, NUM_CHANNELS))
    outputs = unet(inputs, num_filters)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model




def trainUNET(model, x_train, y_train, batch_size, num_epochs):
    """
    Optimize the model and return the model with the training F1-Score
    Inputs:
    model = unet model
    x_train = training images
    y_train = ground truth images
    batch_size = batch size
    num_epochs = number of epoch

    Outputs:
    model = unet model
    """
    try:
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)
    except KeyboardInterrupt:
            pass
    print("model trained!")
    return model

    
    
    
    
    
    
    
    
    
