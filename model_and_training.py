# -*- coding: utf-8 -*-

from __future__ import print_function


import numpy as np
from matplotlib import pyplot as plt
import gc


import tensorflow-gpu as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Reshape
from keras.layers import Input, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers import Input, merge, Convolution2D
from keras.initializers import constant
from keras.layers import Dense, Flatten, Reshape


from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from BRATs_data_unet_2 import load_train_data, load_val_data
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
# with tf.Session(config = config) as s:

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ==========================================================================
smooth = 1.
nclasses = 5 # no of classes, if the output layer is softmax
# nclasses = 1 # if the output layer is sigmoid
img_rows = 240
img_cols = 240
def step_decay(epochs):
    init_rate = 0.003
    fin_rate = 0.00003
    total_epochs = 24
    print ('ep: {}'.format(epochs))
    if epochs<25:
        lrate = init_rate - (init_rate - fin_rate)/total_epochs * float(epochs)
    else: lrate = 0.00003
    print ('lrate: {}'.format(model.optimizer.lr.get_value()))
    return lrate
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true.astype('float32'))
    y_pred_f = K.flatten(y_pred.astype('float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)




def cnnBRATsInit_unet():  
    # the output layer is softmax, loss should be categorical_crossentropy
    # Downsampling 2^4 = 16 ==> 240/16 = 15
    
    inputs = Input((img_rows, img_cols, 3))        
    # inputs = Input((img_rows, img_cols, 4))
    
    # Normalize all data before extracting features
    input_nor = BatchNormalization()(inputs)
    # Block 1
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(input_nor)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv1)
    conv1 = BatchNormalization()(conv1)
    # conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)    
    
    # Block 2
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(pool1) 
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv2)
    conv2 = BatchNormalization()(conv2)
    # conv2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    
    # this MaxPooling2D has the same result with the above line.
    # pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    # W_new = (W - F + 2P)/S + 1
    # W_new = (120 - 3 + 2*1)/2 + 1 = 60.5 --> 60 (round down)
    # P = 1, because the padding parameter is 'same', the size of W has no change
    # before divide by Stride parameter
    
    
    # Block 3
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(pool2)  
    # conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(conv3)
    conv3 = BatchNormalization()(conv3)
    # conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # dropout1 = Dropout(0.5)(pool3)
    
    # Block 4
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(pool3)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', 
                   kernel_regularizer=l2(0.01))(conv4)
    conv4 = BatchNormalization()(conv4)
    # conv4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # dropout2 = Dropout(0.5)(pool4)
    
    # Block 5
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(pool4)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(1024, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv5)
    conv5 = BatchNormalization()(conv5)
    # pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv5)
    # conv5 = Dropout(0.5)(conv5)
    
    # Block 6
    # axis = -1 same as axis = last dim order, in this case axis = 3
    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up6) 
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv6)
    conv6 = BatchNormalization()(conv6)
    # conv6 = Dropout(0.5)(conv6)
    
    # Block 7
    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up7) 
    # conv7 = BatchNormalization()(conv7)
    # conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv7)
    conv7 = BatchNormalization()(conv7)
    # conv7 = Dropout(0.5)(conv7)
    
    # Block 8
    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up8)
    # conv8 = BatchNormalization()(conv8)
    # conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv8)
    conv8 = BatchNormalization()(conv8)
    # conv8 = Dropout(0.5)(conv8)
    
    # Block 9
    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), 
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(up9) 
    # conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(0.01))(conv9)
    conv9 = BatchNormalization()(conv9)
    # conv9 = Dropout(0.5)(conv9)
    
    # Block 10
    conv10 = Conv2D(nclasses, kernel_size=(1, 1), padding='same')(conv9)
    
    # Block 10
    # curr_channels = nclasses
    _, curr_width, curr_height, curr_channels = conv10._keras_shape
    out_conv10 = Reshape((curr_width * curr_height, curr_channels))(conv10)
    act_soft = Activation('softmax')(out_conv10)
    
    model = Model(inputs=[inputs], outputs=[act_soft])


    sgd = SGD(lr=0.0001, decay=0.01, momentum=0.9, nesterov=True)    
    # sgd = SGD(lr=0.0001, decay=1, momentum=0.9, nesterov=True)
    #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
    
    # adam = Adam(lr=0.0001)
    # model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
    
     model.compile(optimizer='sgd', loss=dice_coef_loss, metrics=[dice_coef])
    
    # model.summary()
    return model

def save_trained_model(model):
    # apply the histogram normalization method in pre-processing step    
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_BRATs_unet_HN.json", "w") as json_file:    
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_BRATs_unet_HN.h5")   
    print("Saved model to disk")

def train_network():
    print('Loading and preprocessing training data...')

    imgs_train, imgs_label_train = load_train_data('HGG')
    # imgs_train, imgs_label_train = load_train_data('Full_HG')
    print('Imgs train shape', imgs_train.shape)  
    print('Imgs label shape', imgs_label_train.shape)
    
    print('Calculating mean and std of training data...')
    imgs_train = imgs_train.astype('float32') 
    imgs_train /= 255. 
    
    
    minv = np.min(imgs_train)  # mean for data centering
    maxv = np.max(imgs_train)  # std for data normalization
    print('min = %f, max = %f' %(minv, maxv))  
           
    print('Creating and compiling model...')    
    model = cnnBRATsInit_unet()


	#initialize the optimizer and model
    model.summary()
    
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    
    print('Fitting model...')
    batch_size = 10
    history = model.fit(imgs_train, imgs_label_train, batch_size=batch_size, epochs=50, 
                        verbose=1, shuffle=True, validation_split=0.05,
                        callbacks=[model_checkpoint])
    
    save_trained_model(model)
    
    # release memory in GPU and RAM
    del history
    del model
    for i in range(30):
        gc.collect()
    
    print('Evaluating model...')
    scores = model.evaluate(imgs_train, imgs_label_train, batch_size=4, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
        
if __name__ == '__main__':
    train_network()
