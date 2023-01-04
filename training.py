
import tensorflow as tf
import json
import numpy as np
import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pickle
import argparse
from tqdm.notebook import tqdm
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout, GRU, Reshape
from keras import backend as K
from keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler


##### GLOBAL VARIABLES #####
MEL_SHAPE = 128
N_CATEGORIES = 16
TIME_30_SHAPE = 1292
TIME_10_SHAPE = 431
TIME_3_SHAPE = 130
INPUT_30S_SHAPE = (TIME_30_SHAPE, MEL_SHAPE, 1)
INPUT_10S_SHAPE = (TIME_10_SHAPE, MEL_SHAPE, 1)
INPUT_3S_SHAPE = (TIME_3_SHAPE, MEL_SHAPE, 1)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE_ADAM = 0.0001
ADAM = keras.optimizers.Adam(learning_rate = LEARNING_RATE_ADAM)
REGULARIZERL1 = tf.keras.regularizers.l1(0.001)
REGULARIZERL2 = tf.keras.regularizers.l2(0.001)

path30sec = "Datasets/preprocess_mfcc/full_30sec"
path10sec = "Datasets/preprocess_mfcc/cut10s"
path3sec = "Datasets/preprocess_mfcc/cut3s"
pathmodel = "Models/"

##### METHODS #####

def normalize_data(train, val, test):
    """ performs a min-max normalisation over the data

    Args:
        train (_type_): training dataset
        val (_type_): validation dataset
        test (_type_): test dataset

    Returns:
        _type_: normalized datasets
    """
    # get max, min and compute the difference
    data_max = max(np.amax(train), np.amax(val), np.amax(test))
    data_min = min(np.amin(train), np.amin(val), np.amin(test))
    diff = data_max - data_min
    
    # normalize data
    train_norm = (train-data_min)/diff
    val_norm = (val-data_min)/diff
    test_norm = (test-data_min)/diff
    
    #return normalized dataset
    return train_norm, val_norm, test_norm

def scheduler(epoch, lr):
    """ scheduler to adjust the learning rate per epochs

    Args:
        epoch (_type_): the epoch
        lr (_type_): actual learning rate

    Returns:
        _type_: adjusted learning rate
    """
    if epoch <=7 :
        return lr 
    else:
        return lr * tf.math.exp(-0.1)

def load_data(data_path: str, input_shape):
    """loads preprocessed dataset

    Args:
        data_path (str): path to the dataset
        input_shape (_type_): the input shape

    Returns:
        _type_: training, validation and testing dataset
    """
    x_train, y_train = [],[]
    x_val, y_val = [],[]
    x_test, y_test = [],[]
    for type_i in os.listdir(data_path):
        if not type_i.startswith('.'):
            print(type_i)
            for genre_i in tqdm(os.listdir(data_path+"/"+type_i)):
                if str(genre_i).isdigit():
                    for file in os.listdir(data_path+"/"+type_i+'/'+genre_i):
                        if "_" in file:
                            track_id = file.split('_')[0] #exlcude the _XX.npy
                        else:
                            track_id = file.split('.')[0] #exclude the .npy
                        if track_id.isdigit():
                            f = data_path+"/"+type_i+'/'+genre_i+"/"+file              
                            arr = np.load(f)
                            if np.array(arr).shape == input_shape:
                                if type_i == "training":
                                    x_train.append(arr)
                                    y_train.append([1 if i ==int(genre_i) else 0 for i in range(N_CATEGORIES)])
                                elif type_i == "validation":
                                    x_val.append(arr)
                                    y_val.append([1 if i ==int(genre_i) else 0 for i in range(N_CATEGORIES)])
                                elif type_i == "test":
                                    x_test.append(arr)
                                    y_test.append([1 if i ==int(genre_i) else 0 for i in range(N_CATEGORIES)])
                            else:
                                print(np.array(arr).shape)
                                print('problem with '+ track_id + ": incorrect shape")
    # ensures/forces type
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_test, y_test = np.array(x_test), np.array(y_test)

    # min-max normalization
    print("Normalizing the data...")
    x_train, x_val, x_test = normalize_data(x_train, x_val, x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

def save_model(model, history, cut: str, name: str):
    """ saves model and training history

    Args:
        model (_type_): the model 
        history (_type_): the history of the training process
        name (str): the name of the model
    """

    model.save(pathmodel + cut + "/" + name)
    with open(pathmodel + cut + "/" +"history_" + name, 'wb') as f:
        pickle.dump(history.history, f)

##### MODELS
################################# Paper Model Models ####################################


def paper_model_30sec(x_train, x_val, y_train, y_val, epoch20=False):
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout, GRU, Reshape
    from keras import backend as K

    tf.keras.backend.set_image_data_format("channels_last")
    name ="papermodel"
    model = Sequential(name="fma_crnn_basic")

    # Convolutional Block 1
    model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', activation='elu')) # X_train holds the processed training spectrograms
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) #2 -> 1
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) #2 -> 1
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(4,4), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((17,128)))

    # Recurrent Layer
    model.add(GRU(32, return_sequences=True, activation="tanh"))
    model.add(GRU(32, return_sequences=False, activation="tanh"))

    model.add(Dropout(0.3))

    # Dense Layer
    model.add(Dense(64))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(16, activation="sigmoid"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20

    history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    save_model(model, history_CRNN, "30sec", name)

def paper_model_10sec(x_train, x_val, y_train, y_val, epoch20=False):
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout, GRU, Reshape
    from keras import backend as K

    tf.keras.backend.set_image_data_format("channels_last")
    name ="papermodel"
    model = Sequential(name="fma_crnn_basic")

    # Convolutional Block 1
    model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', activation='elu')) # X_train holds the processed training spectrograms
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) #2 -> 1
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) #2 -> 1
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(2,8), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((25,128)))

    # Recurrent Layer
    model.add(GRU(32, return_sequences=True, activation="tanh"))
    model.add(GRU(32, return_sequences=False, activation="tanh"))

    model.add(Dropout(0.3))

    # Dense Layer
    model.add(Dense(64))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(16, activation="sigmoid"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20

    history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    save_model(model, history_CRNN, "10sec", name)

def paper_model_3sec(x_train, x_val, y_train, y_val, epoch20=False):
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout, GRU, Reshape
    from keras import backend as K

    tf.keras.backend.set_image_data_format("channels_last")
    name ="papermodel"
    model = Sequential(name="fma_crnn_basic")

    # Convolutional Block 1
    model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', activation='elu')) # X_train holds the processed training spectrograms
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) #2 -> 1
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) #2 -> 1
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', activation='elu'))
    model.add(MaxPool2D(pool_size=(2,12), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((6,128)))

    # Recurrent Layer
    model.add(GRU(32, return_sequences=True, activation="tanh"))
    model.add(GRU(32, return_sequences=False, activation="tanh"))

    model.add(Dropout(0.3))

    # Dense Layer
    model.add(Dense(64))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(16, activation="sigmoid"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20

    history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    save_model(model, history_CRNN, "3sec", name)



################################# 30sec Models ####################################

def model_30sec_4conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_30sec")
    name = "4conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=64, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(4,4), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((17,128)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(256, return_sequences=True, activation="tanh"))

    model.add(GRU(256, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "30sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "30sec", name )

def model_30sec_3conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_30sec")
    name = "3conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"
    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=64, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(10,10), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))


    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((21,64)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(128, return_sequences=True, activation="tanh"))

    model.add(GRU(128, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=REGULARIZER))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "30sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "30sec", name )


def model_30sec_2conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_30sec")
    name = "2conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"
    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(24,32), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((26,32)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(64, return_sequences=True, activation="tanh"))

    model.add(GRU(64, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "30sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "30sec", name )

################################# 10 sec Models ####################################

def model_10sec_4conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_10sec")
    name = "4conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=64, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,4), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((10,128)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(256, return_sequences=True, activation="tanh"))

    model.add(GRU(256, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "10sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "10sec", name )


def model_10sec_3conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_10sec")
    name = "3conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,3), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=64, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(3,10), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))


    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((22,64)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(128, return_sequences=True, activation="tanh"))

    model.add(GRU(128, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "10sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "10sec", name )

def model_10sec_2conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_10sec")
    name = "2conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(8,32), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((26,32)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(64, return_sequences=True, activation="tanh"))

    model.add(GRU(64, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "10sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "10sec", name )

################################# 3 sec Models ####################################

def model_3sec_4conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_3sec")
    name = "4conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=64, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,12), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((6,128)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(256, return_sequences=True, activation="tanh"))

    model.add(GRU(256, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "3sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "3sec", name )

def model_3sec_3conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_3sec")
    name = "3conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 3
    model.add(Conv2D(filters=64, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,16), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((14,64)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(128, return_sequences=True, activation="tanh"))

    model.add(GRU(128, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "3sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "3sec", name )

def model_3sec_2conv(x_train, x_val, y_train, y_val, gru2=False, l1_loss=False, l2_loss=False, lrs=False, epoch20=False):

    #Build the model
    model = Sequential(name="crnn_3sec")
    name = "2conv"
    callback = None
    regularizer = None
    #if we use the scheduler on learning rate:
    if lrs:
        callback = LearningRateScheduler(scheduler)
        name+= "_LRS"

    if l1_loss:
        name += "_L1"
        regularizer = REGULARIZERL1
    elif l2_loss:
        regularizer = REGULARIZERL2
        name += "_L2"

    # Convolutional Block 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), input_shape=x_train.shape[1:], data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(2,2), data_format='channels_last'))
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Convolutional Block 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), data_format='channels_last', kernel_regularizer=regularizer))
    model.add(MaxPool2D(pool_size=(4,32), data_format='channels_last')) 
    model.add(BatchNormalization())
    # Dropout
    model.add(Dropout(0.1))

    # Reshape Layer (effectively squeezes the frequency dimension away)
    model.add(Reshape((15,32)))

    # Recurrent Layer
    if gru2:
        name += "_2GRU"
        model.add(GRU(64, return_sequences=True, activation="tanh"))

    model.add(GRU(64, return_sequences=False, activation="tanh"))


    # Dense Layer
    model.add(Dense(64,  kernel_regularizer=regularizer))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(16, activation="softmax"))

    print(model.summary())

    # compile le model
    model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # train the model
    if epoch20:
        name += "_20epochs"
        EPOCHS = 20
    
    if lrs: 
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback], shuffle=True)
        save_model(model, history_CRNN, "3sec", name )
    else:
        history_CRNN = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
        save_model(model, history_CRNN, "3sec", name )




def main(args):
    """ runs the whole preprocessing pipeline according to command line arguments
    Args: 
        command line arguments including file paths and verbose flag
    """   

    tf.keras.backend.set_image_data_format("channels_last")
    if args.verbose:
        print("Training the model with following params: ")
        print(args)
    
    if (args.s30sec and args.s10sec) or (args.s30sec and args.s3sec) or (args.s10sec and args.s3sec):
        print("error: multiple sample/snippet size selected. Please chose only 1")
        return
    if (args.s4convo and args.s3convo) or (args.s4convo and args.s2convo) or (args.s3convo and args.s2convo):
        print("error: multiple number of convolution block selected. Please chose only 1")
        return

    if args.l1loss and args.l2loss:
        print("error: you selecteded L1 AND L2 loss (regularization), please only chose 1")
        return
    # specific case where we run the model from the paper
    if args.papermodel:
        if args.s30sec:
            if args.verbose:
                print("Loading the data...")
            x_train, y_train, x_val, y_val, x_test, y_test = load_data(path30sec, INPUT_30S_SHAPE)
            if args.verbose:
                print("Training the model from the original paper...")
            paper_model_30sec(x_train, x_val, y_train, y_val, epoch20=args.epoch20)
        elif args.s10sec:
            if args.verbose:
                print("Loading the data...")
            x_train, y_train, x_val, y_val, x_test, y_test = load_data(path10sec, INPUT_10S_SHAPE)
            if args.verbose:
                print("Training the model from the original paper...")
            paper_model_10sec(x_train, x_val, y_train, y_val, epoch20=args.epoch20)
        elif args.s3sec:
            if args.verbose:
                print("Loading the data...")
            x_train, y_train, x_val, y_val, x_test, y_test = load_data(path3sec, INPUT_3S_SHAPE)
            if args.verbose:
                print("Training the model from the original paper...")
            paper_model_3sec(x_train, x_val, y_train, y_val, epoch20=args.epoch20)

    # 30sec sample models
    if args.s30sec:
        if args.verbose:
            print("Loading the data...")
        x_train, y_train, x_val, y_val, x_test, y_test = load_data(path30sec, INPUT_30S_SHAPE)
        if args.s4convo:
            if args.verbose:
                print("Training the model ...")
            model_30sec_4conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)
        elif args.s3convo:
            if args.verbose:
                print("Training the model ...")
            model_30sec_3conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)
        elif args.s2convo:
            if args.verbose:
                print("Training the model ...")
            model_30sec_2conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)

    # 10sec sample models
    elif args.s10sec:
        if args.verbose:
            print("Loading the data...")
        x_train, y_train, x_val, y_val, x_test, y_test = load_data(path10sec, INPUT_10S_SHAPE)
        if args.s4convo:
            if args.verbose:
                print("Training the model ...")
            model_10sec_4conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)
        elif args.s3convo:
            if args.verbose:
                print("Training the model ...")
            model_10sec_3conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)
        elif args.s2convo:
            if args.verbose:
                print("Training the model ...")
            model_10sec_2conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)

    # 3sec sample models
    elif args.s3sec:
        if args.verbose:
            print("Loading the data...")
        x_train, y_train, x_val, y_val, x_test, y_test = load_data(path3sec, INPUT_3S_SHAPE)
        if args.s4convo:
            if args.verbose:
                print("Training the model ...")
            model_3sec_4conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)
        elif args.s3convo:
            if args.verbose:
                print("Training the model ...")
            model_3sec_3conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)
        elif args.s2convo:
            if args.verbose:
                print("Training the model ...")
            model_3sec_2conv(x_train, x_val, y_train, y_val, gru2=args.grux2, l1_loss=args.l1loss, l2_loss=args.l2loss, lrs=args.lrscheduler, epoch20=args.epoch20)


    if not (args.s30sec or args.s10sec or args.s3sec):
        print("error: no sample/snippet size selected. Please chose 1")
        return

    

    if not (args.s4convo or args.s3convo or args.s2convo):
        if not args.papermodel:
            print("error: no number of convolution block selected. Please chose 1")
            return

# running this file from command-line will do a full preprocessing pass on specified data
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='take model params, outputs trained model')
    parser.add_argument('-30sec', '--30sec', dest='s30sec', help='do you want to train a model with 30sec sample?', action='store_true')
    parser.add_argument('-10sec', '--10sec', dest='s10sec', help='do you want to train a model with 10sec sample?', action='store_true')
    parser.add_argument('-3sec', '--3sec', dest='s3sec', help='do you want to train a model with 3sec sample?', action='store_true')
    
    parser.add_argument('-papermodel', '--papermodel', dest='papermodel', help='do you want to use the model form the original paper?', action='store_true')

    parser.add_argument('-4c', '--4convo', dest='s4convo', help='do you want to train a model with 4 convolution layer blocks?', action='store_true')
    parser.add_argument('-3c', '--3convo', dest='s3convo', help='do you want to train a model with 3 convolution layer blocks?', action='store_true')
    parser.add_argument('-2c', '--2convo', dest='s2convo', help='do you want to train a model with 2 convolution layer blocks?', action='store_true')
    
    parser.add_argument('-l1', '--l1loss', dest='l1loss', help='do you want to use l1 loss?', action='store_true')
    parser.add_argument('-l2', '--l2loss', dest='l2loss', help='do you want to use l2 loss?', action='store_true')
    
    parser.add_argument('-lrs', '--lrscheduler', dest='lrscheduler', help='do you want to use the LRS during training', action='store_true')
    parser.add_argument('-gru2', '--grux2', dest='grux2', help='do you want to add a second GRU?', action='store_true')
    parser.add_argument('-ep20', '--epoch20', dest='epoch20', help='want to run only 20epoch', action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', help='do you want to print information', action='store_false')
    args = parser.parse_args()
    main(args)