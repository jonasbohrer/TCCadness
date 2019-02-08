import warnings
warnings.simplefilter('ignore')

import imp
import matplotlib.pyplot as plot
import numpy as np
import os

import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils

import innvestigate
import innvestigate.utils as iutils

epochs = 20
batch_size = 128
checkpoints = 1 #Number of epochs between each checkpoint
train_size = 60000
test_size = 6000

# Use utility libraries to focus on relevant iNNvestigate routines.
mnistutils = imp.load_source("utils_mnist", "../utils/utils_mnist.py")

# Load data
# returns x_train, y_train, x_test, y_test as numpy.ndarray
data_not_preprocessed = mnistutils.fetch_data(train_size, test_size)

# Create preprocessing functions
input_range = [-1, 1]
preprocess, revert_preprocessing = mnistutils.create_preprocessing_f(data_not_preprocessed[0], input_range)

# Preprocess data
data = (
    preprocess(data_not_preprocessed[0]), keras.utils.to_categorical(data_not_preprocessed[1], 10),
    preprocess(data_not_preprocessed[2]), keras.utils.to_categorical(data_not_preprocessed[3], 10),
)

if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)
    
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(10,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer=keras.initializers.RandomUniform(),
    bias_initializer=keras.initializers.RandomUniform(),
    input_shape=(28,28,1)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(10,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer=keras.initializers.RandomUniform(),
    bias_initializer=keras.initializers.RandomUniform()))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(100,
    kernel_initializer=keras.initializers.RandomUniform(),
    bias_initializer=keras.initializers.RandomUniform(),
    activation='relu'))

model.add(keras.layers.Dense(10,
    kernel_initializer=keras.initializers.RandomUniform(),
    bias_initializer=keras.initializers.RandomUniform(),
    activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(momentum=0.1, lr=0.005),
    metrics=['accuracy'])

n = 0

while n < epochs:
    model.fit(data[0], data[1], epochs=checkpoints, batch_size=batch_size)
    n +=checkpoints

    scores = model.evaluate(data[2], data[3], batch_size=batch_size)
    print("Scores on test set for run {}: loss/accuracy={}".format(n, tuple(scores)))
    model.save('models/test_model{}.h5'.format(n))

    if n+checkpoints > epochs:
        checkpoints = epochs - n