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
train_size = 20000
test_size = 2000

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "../utils/utils.py")
imgnetutils = imp.load_source("utils_imagenet", "../utils/utils_imagenet.py")

exit(0)

# Load data
images, label_to_class_name = eutils.get_imagenet_data(net["image_shape"][0])


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
    
model = keras.models.Sequential([
    keras.layers.Conv2D(20, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(40, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

n = 0

while n < epochs:
    model.fit(data[0], data[1], epochs=checkpoints, batch_size=batch_size)
    n +=checkpoints

    scores = model.evaluate(data[2], data[3], batch_size=batch_size)
    print("Scores on test set for run {}: loss/accuracy={}".format(n, tuple(scores)))
    model.save('models/test_model{}.h5'.format(n))

    if n+checkpoints > epochs:
        checkpoints = epochs - n