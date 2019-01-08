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

# Use utility libraries to focus on relevant iNNvestigate routines.
mnistutils = imp.load_source("utils_mnist", "../utils_mnist.py")

# Load data
# returns x_train, y_train, x_test, y_test as numpy.ndarray
data_not_preprocessed = mnistutils.fetch_data()

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
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.load_weights('test_model10.h5')

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.fit(data[0], data[1], epochs=10, batch_size=128)

#scores = model.evaluate(data[2], data[3], batch_size=128)
#print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

#model.save('test_model10.h5')

model.fit(data[0], data[1], epochs=10, batch_size=128)

scores2 = model.evaluate(data[2], data[3], batch_size=128)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores2))

model.save('test_model20.h5')

model.fit(data[0], data[1], epochs=10, batch_size=128)

scores3 = model.evaluate(data[2], data[3], batch_size=128)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores3))

model.save('test_model30.h5')

exit(0)

# Choosing a test image for the tutorial:
image = data[2][7:8]

plot.imshow(image.squeeze(), cmap='gray', interpolation='nearest')
plot.show()

# Stripping the softmax activation from the model
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

analyzer = innvestigate.create_analyzer("lrp.z", model_wo_sm, )
analysis = analyzer.analyze(image)
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')

plot.show()