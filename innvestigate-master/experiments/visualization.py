import warnings
warnings.simplefilter('ignore')

import imp
import matplotlib.pyplot as plot
import numpy as np
import os, copy
import imageio

import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils

import innvestigate
import innvestigate.utils as iutils

# Use utility libraries to focus on relevant iNNvestigate routines.
mnistutils = imp.load_source("utils_mnist", "utils/utils_mnist.py")

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

models = [
'test_model1.h5',
'test_model2.h5',
'test_model3.h5',
'test_model4.h5',
'test_model5.h5',
'test_model6.h5',
'test_model7.h5',
'test_model8.h5',
'test_model9.h5',
'test_model10.h5',
]

images = [
    data[2][7:8],
    data[2][8:9],
    data[2][9:10]
]

# Choosing a test image for the relevance test:
i = 0
for image in images:
    plot.imshow(image.squeeze(), cmap='gray', interpolation='nearest')
    plot.savefig("models/figs/original"+str(i)+".png")

    # Generate images for every model checkpoint
    for modelname in models:
        model.load_weights("models/"+modelname)

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

        analyzer = innvestigate.create_analyzer("lrp.z", model_wo_sm, )
        analysis = analyzer.analyze(image)
        plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
        plot.savefig("models/figs/fig"+str(i)+"_"+modelname.replace(".h5", ".png"))

    png_dir = 'models/figs/'
    files = []
    try:
        for file_name in os.listdir(png_dir):
            if file_name.endswith('.png') and file_name.startswith('fig'+str(i)):
                print (file_name)
                file_path = os.path.join(png_dir, file_name)
                files.append(imageio.imread(file_path))
        
        imageio.mimsave('models/figs/movie'+str(i)+'.gif', files)
    except:
        pass

    i += 1

exit(0)

