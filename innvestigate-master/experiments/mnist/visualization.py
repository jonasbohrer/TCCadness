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
mnistutils = imp.load_source("utils_mnist", "../utils/utils_mnist.py")

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

methods = [ ("lrp.z",                           {},             'imgnetutils.heatmap',    "LRP-Z"),
            ("lrp.epsilon",                     {"epsilon": 1}, 'imgnetutils.heatmap',    "LRP-Epsilon"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, 'imgnetutils.heatmap',    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, 'imgnetutils.heatmap',    "LRP-PresetBFlat")]
method = methods[1]

models_dir = 'models/'
models = []
for file_name in os.listdir(models_dir):
    if file_name.endswith('.h5'):
        models.append(file_name)
models = sorted(models, key=lambda x: (len(x), str.lower(x)))
print (models)

images = [
    data[2][7:8],
    data[2][10:11],
    data[2][20:21],
    data[2][30:31],
    data[2][40:41],
    data[2][50:51],
    data[2][61:62],
    data[2][70:71],
    data[2][80:81],
    data[2][90:91]
]

# Choosing a test image for the relevance test:
i = 0
for image in images:
    plot.imshow(image.squeeze(), cmap='gray', interpolation='nearest')
    plot.savefig("models/figs/original"+str(i)+".png")

    # Generate images for every model checkpoint
    for modelname in models:
        print("Generating figs for "+modelname)
        model.load_weights("models/"+modelname)

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

        analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1])
        analysis = analyzer.analyze(image)
        #print (model.predict(image), model.predict_classes(image))

        plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
        plot.savefig("models/figs/fig"+str(i)+"_pred"+str(model.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))

    png_dir = 'models/figs/'
    files = []
    files = [imageio.imread(png_dir+"original"+str(i)+".png")]*5
    file_paths = []
    try:
        for file_name in os.listdir(png_dir):
            if file_name.endswith('.png') and file_name.startswith('fig'+str(i)):
                file_path = os.path.join(png_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x), str.lower(x)))
        for file_path in file_paths:
            files.append(imageio.imread(file_path))
        for n in range(1,10):
            files.append(files[-1])
        print('generated models/figs/movie'+str(i)+'.gif')
        imageio.mimsave('models/figs/movie'+str(i)+'.gif', files)
    except:
        pass
    i += 1

exit(0)

