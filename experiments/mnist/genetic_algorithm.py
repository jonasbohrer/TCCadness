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

def load_models():

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

    methods = [ ("lrp.z",                           {},             'imgnetutils.heatmap',    "LRP-Z"),
                ("lrp.epsilon",                     {"epsilon": 1}, 'imgnetutils.heatmap',    "LRP-Epsilon"),
                ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, 'imgnetutils.heatmap',    "LRP-PresetAFlat"),
                ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, 'imgnetutils.heatmap',    "LRP-PresetBFlat")]
    method = methods[1]

    models_dir = 'models/'
    modelnames = []
    models = []
    for file_name in os.listdir(models_dir):
        if file_name.endswith('.h5'):
            modelnames.append(file_name)
    modelnames = sorted(modelnames, key=lambda x: (len(x), str.lower(x)))
    print (modelnames)

    for modelname in modelnames:
        print("Loading "+modelname)

        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(10, activation="softmax")])
        model.load_weights("models/"+modelname)

        print(model)
        models.append(model)

    return models

####################

initial_population = load_models()
population = copy.deepcopy(initial_population)
print(initial_population)
generations = 5

for generation in range (1, generations+1):
    
    """
    initial_population >> apply lrp to a set of images, save scores
    initial_population >> apply fitness function and sort according to score
    initial_population >> apply elitism, crossovers and mutations
    """

    print ("Generation {0} out of {1}".format(generation, generations))

    for individual in population:
        print(" Applying lrp to individual {0}".format(str(individual)))

        model_wo_sm = iutils.keras.graph.model_wo_softmax(individual)

        analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1])
        analysis = analyzer.analyze(image)
        #print (model.predict(image), model.predict_classes(image))

        #plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
        #plot.savefig("models/figs/fig"+str(i)+"_pred"+str(individual.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))
        analysis.

    print("  Sorting, conserving elite and applying crossovers to the rest")
