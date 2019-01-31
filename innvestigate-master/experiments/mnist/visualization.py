import warnings
warnings.simplefilter('ignore')

import imp
import matplotlib.pyplot as plot
import numpy as np
import os, copy, sys
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
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

methods = [ ("lrp.z",                           {},             mnistutils.heatmap,    "LRP-Z"),
            ("lrp.epsilon",                     {"epsilon": 1}, mnistutils.heatmap,    "LRP-Epsilon"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetBFlat"),
            ("lrp.alpha_2_beta_1",              {},             mnistutils.heatmap,    "LRP-PresetAlpha2Beta1"),
            ("lrp.alpha_2_beta_1_IB",           {},             mnistutils.heatmap,    "LRP-PresetAlpha2Beta1IB"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetBFlat")]
            
method_n = int(sys.argv[1]) if sys.argv[1] != None else 6
analysis_mode = "all"

method = methods[method_n]
print ("Using {}".format(method[3]))

models_dir = 'models/'
figs_dir = models_dir+'figs/'+method[3]+'/'

if not os.path.exists(os.path.dirname(figs_dir)):
    try:
        os.makedirs(os.path.dirname(figs_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

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
    plot.savefig(figs_dir+"/original"+str(i)+".png")

    # Generate images for every model checkpoint
    for modelname in models:
        if analysis_mode == "all":
            print("Generating figs for "+modelname)
            model.load_weights(models_dir+modelname)

            # Stripping the softmax activation from the model
            model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

            analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1], neuron_selection_mode="index")

            for output_node in [0,1,2,3,4,5,6,7,8,9]:
                analysis = analyzer.analyze(image, output_node)
                processed_analysis = mnistutils.postprocess(analysis)
                processed_analysis = method[2](processed_analysis)
                #print (model.predict(image), model.predict_classes(image))
                plot.imshow(processed_analysis.squeeze(), cmap='seismic', interpolation='nearest')      
                plot.savefig(figs_dir+'fig'+str(i)+"_rel"+str(output_node)+"_pred"+str(model.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))
            exit(0)
        else:
            print("Generating figs for "+modelname)
            model.load_weights(models_dir+modelname)

            # Stripping the softmax activation from the model
            model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

            analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1])
            analysis = analyzer.analyze(image)
            processed_analysis = mnistutils.postprocess(analysis)
            processed_analysis = method[2](processed_analysis)
            #print (model.predict(image), model.predict_classes(image))
            plot.imshow(processed_analysis.squeeze(), cmap='seismic', interpolation='nearest')      
            plot.savefig(figs_dir+'fig'+str(i)+"_pred"+str(model.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))

    files = []
    files = [imageio.imread(figs_dir+"original"+str(i)+".png")]*5
    file_paths = []
    print (os.listdir(figs_dir))
    try:
        for file_name in os.listdir(figs_dir):
            if file_name.endswith('.png') and file_name.startswith('fig'+str(i)):
                file_path = os.path.join(figs_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x), str.lower(x)))
        for file_path in file_paths:
            files.append(imageio.imread(file_path))
        for n in range(1,10):
            files.append(files[-1])
        print('generated '+figs_dir+'movie'+str(i)+'.gif')
        imageio.mimsave(figs_dir+'movie'+str(i)+'.gif', files)
    except:
        pass
    i += 1

exit(0)

