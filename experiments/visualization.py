import warnings
warnings.simplefilter('ignore')

import imp
import matplotlib as mpl
import matplotlib.pyplot as plot
import numpy as np
import os, copy, sys
import collections

import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis 
from PIL import Image

from config import Config, Model

modelfilename = sys.argv[1] if sys.argv[1] != None else "test_model"
method_n = int(sys.argv[2]) if sys.argv[2] != None else 6

config = Config(modelfilename, method_n)

method = config.method
print ("Using {}".format(method[3]))

"""
Create figs directory
"""
if not os.path.exists(os.path.dirname(config.figs_dir)):
    try:
        os.makedirs(os.path.dirname(config.figs_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

"""
Load the models and configs
"""
models = Model.get_model_checkpoints()
images = config.get_lrp_image_set()

i = 0
for image in images:
    original = iutils.postprocess_images(image)
    Image.fromarray(np.uint8(original[0]*255)).save(figs_dir+'fig'+str(i)+"_original.png")

    for modelname in models:
        print("Generating figs for "+modelname+", image "+str(i))
        model = models[modelname]

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
        if config.analysis_mode == "all":
            analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1], neuron_selection_mode="index")
            lrp_images = [original]
            predicted = model.predict_classes(image)

            #Generate heatmaps for every class
            for output_node in output_nodes:
                analysis = analyzer.analyze(image, output_node)
                processed_analysis = iutils.postprocess_images(analysis)
                processed_analysis = method[2](processed_analysis, cmap_type="seismic")

                #Draw a box to identify the predicted class
                if output_node == predicted:
                    processed_analysis[0][::2, 0:3, 1] = 0
                    processed_analysis[0][0:3, ::2, 1] = 0
                    processed_analysis[0][::2, -4:-1, 1] = 0
                    processed_analysis[0][-4:-1, ::2, 1] = 0

                lrp_images.append(processed_analysis)
        else:
            analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1])
            analysis = analyzer.analyze(image)
            processed_analysis = iutils.postprocess_images(analysis)
            processed_analysis = method[2](processed_analysis)

            lrp_images = [original, processed_analysis]
        
        #Join images and save an overview of the predictions
        imgs = concat_n_images(lrp_images)
        Image.fromarray(np.uint8(imgs[0]*255)).save(figs_dir+'fig'+str(i)+"_"+modelname.replace(".h5", "")+"_pred"+str(model.predict_classes(image))+'.png')

    generate_gifs(i, figs_dir, model, image)
    i += 1

join_gifs(figs_dir, models, method)

exit(0)