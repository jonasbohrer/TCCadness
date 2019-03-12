from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.preprocessing import image
import keras
import numpy as np

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis 
from PIL import Image

import os

# Load the model
input_shape = (200, 200, 3)
model = keras.models.Sequential([
            keras.layers.Conv2D(48, (3, 3), activation="relu", input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(24, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(16, activation="softmax"),
            ])
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])
model.load_weights("C:/Users/User/Documents/GitHub/TCCadness/experiments/lego_bricks/models/model_4l/checkpoint_47.h5")
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

# Load the evalutation dataset
figs_dir = "models/model_4l/test_figs/"

if not os.path.exists(os.path.dirname(figs_dir)):
    try:
        os.makedirs(os.path.dirname(figs_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#img_path = 'images/valid/ball/Ball_2013_FIFA_Confederations_Cup.jpg'
img_name = '2pin.png'
img = image.load_img('test_images/'+img_name, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

images = [x]

# Create the analyzer
methods = [ ("lrp.z",                           {},             ivis.heatmap,    "LRP-Z"),
            ("lrp.epsilon",                     {"epsilon": 1}, ivis.heatmap,    "LRP-Epsilon"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat"),
            ("lrp.alpha_2_beta_1",              {},             ivis.heatmap,    "LRP-PresetAlpha2Beta1"),
            ("lrp.alpha_2_beta_1_IB",           {},             ivis.heatmap,    "LRP-PresetAlpha2Beta1IB"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat"),]
            
"""modelfilename = sys.argv[1] if sys.argv[1] != None else "test_model"
method_n = int(sys.argv[2]) if sys.argv[2] != None else 6"""

method = methods[2]

analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1], neuron_selection_mode="index")

output_nodes = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
# Prepare the analysis

for image in images:

    # Calculate labels
    predicted = model.predict_classes(image)
    print(predicted, model.predict(image), max(model.predict(image)[0]), model.predict_proba(image))

    for output_node in output_nodes:

        analysis = analyzer.analyze(image, output_node)
        processed_analysis = iutils.postprocess_images(analysis)
        processed_analysis = method[2](processed_analysis, cmap_type="seismic")
        print(str(output_node), sum(analysis.flatten()))
        #Image.fromarray(np.uint8(processed_analysis[0]*255)).save("{figs_dir}/{image_name}_{outputnode}.png".format(figs_dir=figs_dir, image_name = img_name, outputnode=output_node))

    """

    output_node = predicted

    analysis = analyzer.analyze(image, output_node)

    processed_analysis = iutils.postprocess_images(analysis)
    processed_analysis = method[2](processed_analysis, cmap_type="seismic")"""

