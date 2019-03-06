from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.preprocessing import image
import numpy as np

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis 
from PIL import Image

# Load the model
model = VGG16(weights='imagenet')
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

# Load the evalutation dataset
img_path = 'images/valid/ball/Ball_2013_FIFA_Confederations_Cup.jpg'
img = image.load_img(img_path, target_size=(224, 224))
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
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat")]
            
"""modelfilename = sys.argv[1] if sys.argv[1] != None else "test_model"
method_n = int(sys.argv[2]) if sys.argv[2] != None else 6"""

method = methods[6]

analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1], neuron_selection_mode="index")

# Prepare the analysis
for image in images:

    # Calculate labels
    predicted = model.predict(image)
    labels = decode_predictions(predicted)[0]
    output_nodes = predicted.flatten().argsort()[-len(labels):]

    for n in range(0, len(labels)):
        label = labels[n]
        output_node = output_nodes[n]

        analysis = analyzer.analyze(image, output_node)
        processed_analysis = iutils.postprocess_images(analysis)
        processed_analysis = method[2](processed_analysis, cmap_type="seismic")
        print(str(output_node)+label[1])
        Image.fromarray(processed_analysis.flatten()*255).save(str(output_node)+label[1]+".png")