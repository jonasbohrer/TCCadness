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
from PIL import Image

def join_images(images):
    #images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
        
    return new_im

def concat_images(imga, imgb, orientation='horizontal'):
    """
    Combines two color image ndarrays side-by-side.
    """
    if orientation == 'horizontal':
        ha,wa = imga.shape[1:3]
        hb,wb = imgb.shape[1:3]
        max_height = np.max([ha, hb])
        total_width = wa+wb
        new_img = np.zeros(shape=(1, max_height, total_width, 3))
        new_img[0,:ha,:wa]=imga
        new_img[0,:hb,wa:wa+wb]=imgb
        return new_img
    elif orientation == 'vertical':
        ha,wa = imga.shape[1:3]
        hb,wb = imgb.shape[1:3]
        max_width = np.max([wa, wb])
        total_height = ha+hb
        new_img = np.zeros(shape=(1, total_height, max_width, 3))
        new_img[0,:ha,:wa]=imga
        new_img[0,:hb,wa:wa+wb]=imgb
        return new_img
    else:
        return None

def concat_n_images(images, orientation='horizontal'):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img in enumerate(images):
        if i==0:
            output = img
        else:
            output = concat_images(output, img, orientation)
    return output

def generate_gifs(i, figs_dir, model, image):

    files = []
    #files = [imageio.imread(figs_dir+'fig'+str(i)+"_original.png")]*5
    file_paths = []
    #print (os.listdir(figs_dir))
    try:
        for file_name in os.listdir(figs_dir):
            if file_name.endswith('.png') and file_name.startswith('fig'+str(i)+'_pred'):
                file_path = os.path.join(figs_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x), str.lower(x)))
        for file_path in file_paths:
            files.append(imageio.imread(file_path))
        for n in range(1,10):
            files.append(files[-1])
        print('generated '+figs_dir+'movie'+str(i)+'.gif')
        imageio.mimsave(figs_dir+'movie'+str(i)+"_pred_"+str(model.predict_classes(image))+'_'+str(max(model.predict(image)[0]))+'.gif', files)
    except:
        pass

def join_gifs(figs_dir, models, method):

    movie = []
    for modelname in models:
        file_paths = []
        files = []
        for file_name in os.listdir(figs_dir):
            if file_name.endswith(modelname.replace(".h5", ".png")) and file_name.startswith('fig'):
                file_path = os.path.join(figs_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x), str.lower(x)))
        for file_path in file_paths:
            files.append(Image.open(file_path))
        join_images(files).save(figs_dir+'movie_'+modelname.replace(".h5", ".png"))

    file_paths = []
    files = []
    for file_name in os.listdir(figs_dir):
        if file_name.endswith(".png") and file_name.startswith('movie_'):
            file_path = os.path.join(figs_dir, file_name)
            file_paths.append(file_path)
    file_paths = sorted(file_paths, key=lambda x: (len(x), str.lower(x)))
    for file_path in file_paths:
        files.append(imageio.imread(file_path))
    print(len(files))
    for n in range(1,10):
        files.append(files[-1])
    print('generated '+figs_dir+'movie.gif')
    imageio.mimsave(figs_dir+'movie.gif', files)
    imageio.mimsave(figs_dir+"/../"+method[3]+'.gif', files)

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

methods = [ ("lrp.z",                           {},             mnistutils.heatmap,    "LRP-Z"),
            ("lrp.epsilon",                     {"epsilon": 1}, mnistutils.heatmap,    "LRP-Epsilon"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetBFlat"),
            ("lrp.alpha_2_beta_1",              {},             mnistutils.heatmap,    "LRP-PresetAlpha2Beta1"),
            ("lrp.alpha_2_beta_1_IB",           {},             mnistutils.heatmap,    "LRP-PresetAlpha2Beta1IB"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, mnistutils.heatmap,    "LRP-PresetBFlat")]
            
method_n = int(sys.argv[1]) if sys.argv[1] != None else 6

method = methods[method_n]
print ("Using {}".format(method[3]))

output_nodes = [0,1,2,3,4,5,6,7,8,9]
analysis_mode = "all"
models_dir = 'models/'
figs_dir = models_dir+'figs/'+method[3]+'/'

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(10,
    kernel_size=(3, 3),
    activation='relu',
    kernel_initializer=keras.initializers.RandomUniform(),
    bias_initializer=keras.initializers.RandomUniform(),
    input_shape=(28,28,1)))

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(10,
    kernel_initializer=keras.initializers.RandomUniform(),
    bias_initializer=keras.initializers.RandomUniform(),
    activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(momentum=0.1, lr=0.005),
    metrics=['accuracy'])

models = []
for file_name in os.listdir(models_dir):
    if file_name.endswith('.h5'):
        models.append(file_name)
models = sorted(models, key=lambda x: (len(x), str.lower(x)))
print (models)

if not os.path.exists(os.path.dirname(figs_dir)):
    try:
        os.makedirs(os.path.dirname(figs_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

images = [
    data[2][10:11], #0
    data[2][40:41], #1
    data[2][43:44], #2
    data[2][30:31], #3
    #data[2][90:91], #3
    data[2][42:43], #4
    data[2][45:46], #5
    data[2][50:51], #6
    data[2][70:71], #7
    #data[2][80:81], #7
    data[2][61:62], #8
    data[2][20:21], #9
    #data[2][7:8], #9
]

# Choosing a test image for the relevance test:
i = 0

for image in images:
    original = mnistutils.bk_proj(mnistutils.postprocess(image))
    #if analysis_mode == "all":
    #    original = concat_n_images([original]*len(output_nodes))
    Image.fromarray(np.uint8(original[0]*255)).save(figs_dir+'fig'+str(i)+"_original.png")

    # Generate images for every model checkpoint
    for modelname in models:
        print("Generating figs for "+modelname)

        try:
            model.load_weights(models_dir+modelname)
        except: 
            model = keras.models.load_model(models_dir+modelname)

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
        if analysis_mode == "all":
            analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1], neuron_selection_mode="index")
            lrp_images = [original]
            for output_node in output_nodes:
                analysis = analyzer.analyze(image, output_node)
                processed_analysis = mnistutils.postprocess(analysis)
                processed_analysis = method[2](processed_analysis)
                lrp_images.append(processed_analysis)
                #print (model.predict(image), model.predict_classes(image))
                #plot.imshow(processed_analysis.squeeze(), cmap='seismic', interpolation='nearest')      
                #plot.savefig(figs_dir+'fig'+str(i)+"_rel"+str(output_node)+"_pred"+str(model.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))
        else:
            analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1])
            analysis = analyzer.analyze(image)
            processed_analysis = mnistutils.postprocess(analysis)
            processed_analysis = method[2](processed_analysis)
            #print (model.predict(image), model.predict_classes(image))
            #plot.imshow(processed_analysis.squeeze(), cmap='seismic', interpolation='nearest')
            #plot.savefig(figs_dir+'fig'+str(i)+"_pred"+str(model.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))
            lrp_images = [original, processed_analysis]
        imgs = concat_n_images(lrp_images)
        Image.fromarray(np.uint8(imgs[0]*255)).save(figs_dir+'fig'+str(i)+"_pred"+str(model.predict_classes(image))+"_"+modelname.replace(".h5", ".png"))

    generate_gifs(i, figs_dir, model, image)
    i += 1

join_gifs(figs_dir, models, method)

exit(0)

