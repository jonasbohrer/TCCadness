import warnings
warnings.simplefilter('ignore')

import imp
import matplotlib as mpl
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
import innvestigate.utils.visualizations as ivis 
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

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
            if file_name.endswith('.png') and file_name.startswith('fig'+str(i)+'_checkpoint'):
                file_path = os.path.join(figs_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x), str.lower(x)))
        for file_path in file_paths:
            files.append(imageio.imread(file_path))
        for n in range(1,10):
            files.append(files[-1])
        print('generated '+figs_dir+'movie'+str(i)+'.gif')
        imageio.mimsave(figs_dir+'movie'+str(i)+"_pred_"+str(model.predict_classes(image))+str(max(model.predict(image)[0]))+'.gif', files)
    except:
        pass

def join_gifs(figs_dir, models, method):

    movie = []
    for modelname in models:
        file_paths = []
        files = []
        for file_name in os.listdir(figs_dir):
            if file_name.endswith("].png") and file_name.startswith('fig') and (modelname.replace(".h5", "_") in file_name):
                file_path = os.path.join(figs_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x.split("_pred")[0]), str.lower(x)))
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
        max_size = 800,800
        img = Image.open(file_path)
        img.thumbnail(max_size)
        files.append(np.array(img))
    for n in range(1,10):
        files.append(files[-1])
    print('generated '+figs_dir+'movie.gif')
    imageio.mimsave(figs_dir+'movie.gif', files)
    imageio.mimsave(figs_dir+"/../../"+method[3]+'.gif', files)

def create_model():

    input_shape = (200, 200, 3)
    model = keras.models.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(16, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(16, activation="softmax"),
                ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(10,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer=keras.initializers.RandomUniform(),
        bias_initializer=keras.initializers.RandomUniform(),
        input_shape=(28,28,1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(10,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer=keras.initializers.RandomUniform(),
        bias_initializer=keras.initializers.RandomUniform()))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100,
        kernel_initializer=keras.initializers.RandomUniform(),
        bias_initializer=keras.initializers.RandomUniform(),
        activation='relu'))
    model.add(keras.layers.Dense(10,
        kernel_initializer=keras.initializers.RandomUniform(),
        bias_initializer=keras.initializers.RandomUniform(),
        activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(momentum=0.1, lr=0.005),
        metrics=['accuracy'])"""

    return model

def get_models():
    models = []
    for file_name in os.listdir(models_dir):
        if file_name.endswith('.h5'):
            models.append(file_name)
    models = sorted(models, key=lambda x: (len(x), str.lower(x)))
    return (models)

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
         }

blue_red1 = mpl.colors.LinearSegmentedColormap('BlueRed1', cdict1)

methods = [ ("lrp.z",                           {},             ivis.heatmap,    "LRP-Z"),
            ("lrp.epsilon",                     {"epsilon": 1}, ivis.heatmap,    "LRP-Epsilon"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat"),
            ("lrp.alpha_2_beta_1",              {},             ivis.heatmap,    "LRP-PresetAlpha2Beta1"),
            ("lrp.alpha_2_beta_1_IB",           {},             ivis.heatmap,    "LRP-PresetAlpha2Beta1IB"),
            ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat")]
            
modelfilename = sys.argv[1] if sys.argv[1] != None else "test_model"
method_n = int(sys.argv[2]) if sys.argv[2] != None else 6

method = methods[method_n]
print ("Using {}".format(method[3]))

output_nodes = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
analysis_mode = "all"
models_dir = 'models/'+modelfilename+'/'
figs_dir = models_dir+'figs/'+method[3]+'/'

model = create_model()
models = get_models()

if not os.path.exists(os.path.dirname(figs_dir)):
    try:
        os.makedirs(os.path.dirname(figs_dir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'LEGO brick images/valid/',
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary',
        seed= 1)

images = [
    validation_generator[0][0][28:29],#0
    validation_generator[0][0][8:9],#1
    validation_generator[1][0][8:9],#2
    validation_generator[0][0][4:5],#3
    validation_generator[0][0][13:14],#4
    validation_generator[0][0][3:4],#5
    validation_generator[0][0][24:25],#6
    validation_generator[0][0][14:15],#7
    validation_generator[0][0][6:7],#8
    validation_generator[1][0][28:29],#9
    validation_generator[0][0][9:10],#10
    validation_generator[0][0][1:2],#11
    validation_generator[1][0][25:26],#12
    validation_generator[0][0][25:26],#13
    validation_generator[0][0][0:1],#14
    validation_generator[0][0][12:13],#15
]

# Choosing a test image for the relevance test:
i = 0

"""for image in images:
    original = iutils.postprocess_images(image)
    #if analysis_mode == "all":
    #    original = concat_n_images([original]*len(output_nodes))
    Image.fromarray(np.uint8(original[0]*255)).save(figs_dir+'fig'+str(i)+"_original.png")

    # Generate images for every model checkpoint
    for modelname in models:
        print("Generating figs for "+modelname+", image "+str(i))

        try:
            model.load_weights(models_dir+modelname)
        except: 
            model = keras.models.load_model(models_dir+modelname)

        # Stripping the softmax activation from the model
        model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
        if analysis_mode == "all":
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
    i += 1"""

join_gifs(figs_dir, models, method)

exit(0)