import imageio
import numpy as np
import os, sys, copy
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
            if file_name.endswith('.png') and file_name.startswith('fig'+str(i)+'_checkpoint'):
                file_path = os.path.join(figs_dir, file_name)
                file_paths.append(file_path)
        file_paths = sorted(file_paths, key=lambda x: (len(x.split("_pred")[0]), str.lower(x)))
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

def blue_red_colormap():
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
    return blue_red1
            