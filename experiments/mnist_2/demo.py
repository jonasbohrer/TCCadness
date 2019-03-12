from __future__ import print_function

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from devol import DEvol, GenomeHandler

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

K.set_image_data_format("channels_last")

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

train_size = 60000
test_size = 600

x_train = x_train[0:train_size]
y_train = y_train[0:train_size]
x_test = x_test[0:test_size]
y_test = y_test[0:test_size]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

genome_handler = GenomeHandler(max_conv_layers=4, 
                               max_dense_layers=2, # includes final dense layer
                               max_filters=64,
                               max_dense_nodes=128,
                               input_shape=x_train.shape[1:],
                               n_classes=10)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.
# The best model is returned decoded and with `epochs` training done.

devol = DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=20,
                  pop_size=5,
                  epochs=5,
                  train_size=1500,
                  replace=0.4)

print(model.summary())
