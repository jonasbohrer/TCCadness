'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin 
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import input_data

import tensorflow as tf
import numpy as np
import pdb
import scipy.io as sio

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 200,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 10,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 20,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_convolutional_logs','Summaries directory')
flags.DEFINE_boolean("relevance", True,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/epsilon/ww/flat/alphabeta')
flags.DEFINE_boolean("save_model", True,'Save the trained model')
flags.DEFINE_boolean("reload_model", True,'Restore the trained model')
#flags.DEFINE_string("checkpoint_dir", 'mnist_convolution_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_dir", 'mnist_convolutional_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", 'mnist_convolutional_model','Checkpoint dir')

FLAGS = flags.FLAGS

def nn():
    
    return Sequential([Convolution(output_depth=10,input_depth=1,batch_size=FLAGS.batch_size, input_dim=28, act ='relu', stride_size=1, pad='VALID'),
                       AvgPool(),

                       Convolution(output_depth=25,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=4,output_depth=100,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                       #Softmax()
    ])


def feed_dict(mnist, train):
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        k = 1.0
    return (2*xs)-1, ys, k


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  standard_d = feed_dict(mnist, False)

  config = tf.ConfigProto(allow_soft_placement = True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    
    with tf.variable_scope('model'):
        net = nn()
        inp = tf.pad(tf.reshape(x, [FLAGS.batch_size,28,28,1]), [[0,0],[2,2],[2,2],[0,0]])
        op = net.forward(inp)
        y = tf.squeeze(op)
        trainer = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])
    with tf.variable_scope('relevance'):
        if FLAGS.relevance:
            LRP = net.lrp(op, FLAGS.relevance_method, 1e-3)

            # LRP layerwise 
            relevance_layerwise = []
            R = op
            for layer in net.modules[::-1]:
                R = net.lrp_layerwise(layer, R, FLAGS.relevance_method, 1e-3)
                relevance_layerwise.append(R)

        else:
            LRP=[]
            relevance_layerwise = []
            
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    
    utils = Utils(sess, FLAGS.checkpoint_reload_dir)
    ''' Reload from a list of numpy arrays '''
    if FLAGS.reload_model:
        tvars = tf.trainable_variables()
        npy_files = np.load('mnist_convolutional_model/model.npy', encoding='bytes')
        [sess.run(tv.assign(npy_files[tt])) for tt,tv in enumerate(tvars)]
        print('aaaa', sess, net, tf.variable_scope('model'))
        utils.reload_model()

        print('bbbb', sess, net, tf.variable_scope('model'))

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
