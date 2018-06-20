#Author: Nick Steelman
#Date: 6/11/18
#gns126@gmail.com
#cleanestmink.com

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import densenetBlock
import setParameters
import cPickle
import divide
import getData

NUM_CLASSES = 100
INPUT_SHAPE = 28, 28, 1

x = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2]))
m = densenetBlock.block('S', np.arange(NUM_CLASSES)+1, x, INPUT_SHAPE)
m.output
