#Author: Nick Steelman
#Date: 5/29/18
#gns126@gmail.com
#cleanestmink.com

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

class block:
    def __init__(self, name, labels, input_shape):
        #Initialized Immediately
        self.name = name
        self.labels = labels #TRUE PREDICTIONS STARTING AT 0
        self.input_shape = input_shape
        self.input = tf.placeholder(tf.float32, shape=(None, input_shape[0], input_shape[1], input_shape[2]))

        #Should be Initialized before model initialization
        self.block_labels = None
        self.learning_rate = None
        self.beta1 = None
        self.convolutions = None
        self.fully_connected_size = None
        self.input_shape = None
        #Initialized at model initialization
        self.output_shape = None
        self.output = None
        self.output_shape = None
        self.train_step = None
        #initializated when creating new models
        self.children = [] #TRUE PREDICTIONS STARTING AT 1

def define_block_body(x_image, block_info, reuse = False):
    '''Create a classifier from the given inputs'''
    m = block_info
    prefix = m.name + '_'
    with tf.variable_scope(prefix + "block") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        inputs = x_image
        conv_pointers = [inputs]#list of filters
        for i,v in enumerate(m.convolutions):
            curr_layer = BatchNormLayer(Conv2d(conv_pointers[-1],
                m.convolutions[i], (5, 5),strides = (1,1), name=prefix +
                'conv1_%s'%(i)), act=tf.nn.leaky_relu,is_train=True ,name=prefix +
                'batch_norm%s'%(i))
            if i < len(m.convolutions)-1:
                conv_pointers.append(ConcatLayer([curr_layer, conv_pointers[-1]],
                 3, name =prefix + 'concat_layer%s'%(i)))
            else:
                conv_pointers.append(curr_layer)

        flat = FlattenLayer(conv_pointers[-1], name = prefix + 'flatten')
        y_conv = DenseLayer(flat, m.fully_connected_size, act=tf.nn.relu,name = prefix + 'hidden_encode')

        return y_conv, conv_pointers[-1]

def define_block(block_info, reuse_body = False, reuse = False):
    '''Handle the final fully connected layer of the block as well as the necessary
    variables to return'''
    m = block_info
    prefix = m.name + '_'
    tl_input = InputLayer(m.input, name= prefix +'tl_inputs')
    hidden, lastLayer = define_block_body(tl_input, m, reuse_body)
    with tf.variable_scope(prefix + "block") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        m.y_conv = DenseLayer(hidden, len(m.labels), name = prefix + 'output').outputs
        m.y = tf.placeholder(tf.float32, shape=[None, len(m.labels)], name= prefix +'class_inputs') #correct class

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = m.y, logits = m.y_conv))# reduce mean for discriminator
        m.cross_entropy_summary = tf.summary.scalar(prefix + 'loss', cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(m.y_conv,1), tf.argmax(m.y,1)), tf.float32))
        m.accuracy_summary = tf.summary.scalar(prefix + 'accuracy', accuracy)

        m.output = MaxPool2d(lastLayer, filter_size = (2,2)).outputs
        print(m.output.get_shape())
        _, l, w, d = m.output.get_shape()
        m.output_shape = (l, w, d)
        t_vars = tf.trainable_variables()
        b_vars = [var for var in t_vars if prefix in var.name] #find trainable discriminator variable
        for var in b_vars:
            print(var.name)
        m.train_step = tf.train.AdamOptimizer(m.learning_rate, beta1=m.beta1).minimize(cross_entropy, var_list=b_vars)
