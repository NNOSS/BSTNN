from __future__ import print_function

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
import numpy as np
import matplotlib.pyplot as plt
import importImg
import saveMovie

class block:
    def __init__(self, name):
        self.name = name
        self.children = []

def define_block_body(x_image, block_info, reuse = False):
    '''Create a classifier from the given inputs'''
    m = block_info
    prefix = m.name + '_'
    with tf.variable_scope(prefix + "block") as scope:
        if reuse: #get previous variable if we are reusing the discriminator but with fake images
            scope.reuse_variables()
        # inputs = InputLayer(x_image, name=prefix + 'block_inputs')
        inputs = x_image
        conv_pointers = [inputs]#list of filters
        for i,v in enumerate(m.convolutions):
            if i < len(convolutions)-1:
                curr_layer = BatchNormLayer(Conv2d(conv_pointers[-1],
                    abs(convolutions[i+1]), (5, 5),strides = (1,1), name=prefix +
                    'conv1_%s'%(i)), act=tf.nn.leaky_relu,is_train=True ,name=prefix +
                    'batch_norm%s'%(i))

                conv_pointers.append(ConcatLayer([curr_layer, conv_pointers[-1]],
                 3, name =prefix + 'concat_layer%s'%(i)))
            else:
                # fully connected layer
                # l,w,d = m.input_size[0], m.input_size[1], sum(m.convolutions)
                flat = FlattenLayer(conv_pointers[-1], name = prefix + 'flatten')
                y_conv = DenseLayer(flat, m.fully_connected_size, name = prefix + 'hidden_encode')
        return y_conv, conv_pointers[-1]

def define_block(x_image, block_info, reuse_body = False, reuse = False):
    m = block_info
    hidden, lastLayer = define_block_body(x_image, block_info, reuse_body)

    y_conv = DenseLayer(flat, m.fully_connected_size, name = prefix + 'output')
    d_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = m.labels, logits = y_conv))# reduce mean for discriminator
    d_cross_entropy_summary = tf.summary.scalar(prefix + 'loss', d_cross_entropy)

    m.next_input = MaxPool2d(lastLayer, filter_size = (2,2))

    t_vars = tf.trainable_variables()
    prefix = m.name + '_'
    b_vars = [var for var in t_vars if prefix in var.name] #find trainable discriminator variable
    for var in b_vars:
        print(var.name)
    m.train_step = tf.train.AdamOptimizer(m.learning_rate, beta1=m.beta1).minimize(d_cross_entropy, var_list=b_vars)

def new_block(parent, index, list_classes):
    m = block(parent.name + index)
    m.classes = list_classes
    define_block(parent.next_input, m)

def generate_children(block_info):
    #TODO
    #get the confusion matrix
    matrix = get_confusion_matrix(block_info)
    #get the groups
    groups = return_groups(matrix, block_info.threshold)

def get_confusion_matrix(block_info):
    falsePercents = np.zeros((len(block_info.classes), len(block_info.classes)))
    totals = np.zeros((len(block_info.classes), len(block_info.classes)))
    #TODO

if __name__ == "__main__":
    paths = {}
    for i in range(NUM_CLASSES):
        paths.update(i,"0")