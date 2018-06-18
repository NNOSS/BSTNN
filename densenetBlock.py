#Author: Nick Steelman
#Date: 5/29/18
#gns126@gmail.com
#cleanestmink.com

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorlayer.layers import *

class block:
    def __init__(self, name, labels, x, y, predictions):
        #Initialized Immediately
        self.name = name
        self.labels = labels #TRUE PREDICTIONS STARTING AT 0
        self.x = x
        self.y = y
        self.predictions = predictions
        #Should be Initialized before model initialization

        self.block_labels = None
        self.learning_rate = None
        self.beta1 = None
        self.convolutions = None
        self.fully_connected_size = None
        #Initialized at model initialization
        self.output = None
        self.train_step = None
        self.perm_variables = None
        self.temp_variables = None
        self.y_ = None
        self.y_conv = None
        #initializated when creating new models
        self.children = [] #TRUE PREDICTIONS STARTING AT 1

def define_block_body(x_image,block_info):
    '''Create a classifier from the given inputs'''
    m = block_info
    prefix = m.name + '_'
    # if reuse: #get previous variable if we are reusing the discriminator but with fake images
    #     scope.reuse_variables()
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
    # y_conv = DenseLayer(flat, m.fully_connected_size, act=tf.nn.relu,name = prefix + 'hidden_encode')
    return conv_pointers[-1]


def define_block_leaf(block_info):
    '''Handle the final fully connected layer of the block as well as the necessary
    variables to return'''
    m = block_info
    prefix = m.name + '_'
    with tf.variable_scope(prefix + "block") as scope:
        block_labels = tf.constant(m.block_labels, name = prefix +'block_labels')
        m.filtered_labels = tf.where(block_labels[m.predictions] != 0 or block_labels[m.y] != 0,
        x = m.y, name = prefix + 'filtered_labels')
        m.filtered_labels_false = block_labels[m.y]
        one_hot_false = tf.zeros([m.filtered_labels.get_shape()[0], len(m.labels)+1], name=prefix +'One_Hot')
        one_hot_false[m.filtered_labels_false] = 1
        m.filtered_input = tf.where(block_labels[m.predictions] != 0 or block_labels[m.y] != 0,
        x = m.x, name = prefix + 'filtered_input')

        tl_input = InputLayer(m.filtered_input, name= prefix +'tl_inputs')
        lastLayer = define_block_body(tl_input, m)
        flat = FlattenLayer(lastLayer, name = prefix + 'flatten')
        t_vars = tf.trainable_variables()
        print(t_vars)
        m_vars_names = [var.name for var in t_vars if prefix in var.name]
        print(m_vars_names)
        m.perm_variables = m_vars_names
        m.y_conv = DenseLayer(flat, len(m.labels)+1, name = prefix + 'output').outputs
        m.temp_variables = [m.y_conv.name]

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_false, logits = m.y_conv))# reduce mean for discriminator
        m.cross_entropy_summary = tf.summary.scalar(prefix + 'loss_leaf', cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(m.y_conv,1), tf.argmax(one_hot_false,1)), tf.float32))
        m.accuracy_summary_train = tf.summary.scalar(prefix + 'accuracy_train_leaf', accuracy)

        m.output = MaxPool2d(lastLayer, filter_size = (2,2)).outputs
        print(m.output.get_shape())
        _, l, w, d = m.output.get_shape()
        m.output_shape = (l, w, d)

        t_vars = tf.trainable_variables()s
        m_vars = [var for var in t_vars if prefix in var.name] #find trainable discriminator variable
        for var in b_vars:
            print(var.name)
        m.train_step = tf.train.AdamOptimizer(m.learning_rate, beta1=m.beta1).minimize(cross_entropy, var_list=m_vars)

def define_block_branch(block_info):
    '''Handle the final fully connected layer of the block as well as the necessary
    variables to return'''
    m = block_info
    prefix = m.name + '_'
    with tf.variable_scope(prefix + "block") as scope:
        block_labels = tf.constant(m.block_labels, name = prefix +'block_labels')
        m.filtered_labels = tf.where(block_labels[m.predictions] != 0 or block_labels[m.y] != 0,
        x = m.y, name = prefix + 'filtered_labels')
        filtered_labels_false = block_labels[m.y]
        one_hot_false = tf.zeros([filtered_labels.get_shape()[0], len(m.children)+1], name=prefix +'One_Hot')
        one_hot_false[filtered_labels_false] = 1
        m.filtered_input = tf.where(block_labels[m.predictions] != 0 or block_labels[m.y] != 0,
        x = m.x, name = prefix + 'filtered_input')
        tl_input = InputLayer(filtered_input, name= prefix +'tl_inputs')
        lastLayer = define_block_body(tl_input, m)
        flat = FlattenLayer(conv_pointers[-1], name = prefix + 'flatten')
        m.y_conv = DenseLayer(flat, len(m.children)+1, name = prefix + 'output').outputs

        t_vars = tf.trainable_variables()
        print(t_vars)
        m_vars_names = [var.name for var in t_vars if prefix in var.name]
        print(m_vars_names)
        m.perm_variables = m_vars_names

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_false, logits = m.y_conv))# reduce mean for discriminator
        m.cross_entropy_summary = tf.summary.scalar(prefix + 'loss_branch', cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(m.y_conv,1), tf.argmax(one_hot_false,1)), tf.float32))
        m.accuracy_summary_train = tf.summary.scalar(prefix + 'accuracy_train_branch', accuracy)

        m.output = MaxPool2d(lastLayer, filter_size = (2,2)).outputs
        print(m.output.get_shape())
        _, l, w, d = m.output.get_shape()
        m.output_shape = (l, w, d)

        t_vars = tf.trainable_variables()s
        m_vars = [var for var in t_vars if prefix in var.name] #find trainable discriminator variable
        for var in b_vars:
            print(var.name)
        m.train_step = tf.train.AdamOptimizer(m.learning_rate, beta1=m.beta1).minimize(cross_entropy, var_list=m_vars)
