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
import numpy as np

class block:
    def __init__(self, name, labels, x, y, predictions, children_groups = []):
        #Initialized Immediately
        self.name = name
        self.labels = labels #TRUE PREDICTIONS STARTING AT 0
        self.x = x
        self.y = y
        self.predictions = predictions
        self.children_groups = children_groups #TRUE PREDICTIONS STARTING AT 1
        #Should be Initialized before model initialization

        self.block_labels = None
        self.learning_rate = None
        self.beta1 = None
        self.convolutions = None
        self.fully_connected_size = None
        #Initialized at model initialization

        self.perm_variables = None #Needed
        self.temp_variables = None #Needed
        self.y_ = None
        self.y_conv = None
        self.convolutions = None
        self.filtered_labels = None
        self.filtered_labels_false = None
        self.filtered_input = None
        self.cross_entropy_summary = None #Needed
        self.accuracy_summary_train = None #Needed
        self.output = None
        self.train_step = None #Needed
        #initializated when creating new models
        self.children = []

def clean_block_recursive(block_info):
    block_info.x = None
    block_info.y = None
    block_info.predictions = None
    block_info.perm_variables = None #Needed
    block_info.temp_variables = None #Needed
    block_info.y_ = None
    block_info.y_conv = None
    block_info.filtered_labels = None
    block_info.filtered_labels_false = None
    block_info.filtered_input = None
    block_info.cross_entropy_summary = None #Needed
    block_info.accuracy_summary_train = None #Needed
    block_info.accuracy_summary_test = None #Needed
    block_info.num_right = None
    block_info.output = None
    block_info.train_step = None #Needed
    block_info.arbitrary_prediction = None
    for child in block_info.children:
        if child is not None:
            clean_block_recursive(child)


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


def define_block_leaf(block_info, test_bool):
    '''Handle the final fully connected layer of the block as well as the necessary
    variables to return'''
    m = block_info
    prefix = m.name + '_'
    with tf.variable_scope(prefix + "block") as scope:
        block_labels = tf.constant(m.block_labels, name = prefix +'block_labels', dtype = tf.int64)
        keep_instances_p = tf.gather(block_labels, m.predictions)

        keep_instances_l =  tf.cond(test_bool, lambda: keep_instances_p, lambda: tf.gather(block_labels, m.y), name = prefix + 'test_input')


        mask = tf.greater(keep_instances_p + keep_instances_l, tf.zeros_like(keep_instances_p))
        m.filtered_labels = tf.boolean_mask(m.y, mask, name = prefix + 'filtered_labels')
        m.filtered_labels_false = tf.boolean_mask(tf.gather(block_labels, m.y), mask, name = prefix + 'filtered_labels_false')
        one_hot_false = tf.one_hot(m.filtered_labels_false, len(m.labels) + 1, name=prefix +'One_Hot')
        m.filtered_input = tf.boolean_mask(m.x, mask,name = prefix + 'filtered_input')

        tl_input = InputLayer(m.filtered_input, name= prefix +'tl_inputs')
        lastLayer = define_block_body(tl_input, m)
        flat = FlattenLayer(lastLayer, name = prefix + 'flatten')
        t_vars = tf.trainable_variables()
        # print(t_vars)
        m_vars_names = [var for var in t_vars if prefix in var.name]
        # print(m_vars_names)
        m.perm_variables = m_vars_names
        dense_layer = DenseLayer(flat, len(m.labels)+1, name = prefix + 'output')
        m.y_conv = dense_layer.outputs

        t_vars = tf.trainable_variables()
        m_vars = [var for var in t_vars if prefix + 'output' in var.name] #find trainable discriminator variable
        m.temp_variables = m_vars

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_false, logits = m.y_conv))# reduce mean for discriminator
        m.cross_entropy_summary = tf.summary.scalar(prefix + 'loss_leaf', cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(m.y_conv,1), tf.argmax(one_hot_false,1)), tf.float32))
        m.accuracy_summary_train = tf.summary.scalar(prefix + 'accuracy_train_leaf', accuracy)
        m.accuracy_summary_test = tf.summary.scalar(prefix + 'accuracy_test_leaf', accuracy)

        mask = tf.gather(m.terminal_points, m.filtered_labels_false)
        teminal_labels = tf.boolean_mask(m.filtered_labels_false, mask)
        teminal_predictions = tf.boolean_mask(tf.argmax(m.y_conv,1), mask)
        m.num_right = tf.cast(tf.equal(teminal_labels, teminal_predictions), tf.float32)

        m.output = MaxPool2d(lastLayer, filter_size = (2,2)).outputs

        t_vars = tf.trainable_variables()
        m_vars = [var for var in t_vars if prefix in var.name] #find trainable discriminator variabl
        m.train_step = tf.train.AdamOptimizer(m.learning_rate, beta1=m.beta1).minimize(cross_entropy)

def define_block_branch(block_info, test_bool,reuse = None):
    '''Handle the final fully connected layer of the block as well as the necessary
    variables to return'''
    m = block_info
    prefix = m.name + '_'
    with tf.variable_scope(prefix + "block", reuse = reuse) as scope:
        block_labels = tf.constant(m.block_labels, name = prefix +'block_labels', dtype = tf.int64)
        keep_instances_p = tf.gather(block_labels, m.predictions)
        keep_instances_l =  tf.cond(test_bool, lambda: keep_instances_p, lambda: tf.gather(block_labels, m.y), name = prefix + 'test_input')

        mask = tf.greater(keep_instances_p + keep_instances_l, tf.zeros_like(keep_instances_p))
        m.filtered_labels = tf.boolean_mask(m.y, mask, name = prefix + 'filtered_labels')
        m.filtered_labels_false = tf.boolean_mask(tf.gather(block_labels, m.y), mask, name = prefix + 'filtered_labels_false')
        one_hot_false = tf.one_hot(m.filtered_labels_false, len(m.children_groups) + 1, name=prefix +'One_Hot')
        m.filtered_input = tf.boolean_mask(m.x, mask,name = prefix + 'filtered_input')

        tl_input = InputLayer(m.filtered_input, name= prefix +'tl_inputs')
        lastLayer = define_block_body(tl_input, m)
        flat = FlattenLayer(lastLayer, name = prefix + 'flatten')
        m.y_conv = DenseLayer(flat, len(m.children_groups)+1, name = prefix + 'branch_output').outputs
        class_choice = tf.argmax(m.y_conv, 1)
        array = np.arange(len(m.children_groups)+1)
        for i in range(1, len(m.children_groups)+1):
            index = np.argmax(np.equal(i, m.block_labels), axis = 0)
            array[i] = index
        reverse = tf.constant(array, name=prefix + 'reverse_conversion',dtype = tf.int32)
        m.arbitrary_prediction = tf.gather(reverse, class_choice)

        t_vars = tf.trainable_variables()
        # print(t_vars)
        m_vars_names = [var for var in t_vars if prefix in var.name and prefix + 'output' not in var.name]
        # print(m_vars_names)
        m.perm_variables = m_vars_names
        m.temp_variables = []
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_false, logits = m.y_conv))# reduce mean for discriminator
        m.cross_entropy_summary = tf.summary.scalar(prefix + 'loss_branch', cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(m.y_conv,1), tf.argmax(one_hot_false,1)), tf.float32))
        m.accuracy_summary_train = tf.summary.scalar(prefix + 'accuracy_train_branch', accuracy)
        m.accuracy_summary_test = tf.summary.scalar(prefix + 'accuracy_test_branch', accuracy)
        mask = tf.gather(m.terminal_points, m.filtered_labels_false)
        teminal_labels = tf.boolean_mask(m.filtered_labels_false, mask)
        teminal_predictions = tf.boolean_mask(class_choice, mask)
        m.num_right = tf.cast(tf.equal(teminal_labels, teminal_predictions), tf.float32)


        m.output = MaxPool2d(lastLayer, filter_size = (2,2)).outputs
        # print(m.output.get_shape())
        m.train_step = tf.train.AdamOptimizer(m.learning_rate, beta1=m.beta1).minimize(cross_entropy)
