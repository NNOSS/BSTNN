#Author: Nick Steelman
#Date: 6/11/18
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
import denseBlock
import setParameters
import pickle

NUM_CLASSES = 200

def new_block(parent, index, list_classes):
    '''The block is the container for an individual network. This function creates
    a new block that predicts from a list of given classes'''
    m = block(parent.name + index, list_classes)
    m.block_labels = np.zeros(NUM_CLASSES)#Labels specific to the indexing of this block
    m.block_labels[block_info.labels] = np.arange(len(block_info.labels)) + 1
    if len(m.labels) > 1:#If there is more than one class.
        setParameters.set_parameters(m, parent)
        define_block(parent.next_input, m)
    update_dict(m)
    return m

def generate_children(block_info):
    '''This function takes a block and computes its accuracy, making groups that
    have very little inter-group error'''
    #get the confusion matrix
    matrix = get_confusion_matrix(block_info)[1:][1:]
    #get the groups
    groups = return_groups(matrix, block_info.threshold)
    #generate children blocks
    for index, group in enumerate(groups):
        group = block_info.labels[group]
        block = new_block(block_info, str(index + 1), group)
        block_info.chlidren.append(block)


def get_confusion_matrix(block_info, batch_size, num_batches):
    '''Generates a confusion matrix from the data'''
    falsePercents = np.zeros((len(block_info.classes), len(block_info.classes)))
    totals = np.zeros((len(block_info.classes), len(block_info.classes)))
    increment = np.ones(len(block_info.classes))
    for i in range(num_batches):
        #Run Code here to predict class labels
        #data, labels = get_batch(batch_size)
        #feed_dict = {blah blah blah} TODO
        #predicted = sess.run(predicted_labels ,feed_dict = feed_dict)
        it = np.nditer(labels, flags=['f_index'],op_flags=['readwrite'])
        while not it.finished:
            falsePercents[it[0]] += predicted[it.index]
            totals[it[0]] += increment
            it.iternext()
    return falsePercents/totals

def update_dict(block_info):
    '''Update the global dictionary mapping labels to their path.'''
    for i, label in enumerate(block_info.labels):
        path.update(label, (m.name, str(i+1)))

def train_block(block_info, input_x, labels, input_placeholder):
    '''Train an individual block. First we must reconfigure the labels to put them
    in the appropriate indeces for the model, then we train it. Then repeat the
    process with its children.'''
    block_labels_curr = block_info.block_labels[labels]
    #TODO
    #Create Global Variables sess, train_writer, time_step
    train_dict = {input_placeholder: input_x, block_info.y: block_labels_curr}
    _, y_conv, cross_entropy_summary, accuracy_summary = sess.run([m.train_step, m.y_conv, m.cross_entropy_summary, m.accuracy_summary],
    feed_dict=feed_dict)#train generator)
    train_writer.add_summary(cross_entropy_summary, time_step)
    train_writer.add_summary(accuracy_summary, time_step)

    for child in block_info.children:
        if len(child.labels) > 1:
            next_input_x = np.where(np.isin(labels, child.labels) | np.isin(y_conv, child.labels),input_x)
            next_labels = np.where(np.isin(labels, child.labels) | np.isin(y_conv, child.labels),labels)
            train_block(child, next_input_x, next_labels, input_placeholder)


def restore_models():
    pass


if __name__ == "__main__":
    #paths is a dictionary containing the binary search paths to each
    #PATHS START AT 1
    paths = {} #THERE SHOULD NOT BE A 0 IN ANY STRINGS
    classes = np.arange(NUM_CLASSES)
    m = new_block("S", None, classes)
    for i in range(NUM_CLASSES):
        paths.update(classes[i],("S" ,str(i+1)))
