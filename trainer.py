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
import hyperparameterProgression

NUM_CLASSES = 200

def new_block(parent, index, list_classes):
    m = block(parent.name + index)
    m.labels = list_classes
    if len(m.labels) > 1:
        define_block(parent.next_input, m)
    update_dict(m)
    return m

def generate_children(block_info):
    #get the confusion matrix
    matrix = get_confusion_matrix(block_info)[1:][1:]
    #get the groups
    groups = return_groups(matrix, block_info.threshold)
    #generate children blocks
    for index, group in enumerate(groups):
        group = block_info.labels[group]
        block = new_block(block_info, str(index + 1), group)
        block.input_shape = block_info.output_shape
        block_info.chlidren.append(block)


def get_confusion_matrix(block_info, batch_size, num_batches):
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
    for i, label in enumerate(block_info.labels):
        path.update(label, (m.name, str(i+1)))

def train_block(block_info, input, labels):
    pass

def train_network(block_info, input, labels):
    pass


if __name__ == "__main__":
    #paths is a dictionary containing the binary search paths to each
    #PATHS START AT 1
    paths = {} #THERE SHOULD NOT BE A 0 IN ANY STRINGS
    classes = np.arange(NUM_CLASSES)
    m = new_block("S", None, classes)
    for i in range(NUM_CLASSES):
        paths.update(classes[i],("S" ,str(i+1)))
