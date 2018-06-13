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

NUM_CLASSES = 101
ITERATIONS = 100000
BATCH_SIZE = 50
WHEN_SAVE = 200
WHEN_TEST = 5

DATA_PATH = '/home/gtower/Data/cifar-100-python/'
TRAINING_NAME = 'train'
TESTING_NAME = 'test'
INPUT_SHAPE = 32, 32, 3
NUM_OUTPUTS = 5

LEARNING_RATE = 1e-3
BETA1 = .9
CONVOLUTIONS = [64, 128, 256]
FULLY_CONNECTED_SIZE = 4096

RESTORE = False

MODEL_FILEPATH = '/home/gtower/Models/CIFAR100/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH = '/home/gtower/Models/CIFAR100/Summaries/'

def new_block(parent, index, list_classes):
    '''The block is the container for an individual network. This function creates
    a new block that predicts from a list of given classes'''
    m = densenetBlock.block(parent.name + index, list_classes)
    m.block_labels = np.zeros(NUM_CLASSES, dtype=np.int16)#Labels specific to the indexing of this block
    m.block_labels[block_info.labels] = np.arange(len(block_info.labels)) + 1
    if len(m.labels) > 1:#If there is more than one class.
        setParameters.set_parameters(m, parent)
        define_block(m)
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

def get_confusion_matrix(block_info, batch_size):
    '''Generates a confusion matrix from the data'''
    m = block_info
    falsePercents = np.zeros((len(m.classes), len(m.classes)))
    totals = np.zeros((len(m.classes), len(m.classes)))
    increment = np.ones(len(m.classes))
    test_gen = get_batch_generator(batch_size, DATA_PATH+TESTING_NAME)
    x_batch_test, y_labels_test = next(test_gen,(None, None))
    while x_batch_test is not None:#when the generator is done, instantiate a new one
        input_x_test = np.reshape(x_batch_test, [len(y_labels), INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]])
        input_x_test = np.transpose(input_x_test, (0, 2, 3, 1))
        block_labels_curr = m.block_labels[labels]
        indexes = np.arange(len(block_labels_curr))
        one_hot_labels = np.zeros((len(block_labels_curr),len(block_info.labels)))
        one_hot_labels[indexes, block_labels_curr] = 1
        feed_dict = {m.input: input_x, m.y: one_hot_labels}
        y_conv = sess.run([m.y_conv], feed_dict=feed_dict)#train generator)
        y_exp = np.exp(y_conv)
        y_tot = np.sum(y_exp, axis = 1)
        y_conv = y_exp / y_tot[:, np.newaxis]
        it = np.nditer(block_labels_curr, flags=['f_index'],op_flags=['readwrite'])
        while not it.finished:
            falsePercents[it[0]] += y_conv[it.index]
            totals[it[0]] += increment
            it.iternext()
        x_batch_test, y_labels_test = next(test_gen,(None, None))
    matrix = falsePercents/totals
    divide.symm_matrix(matrix)
    return matrix

def update_dict(block_info):
    '''Update the global dictionary mapping labels to their path.'''
    for i, label in enumerate(block_info.labels):
        path[label] = (block_info.name, str(i+1))

def define_head(list_classes):
    head_block = densenetBlock.block('S', list_classes, INPUT_SHAPE)
    head_block.block_labels = np.zeros(NUM_CLASSES, dtype=np.int16)#Labels specific to the indexing of this block
    head_block.block_labels[head_block.labels] = np.arange(len(head_block.labels)) + 1
    head_block.learning_rate = LEARNING_RATE
    head_block.beta1 = BETA1
    head_block.convolutions = CONVOLUTIONS
    head_block.fully_connected_size = FULLY_CONNECTED_SIZE
    densenetBlock.define_block(head_block)
    update_dict(head_block)
    return head_block

def train_model(head_block ,num_iterations):
    global time_step
    time_step = 0
    batch_size = BATCH_SIZE
    train_gen = get_batch_generator(batch_size, DATA_PATH+TRAINING_NAME)
    test_gen = get_batch_generator(batch_size, DATA_PATH+TESTING_NAME)
    epoch = 0
    input_images_summary = tf.summary.image("image_inputs", head_block.input ,max_outputs = NUM_OUTPUTS)

    for iteration in range(num_iterations):
        time_step += 1
        # if time_step % 10 ==0:
        #     # print(time_step)
        x_batch, y_labels = next(train_gen,(None, None))

        while x_batch is None:#when the generator is done, instantiate a new one
            train_gen = get_batch_generator(batch_size, DATA_PATH+TRAINING_NAME)
            epoch += 1
            print("Completed Epoch. Num Completed: ",epoch)
            x_batch, y_labels = next(train_gen,(None, None))
        input_x = np.reshape(x_batch, [len(y_labels), INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]])
        input_x = np.transpose(input_x, (0, 2, 3, 1))
        train_block(head_block, input_x, y_labels)

        if iteration % WHEN_TEST == 0:
            x_batch_test, y_labels_test = next(test_gen,(None, None))
            while x_batch_test is None:#when the generator is done, instantiate a new one
                test_gen = get_batch_generator(batch_size, DATA_PATH+TESTING_NAME)
                x_batch_test, y_labels_test = next(test_gen,(None, None))
            input_x_test = np.reshape(x_batch_test, [len(y_labels), INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]])
            input_x_test = np.transpose(input_x_test, (0, 2, 3, 1))
            test_block(head_block, input_x_test, y_labels_test)

        if iteration % WHEN_SAVE ==0:
            saver.save(sess, MODEL_FILEPATH)

def train_block(block_info, input_x, labels):
    '''Train an individual block. First we must reconfigure the labels to put them
    in the appropriate indeces for the model, then we train it. Then repeat the
    process with its children.'''
    m = block_info
    block_labels_curr = block_info.block_labels[labels]
    indexes = np.arange(len(block_labels_curr))
    one_hot_labels = np.zeros((len(block_labels_curr),len(block_info.labels)))
    one_hot_labels[indexes, block_labels_curr] = 1
    feed_dict = {m.input: input_x, m.y: one_hot_labels}
    _, output, y_conv, cross_entropy_summary, accuracy_summary = sess.run([m.train_step,
     m.output, m.y_conv, m.cross_entropy_summary, m.accuracy_summary_train],
    feed_dict=feed_dict)#train generator)
    # print(one_hot_labels[0:5])
    # print(block_labels_curr[0:5])
    # print(time_step)
    # print(labels[0:5])
    train_writer.add_summary(cross_entropy_summary, time_step)
    train_writer.add_summary(accuracy_summary, time_step)

    for child in block_info.children:
        if len(child.labels) > 1:
            next_input_x = np.where(np.isin(labels, child.labels) | np.isin(y_conv, child.labels),output)
            next_labels = np.where(np.isin(labels, child.labels) | np.isin(y_conv, child.labels),labels)
            train_block(child, next_input_x, next_labels)

def test_block(block_info, input_x, labels):
    '''Train an individual block. First we must reconfigure the labels to put them
    in the appropriate indeces for the model, then we train it. Then repeat the
    process with its children.'''
    m = block_info
    block_labels_curr = block_info.block_labels[labels]
    indexes = np.arange(len(block_labels_curr))
    one_hot_labels = np.zeros((len(block_labels_curr),len(block_info.labels)))
    one_hot_labels[indexes, block_labels_curr] = 1
    feed_dict = {m.input: input_x, m.y: one_hot_labels}
    output, y_conv, accuracy_summary = sess.run([
     m.output, m.y_conv, m.accuracy_summary_test],
    feed_dict=feed_dict)#train generator)
    train_writer.add_summary(accuracy_summary, time_step)
    for child in block_info.children:
        if len(child.labels) > 1:
            next_input_x = np.where(np.isin(labels, child.labels) | np.isin(y_conv, child.labels),output)
            next_labels = np.where(np.isin(labels, child.labels) | np.isin(y_conv, child.labels),labels)
            test_block(child, next_input_x, next_labels)

def get_batch_generator(batch_size, file_name):
    data_dict = unpickle(file_name)
    data = data_dict['data']
    labels = data_dict['fine_labels']
    for i in range(0,len(labels),batch_size):
        j = min(i+batch_size,len(labels))
        yield data[i:j], labels[i:j]

def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        data_dict = cPickle.load(fo)
    return data_dict

def restore_models():
    pass

if __name__ == "__main__":
    #paths is a dictionary containing the binary search paths to each
    #PATHS START AT 1
    path = {} #THERE SHOULD NOT BE A 0 IN ANY STRINGS
    sess = tf.Session()#start the session
    classes = np.arange(NUM_CLASSES)
    head_block = define_head(classes)
    saver = tf.train.Saver()#saving and tensorboard
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    if MODEL_FILEPATH is not None and RESTORE:
        saver.restore(sess, MODEL_FILEPATH)
    else:
        print('SAVE')
        saver.save(sess, MODEL_FILEPATH)


    train_model(head_block , ITERATIONS)
