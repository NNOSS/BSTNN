#Author: Nick Steelman
#Date: 6/11/18
#gns126@gmail.com
#cleanestmink.com

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import os
import sys
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

NUM_CLASSES = 10
ITERATIONS = 100000
BATCH_SIZE = 100
TEST_BATCH_SIZE = 10000
WHEN_SAVE = 3000
WHEN_TEST = 100
TEST_SIZE = 10000

DATA_PATH = '/home/gtower/Data/cifar-100-python/'
TRAINING_NAME = 'train'
TESTING_NAME = 'test'
META_NAME = 'meta'
INPUT_SHAPE = 28, 28, 1
NUM_OUTPUTS = 5

LEARNING_RATE = 1e-3
BETA1 = .9
CONVOLUTIONS = [32, 64]
FULLY_CONNECTED_SIZE = 4096

RESTORE = True

trainers = []
perm_variables = []
temp_variables = []
train_stats = []
test_stats = [[],[]]

# PERM_MODEL_FILEPATH = '/home/gtower/Models/MNIST/perm_model.ckpt' #filepaths to model and summaries
# TEMP_MODEL_FILEPATH = '/home/gtower/Models/MNIST/temp_model.ckpt' #filepaths to model and summaries
PERM_MODEL_FILEPATH = '/home/gtower/Models/MNIST/perm2_model.ckpt' #filepaths to model and summaries
TEMP_MODEL_FILEPATH = '/home/gtower/Models/MNIST/temp2_model.ckpt'
PICKLE_FILEPATH = '/home/gtower/Models/MNIST/head.p'
SUMMARY_FILEPATH = '/home/gtower/Models/MNIST/Summaries/'
def update_dict(block_info):
    '''Update the global dictionary mapping labels to their path.'''
    for i, label in enumerate(block_info.labels):
        path[label] = (block_info.name, str(i+1))

def new_leaf_block(parent, index, list_classes):
    '''The block is the container for an individual network. This function creates
    a new block that predicts from a list of given classes'''
    global trainers
    global perm_variables
    global temp_variables
    global train_stats
    global test_stats
    m = densenetBlock.block(parent.name + index, list_classes, parent.filtered_input,
     parent.filtered_labels, parent.arbitrary_prediction)
    m.block_labels = np.zeros(NUM_CLASSES+1, dtype=np.int16)#Labels specific to the indexing of this block
    m.block_labels[m.labels] = np.arange(len(m.labels)) + 1
    m.terminal_points = np.full(NUM_CLASSES+1, True)
    m.terminal_points[0] = False
    setParameters.set_parameters(m, parent)
    densenetBlock.define_block_leaf(m, test_bool)
    trainers += [m.train_step]
    perm_variables += m.perm_variables
    temp_variables += m.temp_variables
    train_stats += [m.cross_entropy_summary , m.accuracy_summary_train]
    test_stats[0] += [m.accuracy_summary_test]
    test_stats[1] += [m.num_right]
    return m

def new_branch_block(parent, index, list_classes, children_groups):
    '''The block is the container for an individual network. This function creates
    a new block that predicts from a list of given classes'''
    m = densenetBlock.block(parent.name + index, list_classes, parent.filtered_input,
     parent.filtered_labels, parent.y_conv, children_groups = children_groups)
    m.block_labels = np.zeros(NUM_CLASSES, dtype=np.int16)#Labels specific to the indexing of this block
    for i, group in enumerate(children_groups):
        m.block_labels[group] = i + 1
    setParameters.set_parameters(m, parent)
    densenetBlock.define_block_branch(m, test_bool)
    return m

def get_input(next_train, next_test):
    test_bool = tf.placeholder_with_default(tf.constant(False), ())
    input_x, input_y =  tf.cond(test_bool, lambda: next_test, lambda: next_train, name = 'which_input')
    fake_prediction = tf.ones_like(input_y)
    return input_x, input_y, fake_prediction, test_bool

def define_head(list_classes, input_x, input_y, fake_prediction, children_groups = None):
    global trainers
    global perm_variables
    global temp_variables
    global train_stats
    global test_stats
    head_block = densenetBlock.block('S', list_classes, input_x, input_y, fake_prediction)
    head_block.block_labels = np.zeros(NUM_CLASSES+1, dtype=np.int16)#Labels specific to the indexing of this block
    head_block.block_labels[head_block.labels] = np.arange(len(head_block.labels)) + 1
    head_block.terminal_points = np.full(NUM_CLASSES+1, True)
    head_block.terminal_points[0] = False
    head_block.learning_rate = LEARNING_RATE
    head_block.beta1 = BETA1
    head_block.convolutions = CONVOLUTIONS
    head_block.fully_connected_size = FULLY_CONNECTED_SIZE
    densenetBlock.define_block_leaf(head_block, test_bool)
    trainers += [head_block.train_step]
    perm_variables += head_block.perm_variables
    temp_variables += head_block.temp_variables
    train_stats += [head_block.cross_entropy_summary , head_block.accuracy_summary_train]
    test_stats[0] += [head_block.accuracy_summary_test]
    test_stats[1] += [head_block.num_right]
    return head_block

def test_group(classifications, groups):
    wrong = 0
    tot = 0
    #This is horribly inefficient
    for group in groups:
        # print(class_names[group])
        for class_label in group:
            for i in range(len(classifications[0])):
                tot += classifications[class_label, i]
                if i not in group:
                    wrong += classifications[class_label, i]
    print(wrong/tot)

def get_confusion_matrix(block_info):
    '''Generates a confusion matrix from the data'''
    m = block_info
    falsePercents = np.zeros((len(m.labels) + 1, len(m.labels) + 1))
    classifications = np.zeros((len(m.labels) + 1, len(m.labels) + 1))
    totals = np.zeros((len(m.labels) + 1, len(m.labels) + 1))
    increment = np.ones(len(m.labels) + 1)
    # test_gen = getData.get_batch_generator(batch_size, DATA_PATH+TESTING_NAME)
    i=0
    while i<=TEST_SIZE:#when the generator is done, instantiate a new one
        i+=TEST_BATCH_SIZE
        try:
            y_conv, filtered_labels_false = sess.run([m.y_conv, m.filtered_labels_false], feed_dict={test_bool: True})#train generator)
            y_conv = np.squeeze(y_conv)
            y_exp = np.exp(y_conv)
            y_tot = np.sum(y_exp, axis = 1)
            y_conv_softmax = y_exp / y_tot[:, np.newaxis]
            it = np.nditer(filtered_labels_false, flags=['f_index'],op_flags=['readwrite'])
            predicted_labels = np.argmax(y_exp, axis = 1)
            # print(y_conv)
            # print(np.sum(y_conv, axis = 1))
            while not it.finished:
                totals[it[0]] += increment
                falsePercents[it[0]] += y_conv_softmax[it.index]
                classifications[it[0], predicted_labels[it.index] ] += 1
                it.iternext()
        except tf.errors.OutOfRangeError:
            break
    matrix = falsePercents/totals
    divide.symm_matrix(matrix)
    matrix = matrix[1:,1:]
    classifications = classifications[1:,1:]
    return matrix, classifications

def get_promising_group(matrix, classifications):
    y_tot = np.sum(classifications, axis = 1)
    classifications_norm = classifications / y_tot[:, np.newaxis]
    divide.symm_matrix(classifications_norm)
    groups_dict = divide.find_thresholds(classifications_norm)
    tot_values = len(matrix[0])+1
    print(tot_values)
    least_groups = len(matrix[0]) + 2
    for key in groups_dict.keys():
        max_group = 0
        for group in groups_dict[key]:
            if max_group < len(group):
                max_group = len(group)
        if max_group <= (tot_values/2)  and key < least_groups:
            least_groups = key
    print(groups_dict)
    print(classifications_norm)
    return groups_dict[least_groups]

def generate_children(block_info):
    '''This function takes a block and computes its accuracy, making groups that
    have very little inter-group error'''
    global trainers
    global perm_variables
    global temp_variables
    global train_stats
    global test_stats
    matrix, classifications = get_confusion_matrix(block_info)
    best_group = get_promising_group(matrix, classifications)
    test_group(classifications, best_group)
    block_info.children_groups = [block_info.labels[group] for group in best_group]
    block_info.block_labels = np.zeros(NUM_CLASSES+1, dtype=np.int16)#Labels specific to the indexing of this block
    block_info.terminal_points = np.full(len(block_info.children_groups)+1, False)
    for index, group in enumerate(block_info.children_groups):
        block_info.block_labels[group] = index + 1
        if len(group) ==1:
            block_info.terminal_points[index + 1] = True
    train_stats.remove(block_info.cross_entropy_summary)
    train_stats.remove(block_info.accuracy_summary_train)
    test_stats[0].remove(block_info.accuracy_summary_test)
    test_stats[1].remove(block_info.num_right)
    trainers.remove(block_info.train_step)

    densenetBlock.define_block_branch(block_info, test_bool,reuse = tf.AUTO_REUSE)
    for var in block_info.perm_variables:
        if var not in perm_variables:
            perm_variables += [var]

    for var in block_info.temp_variables:
        if var in temp_variables:
            temp_variables.remove(var)

    train_stats += [block_info.cross_entropy_summary , block_info.accuracy_summary_train]
    test_stats[0] += [block_info.accuracy_summary_test]
    test_stats[1] += [block_info.num_right]
    trainers += [block_info.train_step]

    for index, group in enumerate(block_info.children_groups):
        if len(group) > 1:
            block = new_leaf_block(block_info, str(index + 1), group)
            block_info.children.append(block)
        else:
            block_info.children.append(None)

    # print 'train_stats ', train_stats
    # print 'test_stats ',test_stats
    # print 'trainers ' ,trainers
    # print 'perm_variables ' ,perm_variables
    # print 'temp_variables ' ,temp_variables
    return block_info

def train_model(head_block ,num_iterations, test_bool):
    global time_step
    global test_stats
    time_step = 0
    epoch = 0
    input_images_summary = tf.summary.image("image_inputs", head_block.x ,max_outputs = NUM_OUTPUTS)
    test_accuracy = tf.reduce_sum(tf.concat(test_stats[1],axis = 0))
    merged_summary = tf.summary.merge(train_stats)
    test_stats[0] += [tf.summary.scalar('Total Accuracy', test_accuracy/TEST_BATCH_SIZE)]
    test_summary = tf.summary.merge(test_stats[0])
    for iteration in range(num_iterations):
        time_step += 1
        if time_step % 1000 ==0:
            sys.stdout.write(str(time_step) + ', ')

        # feed_dict = {m.input: input_x, m.y: one_hot_labels}
        outputs = [merged_summary] + [input_images_summary] + trainers
        outputs_tuple = sess.run(outputs)#train generator)
        train_writer.add_summary(outputs_tuple[0], time_step)
        train_writer.add_summary(outputs_tuple[1], time_step)
        if iteration % WHEN_TEST == 0:
            outputs = test_summary
            outputs_tuple = sess.run(outputs, feed_dict = {test_bool: True})#train generator)
            train_writer.add_summary(outputs_tuple, time_step)

        if iteration % WHEN_SAVE ==0:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)
            saver_temp.save(sess, TEMP_MODEL_FILEPATH)

def create_and_store_branches(head_block):
    global saver_perm
    global saver_temp
    traverse_tree(head_block)
    sess.run(tf.global_variables_initializer())
    saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    saver_temp.restore(sess, TEMP_MODEL_FILEPATH)
    saver_perm = tf.train.Saver(var_list=perm_variables)
    saver_temp = tf.train.Saver(var_list=temp_variables)
    saver_perm.save(sess, PERM_MODEL_FILEPATH)
    saver_temp.save(sess, TEMP_MODEL_FILEPATH)
    densenetBlock.clean_block_recursive(head_block)
    cPickle.dump(head_block, open( PICKLE_FILEPATH, "wb" ))

def traverse_tree(block_info):
    if block_info is None:
        return None
    elif len(block_info.children):
        [traverse_tree(child) for child in block_info.children]
    else:
        generate_children(block_info)

def restore_models(block_info, x, y, predictions):
    global trainers
    global perm_variables
    global temp_variables
    global train_stats
    global test_stats

    if block_info is None:
        return block_info
    elif len(block_info.children):
        block_info.terminal_points = np.full(len(block_info.children_groups)+1, False)
        for index, group in enumerate(block_info.children_groups):
            if len(group) ==1:
                block_info.terminal_points[index + 1] = True
        block_info.x = x
        block_info.y = y
        block_info.predictions = predictions
        densenetBlock.define_block_branch(block_info, test_bool)
        [restore_models(child, block_info.filtered_input,
         block_info.filtered_labels, block_info.arbitrary_prediction) for child in block_info.children]
    else:
        block_info.terminal_points = np.full(NUM_CLASSES+1, True)
        block_info.terminal_points[0] = False
        block_info.x = x
        block_info.y = y
        block_info.predictions = predictions
        densenetBlock.define_block_leaf(block_info, test_bool)

    trainers += [block_info.train_step]
    perm_variables += block_info.perm_variables
    temp_variables += block_info.temp_variables
    train_stats += [block_info.cross_entropy_summary , block_info.accuracy_summary_train]
    test_stats[0] += [block_info.accuracy_summary_test]
    test_stats[1] += [block_info.num_right]

if __name__ == "__main__":
    #paths is a dictionary containing the binary search paths to each
    #PATHS START AT 1
    ##############GET NAMES #############
    # class_names = getData.get_class_names(DATA_PATH + META_NAME)
    # class_names = np.array(class_names)
    sess = tf.Session()#start the session
    classes = np.arange(NUM_CLASSES) + 1
    ##############GET DATA###############
    train_mnist = getData.return_mnist_datatset_train().repeat().batch(BATCH_SIZE)
    test_mnist = getData.return_mnist_dataset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_mnist.make_initializable_iterator()
    test_iterator = test_mnist.make_initializable_iterator()
    train_input = train_iterator.get_next()
    test_input = test_iterator.get_next()
    sess.run([train_iterator.initializer, test_iterator.initializer])

    input_x, input_y, fake_prediction, test_bool = get_input(train_input, test_input)
    ############DEFINE#######
    if RESTORE:
        head_block = cPickle.load(open( PICKLE_FILEPATH, "rb" ))
        restore_models(head_block, input_x, input_y, fake_prediction)
    else:
        head_block = define_head(classes, train_input, test_input)
    ###########SAVE###########


    saver_perm = tf.train.Saver(var_list=perm_variables)
    saver_temp = tf.train.Saver(var_list=temp_variables)
    sess.run(tf.global_variables_initializer())
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
        saver_temp.restore(sess, TEMP_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)
        saver_temp.save(sess, TEMP_MODEL_FILEPATH)

    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    # print(head_block.block_labels)
    #


    # matrix, classifications = get_confusion_matrix(head_block, test_bool)
    # best_group = get_promising_group(matrix, classifications)
    # print(best_group)
    # test_group(classifications, best_group)
    # print(classifications)
    # create_and_store_branches(head_block)
    train_model(head_block , ITERATIONS, test_bool)
