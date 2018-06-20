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

NUM_CLASSES = 10
ITERATIONS = 100000
BATCH_SIZE = 100
WHEN_SAVE = 200
WHEN_TEST = 5

DATA_PATH = '/home/gtower/Data/cifar-100-python/'
TRAINING_NAME = 'train'
TESTING_NAME = 'test'
META_NAME = 'meta'
INPUT_SHAPE = 28, 28, 1
NUM_OUTPUTS = 5

LEARNING_RATE = 1e-3
BETA1 = .9
CONVOLUTIONS = [32, 64, 128]
FULLY_CONNECTED_SIZE = 4096

RESTORE = False

trainers = []
perm_variables = []
temp_variables = []
statistics = []

PERM_MODEL_FILEPATH = '/home/gtower/Models/MNIST/perm_model.ckpt' #filepaths to model and summaries
TEMP_MODEL_FILEPATH = '/home/gtower/Models/MNIST/temp_model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH = '/home/gtower/Models/MNIST/Summaries/'
def update_dict(block_info):
    '''Update the global dictionary mapping labels to their path.'''
    for i, label in enumerate(block_info.labels):
        path[label] = (block_info.name, str(i+1))

def new_leaf_block(parent, index, list_classes):
    '''The block is the container for an individual network. This function creates
    a new block that predicts from a list of given classes'''
    m = densenetBlock.block(parent.name + index, list_classes, parent.filtered_input,
     parent.filtered_labels, parent.y_conv)
    m.block_labels = np.zeros(m.labels+1, dtype=np.int16)#Labels specific to the indexing of this block
    m.block_labels[m.labels] = np.arange(len(m.labels)) + 1
    if len(m.labels) > 1:#If there is more than one class.
        setParameters.set_parameters(m, parent)
        densenetBlock.define_block_leaf(m)
    update_dict(m)
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
    densenetBlock.define_block_branch(m)
    update_dict(m)
    return m

def define_head(list_classes, x, y, children_groups = None):
    fake_prediction = tf.zeros_like(y)
    if children_groups is None:
        head_block = densenetBlock.block('S', list_classes, x, y, fake_prediction)
        head_block.block_labels = np.zeros(NUM_CLASSES+1, dtype=np.int16)#Labels specific to the indexing of this block
        head_block.block_labels[head_block.labels] = np.arange(len(head_block.labels)) + 1
        head_block.learning_rate = LEARNING_RATE
        head_block.beta1 = BETA1
        head_block.convolutions = CONVOLUTIONS
        head_block.fully_connected_size = FULLY_CONNECTED_SIZE
        densenetBlock.define_block_leaf(head_block)
    else:
        head_block = densenetBlock.block('S', list_classes, x, y, fake_prediction, children_groups = children_groups)
        head_block.block_labels = np.zeros(NUM_CLASSES, dtype=np.int16)#Labels specific to the indexing of this block
        for i, group in enumerate(children_groups):
            head_block.block_labels[group] = i + 1
        head_block.learning_rate = LEARNING_RATE
        head_block.beta1 = BETA1
        head_block.convolutions = CONVOLUTIONS
        head_block.fully_connected_size = FULLY_CONNECTED_SIZE
        setParameters.set_parameters(head_block, parent)
        densenetBlock.define_block_branch(head_block)
    update_dict(head_block)
    trainers += [head_block.train_step]
    perm_variables += [head_block.perm_variables]
    temp_variables += [head_block.temp_variables]
    statistics += [head_block.cross_entropy_summary , head_block.accuracy_summary_train]
    return head_block

def test_group(classifications, groups):
    wrong = 0
    tot = 0
    #This is horribly inefficient
    for group in groups:
        print(class_names[group])
        for class_label in group:
            for i in range(NUM_CLASSES-1):
                tot += classifications[class_label, i]
                if i not in group:
                    wrong += classifications[class_label, i]
    print(wrong/tot)

def get_confusion_matrix(block_info, batch_size):
    '''Generates a confusion matrix from the data'''
    m = block_info
    falsePercents = np.zeros((len(m.labels) + 1, len(m.labels) + 1))
    classifications = np.zeros((len(m.labels) + 1, len(m.labels) + 1))
    totals = np.zeros((len(m.labels) + 1, len(m.labels) + 1))
    increment = np.ones(len(m.labels) + 1)
    # test_gen = getData.get_batch_generator(batch_size, DATA_PATH+TESTING_NAME)
    test_gen = getData.return_mnist_test_generator(batch_size)

    x_batch_test, y_labels_test = next(test_gen,(None, None))
    while x_batch_test is not None:#when the generator is done, instantiate a new one

        feed_dict = {x: x_batch_test, y: y_labels_test}
        y_conv, filtered_labels_false = sess.run([m.y_conv, m.filtered_labels_false], feed_dict=feed_dict)#train generator)
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
        x_batch_test, y_labels_test = next(test_gen,(None, None))
    matrix = falsePercents/totals
    divide.symm_matrix(matrix)
    matrix = matrix[1:,1:]
    classifications = classifications[1:,1:]
    return matrix, classifications

def generate_children(block_info):
    '''This function takes a block and computes its accuracy, making groups that
    have very little inter-group error'''
    matrix, classifications = get_confusion_matrix(block_info)
    y_tot = np.sum(classifications, axis = 1)
    classifications_norm = classifications / y_tot[:, np.newaxis]
    divide.symm_matrix(classifications_norm)
    groups_dict = divide.find_thresholds(classifications_norm)
    least_groups = len(block_info.labels) + 2
    for key in groups_dict.keys():
        max_group = 0
        for group in groups_dict[key]:
            if max_group < len(group):
                max_group = len(group)
        if max_group < (len(block_info.labels)/2)  and key < least_groups:
            least_groups = key
    block_info.children_groups = groups_dict[least_groups]
    test_group(classifications, block_info.children_groups)
    block_info.block_labels = np.zeros(NUM_CLASSES, dtype=np.int16)#Labels specific to the indexing of this block
    for index, group in enumerate(block_info.children_groups):
        block_info.block_labels[group] = index + 1

    statistics.remove(block_info.cross_entropy_summary)
    statistics.remove(block_info.accuracy_summary_train)
    trainers.remove(block_info.train_step)

    densenetBlock.define_block_branch(m)
    for var in block_info.perm_variables:
        if var not in perm_variables:
            perm_variables += [var]

    for var in block_info.temp_variables:
        if var in temp_variables:
            temp_variables.remove(var)
    statistics += [block_info.cross_entropy_summary , block_info.accuracy_summary_train]
    trainers += [block_info.train_step]

    for index, group in enumerate(block_info.children_groups):
        group = block_info.labels[group]
        block = new_leaf_block(block_info, str(index + 1), group)
        block_info.children.append(block)
        trainers += [block.train_step]
        perm_variables += [block.perm_variables]
        temp_variables += [block.temp_variables]
        statistics += [block.cross_entropy_summary , block.accuracy_summary_train]

    return block_info

def train_model(head_block ,num_iterations):
    global time_step
    time_step = 0
    epoch = 0
    input_images_summary = tf.summary.image("image_inputs", head_block.input ,max_outputs = NUM_OUTPUTS)
    merged_summary = tf.summary.merge(statistics)
    for iteration in range(num_iterations):
        time_step += 1
        # if time_step % 10 ==0:
        #     # print(time_step

        while x_batch is None:#when the generator is done, instantiate a new one
            # train_gen = getData.get_batch_generator(batch_size, DATA_PATH+TRAINING_NAME)
            train_gen = getData.return_mnist_train_generator(batch_size)
            epoch += 1
            print("Completed Epoch. Num Completed: ",epoch)
            x_batch, y_labels = next(train_gen,(None, None))

        # feed_dict = {m.input: input_x, m.y: one_hot_labels}
        outputs = [merged_summary] + [input_images_summary] + trainers
        outputs_tuple = sess.run(outputs)#train generator)
        train_writer.add_summary(outputs_tuple[0])
        train_writer.add_summary(outputs_tuple[1])
        # if iteration % WHEN_TEST == 0:
        #     x_batch_test, y_labels_test = next(test_gen,(None, None))
        #     while x_batch_test is None:#when the generator is done, instantiate a new one
        #         # test_gen = getData.get_batch_generator(batch_size, DATA_PATH+TESTING_NAME)
        #         test_gen = getData.return_mnist_test_generator(batch_size)
        #         x_batch_test, y_labels_test = next(test_gen,(None, None))
        #     test_block(head_block, x_batch_test, y_labels_test)

        if iteration % WHEN_SAVE ==0:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)
            saver_temp.save(sess, TEMP_MODEL_FILEPATH)

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

def restore_models():
    pass

if __name__ == "__main__":
    #paths is a dictionary containing the binary search paths to each
    #PATHS START AT 1
    ##############GET NAMES #############
    class_names = getData.get_class_names(DATA_PATH + META_NAME)
    class_names = np.array(class_names)

    path = {} #THERE SHOULD NOT BE A 0 IN ANY STRINGS
    sess = tf.Session()#start the session
    classes = np.arange(NUM_CLASSES) + 1
    ##############GET DATA###############
    train_mnist = getData.return_mnist_datatset_train().repeat().batch(BATCH_SIZE)
    test_mnist = getData.return_mnist_dataset_test().batch(BATCH_SIZE)
    train_iterator = train_mnist.make_initializable_iterator()
    test_iterator = test_mnist.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    test_input, test_label = test_iterator.get_next()
    sess.run([train_iterator.initializer, test_iterator.initializer])

    ############DEFINE#######
    head_block = define_head(classes, train_input, train_label)
    ###########SAVE###########
    saver_perm = tf.train.Saver(perm_variables)
    saver_temp = tf.train.Saver(temp_variables)
    sess.run(tf.global_variables_initializer())
    if MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, MODEL_FILEPATH)
        saver_temp.restore(sess, MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, MODEL_FILEPATH)
        saver_temp.save(sess, MODEL_FILEPATH)

    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    train_model(head_block , ITERATIONS)
