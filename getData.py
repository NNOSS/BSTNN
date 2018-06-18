from tensorflow.examples.tutorials.mnist import input_data
import cPickle
import numpy as np

def return_mnist_train_generator(batch_size):
    mnist = input_data.read_data_sets('/home/gtower/Data/MNIST_data', one_hot=False)

    for i in range(1000000):
        batch = mnist.train.next_batch(batch_size)
        batch[0] = np.reshape(batch[0], [batch_size,INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        yield batch[0], batch[1]

def return_mnist_test_generator(batch_size):
    mnist = input_data.read_data_sets('/home/gtower/Data/MNIST_data', one_hot=False)
    for i in range(0, len(mnist.test.labels), batch_size):
        batch = np.reshape(mnist.test.images[i:i+batch_size], [batch_size,INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]])
        yield batch, mnist.test.labels[i:i+batch_size]

def get_batch_generator(batch_size, file_name):
    data_dict = unpickle(file_name)
    data = data_dict['data']
    labels = data_dict['fine_labels']
    for i in range(0,len(labels),batch_size):
        j = min(i+batch_size,len(labels))
        batch = data[i:j]
        batch = np.reshape(batch, [batch_size, INPUT_SHAPE[2], INPUT_SHAPE[0], INPUT_SHAPE[1]])
        batch = np.transpose(batch, (0, 2, 3, 1))
        yield batch, labels[i:j]

def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        data_dict = cPickle.load(fo)
    return data_dict

def get_class_names(file_name):
    meta_dict = unpickle(file_name)

    return meta_dict['fine_label_names']

if __name__ == "__main__":
    DATA_PATH = '/home/gtower/Data/cifar-100-python/'
    TRAINING_NAME = 'train'
    TESTING_NAME = 'test'
    META_NAME = 'meta'
    print(get_class_names(DATA_PATH + META_NAME))
