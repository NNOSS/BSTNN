import tensorflow as tf
import cPickle
import numpy as np

file_path = '/home/gtower/Data/MNIST_data/'

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [28, 28, 1])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8) + 1  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

def return_mnist_datatset_train():
    images_file = file_path + 'train-images-idx3-ubyte'
    labels_file = file_path + 'train-labels-idx1-ubyte'
    images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def return_mnist_dataset_test():
    images_file = file_path + 't10k-images-idx3-ubyte'
    labels_file = file_path + 't10k-labels-idx1-ubyte'
    images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

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
