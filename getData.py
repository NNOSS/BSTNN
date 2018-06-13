import cPickle

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

def get_class_names(file_name):
    meta_dict = unpickle(file_name)

    return meta_dict['fine_label_names']

if __name__ == "__main__":
    DATA_PATH = '/home/gtower/Data/cifar-100-python/'
    TRAINING_NAME = 'train'
    TESTING_NAME = 'test'
    META_NAME = 'meta'
    print(get_class_names(DATA_PATH + META_NAME))
