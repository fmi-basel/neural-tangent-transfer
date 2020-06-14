import gzip
import os
import numpy as onp
import itertools
from scipy.io import loadmat

# import os
import pickle
import numpy.random as npr

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file


# This script is adapted from the SNIP repository: https://github.com/namhoonlee/snip-public/blob/master/snip/mnist.py

def _read32(bytestream):
    dt = onp.dtype(onp.uint32).newbyteorder(">")
    return onp.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(filename):
    """Extract the images into a 4D uint8 np array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError("Invalid magic number %d in MNIST image file: %s" % (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = onp.frombuffer(buf, dtype=onp.uint8)
#         data = data.reshape(num_images, rows, cols, 1)
        data = data.reshape(num_images, rows * cols)
        return data

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = onp.arange(num_labels) * num_classes
    labels_one_hot = onp.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 np array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError("Invalid magic number %d in MNIST label file: %s" % (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = onp.frombuffer(buf, dtype=onp.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

def read_mnist_data(path_dataset, one_hot=True, STANDARDIZE_BOOL = True, LOAD_TRAIN_PORTION = 1.0):
    TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
    TRAIN_LABELS = "train-labels-idx1-ubyte.gz"
    TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
    TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

    train_images = extract_images(os.path.join(path_dataset, TRAIN_IMAGES))
    train_labels = extract_labels(os.path.join(path_dataset, TRAIN_LABELS), one_hot=one_hot)
    test_images = extract_images(os.path.join(path_dataset, TEST_IMAGES))
    test_labels = extract_labels(os.path.join(path_dataset, TEST_LABELS), one_hot=one_hot)

    # Pre-processing (normalize)
    train_images = onp.divide(train_images, 255, dtype=onp.float32)
    test_images = onp.divide(test_images, 255, dtype=onp.float32)
    
    if STANDARDIZE_BOOL:
        mean = onp.mean(train_images)
        std = onp.std(train_images)
        train_images = (train_images - mean) / std
        test_images = (test_images - mean) / std
    
    load_num = int(len(train_images) * LOAD_TRAIN_PORTION  )
    
    train_images = train_images[:load_num]
    test_images = test_images[:load_num]
    
    train_labels = train_labels[:load_num]
    test_labels = test_labels[:load_num]
    
    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels},
    }
    return dataset

def read_cifar_10_data(path_dataset, STANDARDIZE_BOOL = True):
    # train batch
    train_batch = {}
    for i in range(5):
        filename = os.path.join(path_dataset, 'data_batch_{}'.format(i+1))
        with open(filename, 'rb') as f:
            try:
                batch = pickle.load(f, encoding='bytes')
            except TypeError:
                batch = pickle.load(f) # for python 2
            for key in batch.keys():
                train_batch.setdefault(key, []).extend(batch[key])
    train_batch = {k: onp.stack(v, 0) for k, v in train_batch.items()} # stack into one batch

    # test batch
    filename = os.path.join(path_dataset, 'test_batch')
    with open(filename, 'rb') as f:
        try:
            test_batch = pickle.load(f, encoding='bytes')
        except TypeError:
            test_batch = pickle.load(f)

    # Reshape images: (n, 3072) -> (n, 32, 32, 3)
    label_key = 'labels'.encode('utf-8')
    train_images = onp.transpose(
        onp.reshape(train_batch['data'.encode('utf-8')], [-1, 3, 32, 32]), [0,2,3,1])
    train_labels = onp.asarray(train_batch[label_key])
    train_labels = dense_to_one_hot(train_labels)
    
    
    
    test_images = onp.transpose(
        onp.reshape(test_batch['data'.encode('utf-8')], [-1, 3, 32, 32]), [0,2,3,1])
    test_labels = onp.asarray(test_batch[label_key])
    test_labels = dense_to_one_hot(test_labels)
    
    

    # Pre-processing (normalize)
    train_images = onp.divide(train_images, 255, dtype=onp.float32)
    test_images = onp.divide(test_images, 255, dtype=onp.float32)
    
    if STANDARDIZE_BOOL: 
        channel_mean = onp.mean(train_images, axis=(0,1,2), dtype=onp.float32, keepdims=True)
        channel_std = onp.std(train_images, axis=(0,1,2), dtype=onp.float32, keepdims=True)
        train_images = (train_images - channel_mean) / channel_std
        test_images = (test_images - channel_mean) / channel_std

    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels},
    }
    return dataset




def read_cifar_100_data(path_data, subdir, label_mode='fine', STANDARDIZE_BOOL = True):
    """Loads [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
    This is a dataset of 50,000 32x32 color training images and
    10,000 test images, labeled over 100 fine-grained classes that are
    grouped into 20 coarse-grained classes. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
    Arguments:
      label_mode: one of "fine", "coarse". If it is "fine" the category labels
      are the fine-grained labels, if it is "coarse" the output labels are the
      coarse-grained superclasses.
    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
      x_train, x_test: uint8 arrays of RGB image data with shape
        (num_samples, 3, 32, 32) if the `tf.keras.backend.image_data_format` is
        'channels_first', or (num_samples, 32, 32, 3) if the data format
        is 'channels_last'.
      y_train, y_test: uint8 arrays of category labels with shape
        (num_samples, 1).
    Raises:
      ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(  dirname, origin=origin, untar=True, file_hash= '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7', cache_dir = path_data,
cache_subdir = subdir)

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = onp.reshape(y_train, (len(y_train), 1))
    y_test = onp.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    
    # Pre-processing (normalize)
    train_images = onp.divide(x_train, 255, dtype=onp.float32)
    test_images = onp.divide(x_test, 255, dtype=onp.float32)
    
    train_labels = dense_to_one_hot(y_train, num_classes=100)
    test_labels = dense_to_one_hot(y_test, num_classes=100)
    
    if STANDARDIZE_BOOL: 
        channel_mean = onp.mean(train_images, axis=(0,1,2), dtype=onp.float32, keepdims=True)
        channel_std = onp.std(train_images, axis=(0,1,2), dtype=onp.float32, keepdims=True)
        train_images = (train_images - channel_mean) / channel_std
        test_images = (test_images - channel_mean) / channel_std

    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels}}
    
    return dataset

def construct_bin_dataset(orig_dataset):
    
#     dataset_str = 'mnist'
#     orig_dataset = Dataset(datasource = dataset_str, STANDARDIZE_BOOL = False)

    # train and test images
#     train_images = orig_dataset.dataset['train']['input']
#     test_images = orig_dataset.dataset['test']['input']
    train_images = orig_dataset['train']['input']
    test_images = orig_dataset['test']['input']
    # train and test labels
    train_labels_dense = onp.argmax(orig_dataset['train']['label'], axis = 1)
    test_labels_dense = onp.argmax(orig_dataset['test']['label'], axis = 1)

    # indices of the first two classes
    num_examples_per_class = 250

        
#     train_bin_class_idx = onp.where( (train_labels_dense == 0) | (train_labels_dense == 1) )
#     test_bin_class_idx = onp.where( (test_labels_dense == 0) | (test_labels_dense == 1) )
    
    train_bin_class0_idx = onp.where( train_labels_dense == 0 )[0][:num_examples_per_class]
    train_bin_class1_idx = onp.where( train_labels_dense == 1 )[0][:num_examples_per_class]
    train_bin_class_idx = onp.hstack((train_bin_class0_idx, train_bin_class1_idx)) 
    
    rng = npr.RandomState(0)
    train_bin_class_idx = rng.permutation(train_bin_class_idx)
            
    
    test_bin_class0_idx = onp.where( test_labels_dense == 0)[0][:num_examples_per_class]  
    test_bin_class1_idx = onp.where( test_labels_dense == 1)[0][:num_examples_per_class]
    test_bin_class_idx = onp.hstack((test_bin_class0_idx, test_bin_class1_idx))
    test_bin_class_idx = rng.permutation(test_bin_class_idx)

    # train & test labels in the first two classes
    train_images_bin_class = train_images[train_bin_class_idx]
    test_images_bin_class = test_images[test_bin_class_idx]

    # train & test images in the first two classes
    train_labels_dense_bin_class = train_labels_dense[train_bin_class_idx]
    test_labels_dense_bin_class = test_labels_dense[test_bin_class_idx]

    # train & test labels in the first two classes
    train_labels_bin_class = dense_to_one_hot(train_labels_dense_bin_class, num_classes= 2)
    test_labels_bin_class = dense_to_one_hot(test_labels_dense_bin_class, num_classes= 2)

#     dataset = { 'train': {'input': train_images_bin_class[:num_examples], 'label': train_labels_bin_class[:num_examples]},
#                'test': {'input': test_images_bin_class[:num_examples], 'label': test_labels_bin_class[:num_examples]} }

    dataset = { 'train': {'input': train_images_bin_class, 'label': train_labels_bin_class},
               'test': {'input': test_images_bin_class, 'label': test_labels_bin_class} }

    return dataset



def read_svhn_data(path_dataset, STANDARDIZE_BOOL = True):

    def load_data(path):
        """ Helper function for loading a MAT-File"""
        data = loadmat(path)
        return data['X'], data['y']

    train_images, train_labels = load_data(path_dataset + 'train_32x32.mat')
    test_images, test_labels = load_data(path_dataset +  'test_32x32.mat')

    train_images, train_labels = train_images.transpose((3,0,1,2)), train_labels[:,0]
    test_images, test_labels = test_images.transpose((3,0,1,2)), test_labels[:,0]


    # # Pre-processing (normalize)
    train_images = onp.divide(train_images, 255, dtype=onp.float32)
    test_images = onp.divide(test_images, 255, dtype=onp.float32)


    train_labels = dense_to_one_hot(train_labels)
    test_labels = dense_to_one_hot(test_labels)



    if STANDARDIZE_BOOL: 
        channel_mean = onp.mean(train_images, axis=(0,1,2), dtype=onp.float32, keepdims=True)
        channel_std = onp.std(train_images, axis=(0,1,2), dtype=onp.float32, keepdims=True)
        train_images = (train_images - channel_mean) / channel_std
        test_images = (test_images - channel_mean) / channel_std

    dataset = {
        'train': {'input': train_images, 'label': train_labels},
        'test': {'input': test_images, 'label': test_labels},
    }
    
    return dataset

class Dataset(object):
    def __init__(self, datasource, path_data = '/tungstenfs/scratch/gzenke/liutian/myData/', STANDARDIZE_BOOL  = True,  VALIDATION_FRACTION = 0.1, LOAD_TRAIN_PORTION = 1.0, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = onp.random.RandomState(9)
        
        if VALIDATION_FRACTION < 0.0 or VALIDATION_FRACTION > 1.0:
            raise ValueError('VALIDATION_FRACTION needs to be a number between 0 and 1')
        elif LOAD_TRAIN_PORTION < 0.0 or LOAD_TRAIN_PORTION > 1.0:
            raise ValueError('LOAD_TRAIN_PORTION needs to be a number between 0 and 1')
        
        if self.datasource == 'mnist':
            self.num_classes = 10
            self.dataset = read_mnist_data(os.path.join(self.path_data, 'mnist_dataset'), STANDARDIZE_BOOL = STANDARDIZE_BOOL, LOAD_TRAIN_PORTION = LOAD_TRAIN_PORTION)
        elif self.datasource == 'mnist-2':
            orig_dataset = read_mnist_data(os.path.join(self.path_data, 'mnist_dataset'), STANDARDIZE_BOOL = STANDARDIZE_BOOL, LOAD_TRAIN_PORTION = 1.0 )
            self.dataset = construct_bin_dataset(orig_dataset)
        elif self.datasource == 'fashion_mnist':
            self.num_classes = 10
            self.dataset = read_mnist_data(os.path.join(self.path_data, 'fashion_mnist_dataset'), STANDARDIZE_BOOL = STANDARDIZE_BOOL, LOAD_TRAIN_PORTION = LOAD_TRAIN_PORTION)
        elif self.datasource == 'cifar-10':
            self.num_classes = 10
            self.dataset = read_cifar_10_data(os.path.join(self.path_data, 'cifar_10_dataset/cifar-10-batches-py'), STANDARDIZE_BOOL = STANDARDIZE_BOOL )
        elif self.datasource == 'cifar-100':
            self.dataset = read_cifar_100_data(self.path_data, 'cifar_100_dataset',  STANDARDIZE_BOOL = STANDARDIZE_BOOL )
        elif self.datasource == 'svhn':
            self.num_classes = 10
            self.dataset = read_svhn_data(os.path.join(self.path_data, 'svhn_dataset/'), STANDARDIZE_BOOL = STANDARDIZE_BOOL )        
        elif self.datasource == 'cifar-merged':

            dataset_cifar_10 = read_cifar_10_data(os.path.join(path_data, 'cifar_10_dataset/cifar-10-batches-py'), STANDARDIZE_BOOL = STANDARDIZE_BOOL )
            dataset_cifar_100 = read_cifar_100_data(path_data, 'cifar_100_dataset',  STANDARDIZE_BOOL = STANDARDIZE_BOOL )

            merged_train_input = onp.concatenate((dataset_cifar_10['train']['input'], dataset_cifar_100['train']['input']),axis=0)
            merged_test_input = onp.concatenate((dataset_cifar_10['test']['input'], dataset_cifar_100['test']['input']),axis=0)

            num_train = merged_train_input.shape[0]
            num_test = merged_test_input.shape[0]

#             dataset_merged_cifar = {}
            self.dataset = {'train': { 'input': merged_train_input, 'label': onp.zeros((num_train, 10))}, 'test': { 'input' : merged_test_input, 'label' : onp.zeros((num_test, 10))} }
        
        elif self.datasource == 'mnist-merged':

            dataset_mnist = read_mnist_data(os.path.join(self.path_data, 'mnist_dataset'), STANDARDIZE_BOOL = STANDARDIZE_BOOL, LOAD_TRAIN_PORTION = LOAD_TRAIN_PORTION)
            dataset_fashion_mnist = read_mnist_data(os.path.join(self.path_data, 'fashion_mnist_dataset'), STANDARDIZE_BOOL = STANDARDIZE_BOOL, LOAD_TRAIN_PORTION = LOAD_TRAIN_PORTION)

            merged_train_input = onp.concatenate((dataset_mnist['train']['input'], dataset_fashion_mnist['train']['input']),axis=0)
            merged_test_input = onp.concatenate((dataset_mnist['test']['input'], dataset_fashion_mnist['test']['input']),axis=0)

            num_train = merged_train_input.shape[0]
            num_test = merged_test_input.shape[0]

#             dataset_merged_cifar = {}
            self.dataset = {'train': { 'input': merged_train_input, 'label': onp.zeros((num_train, 10))}, 'test': { 'input' : merged_test_input, 'label' : onp.zeros((num_test, 10))} }

        else:
            raise NotImplementedError
            
            
        if self.datasource == 'mnist-2':
            self.dataset['val'] =  self.dataset['train']
        else:
            self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * VALIDATION_FRACTION ), self.rand) # validation = 10% percent of training data

                
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}
        
        
    def data_stream(self, batch_size):
        rng = npr.RandomState(0)
        train_images = self.dataset['train']['input']
        train_labels = self.dataset['train']['label']
        
        while True:
            perm = rng.permutation(self.num_example['train'])
#             num_complete_batches, leftover = divmod(self.num_example['train'], batch_size)
            num_batches, leftover = divmod(self.num_example['train'], batch_size)
#             num_batches = num_complete_batches + bool(leftover)
    
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    
#     def next_batch(self, batch_size):
#         return next(self.data_stream(batch_size))

    def split_dataset(self, source, target, number, rand):
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}
        
        