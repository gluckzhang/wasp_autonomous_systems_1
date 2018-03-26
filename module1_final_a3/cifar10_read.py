#class written based on and replicating input_data from tensorflow.examples.tutorials.mnist for CIFAR-10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle
import numpy as np
#uncomment this line if you want to be able to save a numpy array with shape w x h x 3 as a png file
#import png 
from scipy import ndimage

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


#uncomment this function if you want to be abe to save a numpy array with shape w x h x 3 as a png file
#def save_array_to_png(fname, y):
#
#    with open(fname, 'wb') as f:
#        z = (65535*((y - y.min())/y.ptp())).astype(np.uint16)
#        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16)
#        # Convert z to the Python list of lists expected by
#        # the png writer.
#        z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
#        writer.write(f, z2list)    

def read_cifar10(fname, one_hot=False):

    height = 32
    width = 32
    depth = 3
    
    with open(fname, 'rb') as fo:
        dict = cPickle.load(fo)
        ni = dict['data'].shape[0]
        images = dict['data']
        images = images.reshape(ni, depth, height, width)
        images = images.transpose(0, 2, 3, 1)
               
        labels = dict['labels']
        labels = np.asarray(labels)

        if one_hot:
            labels = dense_to_one_hot(labels, 10)
        
    return images, labels


def dense_to_one_hot(labels_dense, num_classes):
    
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 distort_batch=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            assert images.shape[3] == 3
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2] * images.shape[3])
 
        if dtype == dtypes.float32:
            #     Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)                
                
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._distort_batch = distort_batch

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    ## apply distortion to implement data-augmentation
    def distort_images(self, images):

        n = images.shape[0]
        dtype = images.dtype
        distorted_images = np.ndarray(shape=(n, 32, 32, 3), dtype=dtype)
        
        i = 0
        for image in images:
            im = np.copy(image)
            if np.random.random_sample() > .5:
                im = np.fliplr(im)

            shift = 8*np.random.random_sample((2, 1)) - 4
            im = ndimage.shift(im, (shift[0], shift[1], 0), mode='wrap')

            distorted_images[i, :, :, :] = im                
            i = i+1
 

        return distorted_images

    
    def next_distorted_batch(self, batch_size, shuffle=True):

        start = self._index_in_epoch
    
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]
            
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]

            #print('At start of an epoch')
            b_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            distorted_images = self.distort_images(b_images)

            return distorted_images, np.concatenate((labels_rest_part, labels_new_part), axis=0)
        
        else:
            
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            #print('In middle of an epoch')
            #print(self._images[start:end].shape)
            distorted_images = self.distort_images(self._images[start:end])            
            #return self._images[start:end], self._labels[start:end]
            return distorted_images, self._labels[start:end]
            
    def next_normal_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        
        start = self._index_in_epoch
    
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
            
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        
        else:
            
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


    def next_batch(self, batch_size, shuffle=True):

        if self._distort_batch:
            return self.next_distorted_batch(batch_size, shuffle=True)
        else:
            return self.next_normal_batch(batch_size, shuffle=True)    


def read_data_sets(data_dir, distort_train=False, one_hot=False, dtype=dtypes.float32, reshape=True, validation_size=5000, seed=None):

    
    base_fname = data_dir + 'data_batch_'
    train_fnames = [base_fname + str(i) for i in xrange(1, 6)]
    
    for f in train_fnames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    nb =  10000
    print('Reading the training images')
    train_images = np.empty([nb*5, 32, 32, 3])
    if one_hot:
        train_labels = np.empty([nb*5, 10])    
    else:
        train_labels = np.empty([nb*5])    
    for i in xrange(1, 6):
        f = base_fname + str(i)
        st = (i-1) * nb
        fin = st + nb
        train_images[st:fin, :, :, :], train_labels[st:fin, :] = read_cifar10(f, one_hot=one_hot)

    test_fname = data_dir + 'test_batch'
    
    print('Reading the test images')
    test_images = np.empty([nb, 32, 32, 3])
    #test_labels = np.empty([nb])    
    test_images, test_labels = read_cifar10(test_fname, one_hot=one_hot)
    #print(test_labels)


    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
            .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train_options = dict(dtype=dtype, reshape=reshape, seed=seed, distort_batch=distort_train)
    test_options = dict(dtype=dtype, reshape=reshape, seed=seed)

    
    train = DataSet(train_images, train_labels, **train_options)        
    validation = DataSet(validation_images, validation_labels, **test_options)
    test = DataSet(test_images, test_labels, **test_options)

    return base.Datasets(train=train, validation=validation, test=test)
