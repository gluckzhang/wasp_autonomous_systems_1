# a CNN + 2-layer neural network with momentum training to classify cifar10 images use the tensorflow library

import numpy as np
import tensorflow as tf
import os
# class written to replicate input_data from tensorflow.examples.tutorials.mnist for CIFAR-10
import cifar10_read

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal(shape=[in_size, out_size], stddev=0.01))
    biases = tf.Variable(tf.constant(0.1, shape=[out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def weight_variable(shape):
    initial = tf.Variable(tf.random_normal(shape, stddev=0.1))
    return initial

def bias_variable(shape):
    initial = tf.Variable(tf.constant(0.1, shape=shape))
    return initial

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# location of the CIFAR-10 dataset
#CHANGE THIS PATH TO THE LOCATION OF THE CIFAR-10 dataset on your local machine
data_dir = 'datasets/cifar-10-batches-py/'

# read in the dataset
print('reading in the CIFAR10 dataset')
dataset = cifar10_read.read_data_sets(data_dir, one_hot=True, reshape=False)

using_tensorboard = True

##################################################
# PHASE 1  - ASSEMBLE THE GRAPH

# 1.1) define the placeholders for the input data and the ground truth labels

# x_input can handle an arbitrary number of input vectors of length input_dim = d
# y_  are the labels (each label is a length 10 one-hot encoding) of the inputs in x_input
# If x_input has shape [N, input_dim] then y_ will have shape [N, 10]
# original image size: 32 x 32 x 3
x_input = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
# keep_prob = tf.placeholder(tf.float32)

# 1.2) define the parameters of the network
## layer1: conv ##
W_conv1 = weight_variable([5, 5, 3, 64]) # patch 5 x 5 x 3, output 64
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1) # output 32 x 32 x 64
h_norm1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_pool1 = max_pool_2x2(h_norm1) # output 16 x 16 x 64

## layer2: conv ##
W_conv2 = weight_variable([5, 5, 64, 128]) # patch 5 x 5 x 64, output 128
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output 16 x 16 x 128
h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_pool2 = max_pool_2x2(h_norm2) # output 8 x 8 x 128

## layer3, layer4: fully connected layer ##
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*128]) # change [8, 8, 128] into [8 x 8 x 128]
h_fc1 = add_layer(h_pool2_flat, 8*8*128, 384, activation_function=tf.nn.relu)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = add_layer(h_fc1, 384, 192, activation_function=tf.nn.relu)

## layer5: prediction ##
y = add_layer(h_fc2, 192, 10, activation_function=None)

# 1.3) define the loss funtion
# cross entropy loss:
# Apply softmax to each output vector in y to give probabilities for each class then compare to the ground truth labels via the cross-entropy loss and then compute the average loss over all the input examples
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 1.4) Define the optimizer used when training the network ie gradient descent or some variation.
# Use AdamOptimizer with a learning rate of .01
learning_rate = .01
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# (optional) definiton of performance measures
# definition of accuracy, count the number of correct predictions where the predictions are made by choosing the class with highest score
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 1.6) Add an op to initialize the variables.
init = tf.global_variables_initializer()

##################################################

# If using TENSORBOARD
if using_tensorboard:
    # keep track of the loss and accuracy for the training set
    tf.summary.scalar('training loss', cross_entropy, collections=['training'])
    tf.summary.scalar('training accuracy', accuracy, collections=['training'])
    # merge the two quantities
    tsummary = tf.summary.merge_all('training')

    # keep track of the loss and accuracy for the validation set
    tf.summary.scalar('validation loss', cross_entropy, collections=['validation'])
    tf.summary.scalar('validation accuracy', accuracy, collections=['validation'])
    # merge the two quantities
    vsummary = tf.summary.merge_all('validation')

##################################################


##################################################
# PHASE 2  - PERFORM COMPUTATIONS ON THE GRAPH

n_iter = 2000

# 2.1) start a tensorflow session
with tf.Session() as sess:

    ##################################################
    # If using TENSORBOARD
    if using_tensorboard:
        # set up a file writer and directory to where it should write info +
        # attach the assembled graph
        summary_writer = tf.summary.FileWriter('tensorboard/network3', sess.graph)
    ##################################################

    # 2.2)  Initialize the network's parameter variables
    # Run the "init" op (do this when training from a random initialization)
    sess.run(init)

    # 2.3) loop for the mini-batch training of the network's parameters
    for i in range(n_iter):

        # grab a random batch (size nbatch) of labelled training examples
        nbatch = 200
        batch = dataset.train.next_batch(nbatch)

        # create a dictionary with the batch data
        # batch data will be fed to the placeholders for inputs "x_input" and labels "y_"
        batch_dict = {
            x_input: batch[0], # input data
            y_: batch[1] # corresponding labels
            # keep_prob: 0.8 # dropping
        }

        # run an update step of mini-batch by calling the "train_step" op
        # with the mini-batch data. The network's parameters will be updated after applying this operation
        sess.run(train_step, feed_dict=batch_dict)

        # periodically evaluate how well training is going
        if i % 50 == 0:

            # compute the performance measures on the training set by
            # calling the "cross_entropy" loss and "accuracy" ops with the training data fed to the placeholders "x_input" and "y_"

            tr = sess.run([cross_entropy, accuracy], feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000]})

            # compute the performance measures on the validation set by
            # calling the "cross_entropy" loss and "accuracy" ops with the validation data fed to the placeholders "x_input" and "y_"

            val = sess.run([cross_entropy, accuracy], feed_dict={x_input:dataset.validation.images[1:5000], y_:dataset.validation.labels[1:5000]})

            info = [i] + tr + val
            print(info)

            ##################################################
            # If using TENSORBOARD
            if using_tensorboard:

                # compute the summary statistics and write to file
                summary_str = sess.run(tsummary, feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000]})
                summary_writer.add_summary(summary_str, i)

                summary_str1 = sess.run(vsummary, feed_dict = {x_input:dataset.validation.images[1:5000], y_: dataset.validation.labels[1:5000]})
                summary_writer.add_summary(summary_str1, i)
            ##################################################

    # evaluate the accuracy of the final model on the test data
    test_acc = sess.run(accuracy, feed_dict={x_input: dataset.test.images, y_: dataset.test.labels})
    final_msg = 'test accuracy:' + str(test_acc)
    print(final_msg)

##################################################
