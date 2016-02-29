#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function
import numpy as np
import six
import chainer
from chainer import computational_graph
import chainer.links as L
from chainer import optimizers
import matplotlib.pyplot as plt

import data
import net

batchsize = 100
n_epoch = 20
n_units = 100

print('# unit: {}'.format(n_units))
print('# Minibatch-size: {}'.format(batchsize))
print('# epoch: {}'.format(n_epoch))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# Prepare multi-layer perceptron model, defined in net.py
model = L.Classifier(net.MnistMLP(''' TODO: define the correct netwoek architecture '''))

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

epochs = six.moves.range(1, n_epoch + 1)
# Learning loop
for epoch in epochs:
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = chainer.Variable(np.asarray(x_train[''' TODO: pick the correct indices from perm array ''']))
        y_batch = chainer.Variable(np.asarray(y_train[''' TODO: pick the correct indices from perm array ''']))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x_batch, y_batch)

        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph((model.loss, ))
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(y_batch.data)
        sum_accuracy += float(model.accuracy.data) * len(y_batch.data)

    loss = sum_loss / N
    accuracy = sum_accuracy / N
    train_loss.append(loss)
    train_accuracy.append(accuracy)
    print('train mean loss={}, accuracy={}'.format(loss, accuracy))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = chainer.Variable(np.asarray(x_test[i:i + batchsize]), volatile='on')
        y_batch = chainer.Variable(np.asarray(y_test[i:i + batchsize]), volatile='on')
        loss = model(x_batch, y_batch)
        
        sum_loss += ''' TODO: update the sum loss variable with the loss value for the batch '''
        sum_accuracy += '''TODO: update the sum accuracy variable with the loss value for the batch'''

    loss = sum_loss / N_test
    accuracy = sum_accuracy / N_test
    test_loss.append(loss)
    test_accuracy.append(accuracy)
    print('test  mean loss={}, accuracy={}'.format(loss, accuracy))

# Plot the results
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train loss')
plt.plot(epochs, test_loss, label='Test loss')
plt.xlabel("Epoch")
plt.ylabel("Loss function")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Train accuracy')
plt.plot(epochs, test_accuracy, label='Test accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
