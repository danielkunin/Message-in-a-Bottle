import numpy as np
from info_measures import *
from plots import *

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    exp = np.exp(x - np.reshape(np.max(x, axis=0), [1, -1]))
    s = exp / np.reshape(np.sum(exp, axis=0), [1, -1])
    return s

def sigmoid(x):
    s = 1. / (1. + np.exp(-x))
    return s

def forward_prop(data, labels, params):

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    z1 = W1.dot(data.T) + b1.reshape([-1, 1])
    h = sigmoid(z1)
    z2 = W2.dot(h) + b2.reshape([-1, 1])
    y = softmax(z2)
    cost = - 1. / (1. * data.shape[0]) * np.sum(labels.T * np.log(y))

    return h, y, cost

def backward_prop(x, a1, a2, y, params):

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']


    lmbda = 0.0001
    dz2 = 1. / (1. * x.shape[0]) * (a2 - y.T)
    gradW2 = dz2.dot(a1.T) #+ 2 * lmbda * W2
    gradb2 = np.sum(dz2, axis=1)
    dz1 = W2.T.dot(dz2) * a1 * (1. - a1)
    gradW1 = dz1.dot(x) #+ 2 * lmbda * W1
    gradb1 = np.sum(dz1, axis=1)


    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    params = {}

    # Constants
    B = 1000
    num_epoch = 30
    epochs = np.arange(num_epoch)
    costs = np.zeros((2,num_epoch))
    accuracy = np.zeros((2,num_epoch))
    I_xx = np.zeros((2, num_epoch))
    I_xy = np.zeros((2, num_epoch))
    # Initialize Parameters
    params['W1'] = np.random.randn(num_hidden, n)
    params['b1'] = np.zeros(num_hidden)
    params['W2'] = np.random.randn(10, num_hidden)
    params['b2'] = np.zeros(10)
    # GD
    for epoch in epochs:
        print('Epoch %d' % (epoch + 1))
        # Mini Batch
        for i in range(50):
            x = trainData[i * B:min((i + 1) * B, m),:]
            y = trainLabels[i * B:min((i + 1) * B, m),:]
            a1, a2, _ = forward_prop(x, y, params)
            grad = backward_prop(x, a1, a2, y, params)
            params['W1'] -=  learning_rate * grad['W1']
            params['b1'] -=  learning_rate * grad['b1']
            params['W2'] -=  learning_rate * grad['W2']
            params['b2'] -=  learning_rate * grad['b2']
        # Training Set
        a1, a2, costs[0, epoch] = forward_prop(trainData, trainLabels, params)
        accuracy[0, epoch] = compute_accuracy(a2, trainLabels)
        # mutual information with input
        I_xx[0, epoch] = KDE_CC(trainData, a1.T)
        I_xx[1, epoch] = KDE_CC(trainData, a2.T)
        # I_xx[1, epoch] = KDE_CC(trainData, np.reshape(np.max(a2, axis=0), (-1,1)))
        # mutual information with output
        I_xy[0, epoch] = KDE_CD(a1.T, trainLabels)
        I_xy[0, epoch] = KDE_CD(a2.T, trainLabels)
        # I_xy[1, epoch] = KDE_CD(np.reshape(np.max(a2, axis=0), (-1,1)), trainLabels)
        # Dev Set
        a1, a2, costs[1, epoch] = forward_prop(devData, devLabels, params)
        accuracy[1, epoch] = compute_accuracy(a2, devLabels)
    # Plot
    info_plane(I_xx[0,:], I_xy[0,:], epochs, "Hidden Layer")
    info_plane(I_xx[1,:], I_xy[1,:], epochs, "Output Layer")
    info_curve(epochs, I_xx[0,:], I_xy[0,:], "Hidden Layer")
    info_curve(epochs, I_xx[1,:], I_xy[1,:], "Output Layer")

    return params

def test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=0) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

