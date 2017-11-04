import numpy as np
import matplotlib.pyplot as plt
from info_measures import *

# Perceptron (https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/)
class PERCEPTRON:

    def __init__(self):
        pass

    # Make a prediction with weights
    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0
     
    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self, train, l_rate, n_epoch):
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                prediction = predict(row, weights)
                error = row[-1] - prediction
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        return weights
     
    # Estimate Perceptron weights and record mutual information  
    def info_train(self, train, l_rate, n_epoch):
        weights = [0.0 for i in range(len(train[0]))]
        Ixx = np.zeros(n_epoch)
        Ixy = np.zeros(n_epoch)
        for epoch in range(n_epoch):
            # Calculate Mutual Information
            predictions = list()
            for row in train:
                prediction = self.predict(row, weights)
                predictions.append(prediction)
            data = np.zeros(train.shape)
            data[:,0:-1] = train[:,0:-1]
            data[:,-1] = predictions
            Ixx[epoch] = KDE(data)
            Ixy[epoch] = DISCRETE(predictions, train[:,-1])
            # Batch Gradient Descent
            for row in train:
                prediction = self.predict(row, weights)
                error = row[-1] - prediction
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        return Ixx, Ixy

    # Perceptron Algorithm With Stochastic Gradient Descent
    def perceptron(self, train, test, l_rate, n_epoch):
        predictions = list()
        weights = train_weights(train, l_rate, n_epoch)
        for row in test:
            prediction = predict(row, weights)
            predictions.append(prediction)
        return predictions

    # Plot to Information Plane
    def plot_IPlane(self,x,y,n):
        fig, ax = plt.subplots()
        plt.scatter(x,y,c=n,s=20)
        plt.colorbar()
        ax.grid(True)
        plt.xlabel('I(X;X~)')
        plt.ylabel('I(X~;Y)')
        plt.title("Perceptron in the Information Plane")
        plt.show()


# Logistic Regression
class LOGISTIC:

    def __init__(self):
        pass


# Linear Discriminant Analysis
class LDA:

    def __init__(self):
        pass


# Quadratic Discriminant Analysis
class QDA:

    def __init__(self):
        pass


# Multinomial Naive Bayes
class BAYES:

    def __init__(self):
        pass

    def readMatrix(file):
        fd = open(file, 'r')
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        return matrix, tokens, np.array(Y)

    def nb_train(matrix, category):
        state = {}
        M,N = matrix.shape
        theta = 0.
        total = [0.,0.]
        phi = np.ones((2,N))
        for i in range(M):
            ni = 0
            for j in range(N):
                ni += matrix[i,j]
                phi[category[i],j] += matrix[i,j]
            total[category[i]] += ni
            theta += category[i]
        phi[0,:] /= (total[0] + N)
        phi[1,:] /= (total[1] + N)
        state["phi"] =  phi
        state["theta"] =  theta / M
        return state

    def nb_test(matrix, state):
        M,N = matrix.shape
        output = np.zeros(M)
        theta = state["theta"]
        phi = state["phi"]
        for i in range(M):
            spam = np.log(theta)
            news = np.log(1. - theta)
            for j in range(N):
                spam += np.log(phi[1,j]) * matrix[i,j]
                news += np.log(phi[0,j]) * matrix[i,j]
            m = min(spam, news)
            logprob = spam - (m + np.log(np.exp(spam - m) + np.exp(news - m)))
            output[i] = np.exp(logprob) > 0.5
        return output

    def evaluate(output, label, name):
        error = (output != label).sum() * 1. / len(output)
        print('Error for ' + name + ': %1.4f' % error)
        return error

# Support Vector Machine
class SVM:
    
    def __init__(self):
        np.random.seed(123)
        tau = 8.

    def readMatrix(file):
        fd = open(file, 'r')
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        category = (np.array(Y) * 2) - 1
        return matrix, tokens, category

    def svm_train(matrix, category):
        state = {}
        M, N = matrix.shape
        Y = category
        matrix = 1. * (matrix > 0)
        squared = np.sum(matrix * matrix, axis=1)
        gram = matrix.dot(matrix.T)
        K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

        alpha = np.zeros(M)
        alpha_avg = np.zeros(M)
        L = 1. / (64 * M)
        outer_loops = 40

        alpha_avg
        for ii in xrange(outer_loops * M):
            i = int(np.random.rand() * M)
            margin = Y[i] * np.dot(K[i, :], alpha)
            grad = M * L * K[:, i] * alpha[i]
            if (margin < 1):
                grad -=  Y[i] * K[:, i]
            alpha -=  grad / np.sqrt(ii + 1)
            alpha_avg += alpha

        alpha_avg /= (ii + 1) * M

        state['alpha'] = alpha
        state['alpha_avg'] = alpha_avg
        state['Xtrain'] = matrix
        state['Sqtrain'] = squared
        return state

    def svm_test(matrix, state):
        M, N = matrix.shape
        output = np.zeros(M)
        Xtrain = state['Xtrain']
        Sqtrain = state['Sqtrain']
        matrix = 1. * (matrix > 0)
        squared = np.sum(matrix * matrix, axis=1)
        gram = matrix.dot(Xtrain.T)
        K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
        alpha_avg = state['alpha_avg']
        preds = K.dot(alpha_avg)
        output = np.sign(preds)
        return output

    def evaluate(output, label):
        error = (output != label).sum() * 1. / len(output)
        print('Error: %1.4f' % error)

