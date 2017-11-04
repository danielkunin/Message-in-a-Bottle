import numpy as np


# Logistic Regression
class LOGISTIC:

    def __init__():


# Linear Discriminant Analysis
class LDA:
    def __init__():


# Quadratic Discriminant Analysis
class QDA:

    def __init__():


# Multinomial Naive Bayes
class BAYES:

    def __init__():

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
    
    def __init__():
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
        print 'Error: %1.4f' % error

