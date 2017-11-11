import numpy as np
from info_measures import *
from plots import *

# Perceptron
class PERCEPTRON:

    def __init__(self, X_trn, Y_trn, X_tst, Y_tst):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.name = "Perceptron"

    # hypothesis function
    def hypothesis(self, theta, x):
    	return np.matmul(x,theta)

    # predict class
    def predict(self, h_theta):
    	return (np.sign(h_theta) + 1) / 2.0

    # gradient descent
    def gd(self, theta, x, err, l_rate, lmbda):
    	theta -= l_rate * np.squeeze(np.matmul(np.reshape(err, [1, -1]), x))
     
    # train algorithm
    def train(self, l_rate, n_epoch, batch, lmbda):
        m, n = self.X_trn.shape
        theta = np.zeros(n)
        I_xx = np.zeros((2, n_epoch))
        I_xy = np.zeros((2, n_epoch))
        Err_trn = np.zeros(n_epoch)
        for k in range(n_epoch):
        	# gradient descent
        	for i in range(0, m, batch):
        		x = self.X_trn[i:min(i + batch,m),:]
        		h = self.hypothesis(theta, x)
        		pred = self.predict(h)
        		err = pred - self.Y_trn[i:min(i + batch,m)]
        		self.gd(theta, x, err, (l_rate / float(batch)), lmbda)
        	# predict
        	h = self.hypothesis(theta, self.X_trn)
        	pred = self.predict(h)
        	# mutual information with input
        	I_xx[0, k] = KDE_CD(self.X_trn[:,1:], pred)
        	I_xx[1, k] = KDE_CC(self.X_trn[:,1:], np.reshape(h, (-1,1)))
        	# mutual information with output
        	I_xy[0, k] = EMPIRICAL_DD(pred, self.Y_trn)
        	I_xy[1, k] = KDE_CD(np.reshape(h, (-1,1)), self.Y_trn)
        	# training error
        	Err_trn[k] = np.sum(pred != self.Y_trn) / m
        self.I_xx = I_xx
        self.I_xy = I_xy
        self.Err_trn = Err_trn
        self.Epoch = np.arange(1, n_epoch + 1)
        self.Theta = theta

    # plot to Information Plane
    def plot(self):
        info_plane(self)

# Logistic Regression
class LOGISTIC:

    def __init__(self, X_trn, Y_trn, X_tst, Y_tst):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.name = "Logistic Regression"

    # hypothesis function
    def hypothesis(self, theta, x):
        return 1 / (1 + np.exp(-np.matmul(x,theta)))

    # class prediction
    def predict(self, h_theta):
    	return (np.sign(h_theta - 0.5) + 1) / 2

    # gradient descent
    def gd(self, theta, x, err, l_rate, lmbda):
        theta -= l_rate * (np.squeeze(np.matmul(np.reshape(err, [1, -1]), x)) + lmbda * np.sign(theta)) #L1
        # theta -= l_rate * (np.squeeze(np.matmul(np.reshape(err, [1, -1]), x)) - 2 * lmbda * theta) 	#L2

    # train
    def train(self, l_rate, n_epoch, batch, lmbda):
    	m, n = self.X_trn.shape
    	theta = np.zeros(n)
    	I_xx = np.zeros((2, n_epoch))
    	I_xy = np.zeros((2, n_epoch))
    	Err_trn = np.zeros(n_epoch)
    	for k in range(n_epoch):
    		# gradient descent
    		for i in range(0, m, batch):
    			x = self.X_trn[i:min(i + batch,m),:]
    			h = self.hypothesis(theta, x)
    			err = h - self.Y_trn[i:min(i + batch,m)]
    			self.gd(theta, x, err, (l_rate / float(batch)), lmbda)
    		# calculate hypothesis and prediction
    		h_theta = self.hypothesis(theta, self.X_trn)
    		pred = self.predict(h_theta)
    		# mutual information with input
    		I_xx[0, k] = KDE_CD(self.X_trn[:,1:], pred)
    		I_xx[1, k] = KDE_CC(self.X_trn[:,1:], np.reshape(h_theta, (-1,1)))
    		# mutual information with output
    		I_xy[0, k] = EMPIRICAL_DD(pred, self.Y_trn)
    		I_xy[1, k] = KDE_CD(np.reshape(h_theta, (-1,1)), self.Y_trn)
    		# training error
    		Err_trn[k] = np.sum(pred != self.Y_trn) / m
    	self.I_xx = I_xx
    	self.I_xy = I_xy
    	self.Err_trn = Err_trn
    	self.Epoch = np.arange(1, n_epoch + 1)
    	self.Theta = theta


    # plot to information plane
    def plot(self):
        info_plane(self)

# Softmax Logistic Regression (Syntax/Style needs to be upated to fit others)
class SOFTMAX:

    def __init__(self):
    	pass

    def gradient(self, theta, X_train, y_train, alpha):
    	theta -= alpha * np.matmul(np.transpose(X_train), (self.h_vec(theta, X_train) - y_train))

    def h_vec(self, theta, X):
    	eta = np.matmul(X, theta)
    	temp = np.exp(eta - np.reshape(np.amax(eta, axis=1), [-1, 1]))
    	return (temp / np.reshape(np.sum(temp, axis=1), [-1, 1]))

    def train(self, X_train, y_orig, alpha, max_iter, num_class):
    	y_train = np.zeros([len(y_orig), num_class])
    	y_train[np.arange(len(y_orig)), y_orig.astype(int)] = 1
    	theta = np.zeros((X_train.shape[1],num_class))
    	Ixx = np.zeros(max_iter)
    	Ixy = np.zeros(max_iter)
    	err = np.zeros(max_iter)
    	for i in range(max_iter):
    		pred = np.argmax(self.h_vec(theta, X_train), axis=1)
    		err[i] = np.sum(pred != y_orig) / len(y_train)
    		Ixx[i] = KDE(X_train[:,1:],pred)
    		Ixy[i] = EMPIRICAL_DD(pred, y_orig)
    		self.gradient(theta, X_train, y_train, alpha)
    	return Ixx, Ixy, err, theta

    # Plot to Information Plane
    def plot(self, I_xx, I_xy, E_train, epoch):
        fig, ax = plt.subplots()
        fig.suptitle("Softmax Regression in the Information Plane", fontsize="x-large")
        # information & epoch
        plt.subplot(1, 2, 1)
        plt.scatter(I_xx, I_xy, c=epoch, s=20, cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Epoch', rotation=270)
        ax.grid(True)
        plt.xlabel('I(X;X~)')
        plt.ylabel('I(X~;Y)')
        # information & training error
        plt.subplot(1, 2, 2)
        plt.scatter(I_xx, I_xy, c=E_train, s=20, cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Training Error', rotation=270)
        ax.grid(True)
        plt.xlabel('I(X;X~)')
        plt.ylabel('I(X~;Y)')
        plt.show()







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

# Support Vector Machine
class SVM:
    
    def __init__(self):
    	pass

