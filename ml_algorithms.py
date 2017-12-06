import numpy as np
from info_measures import *
from plots import *
from sklearn.svm import SVC

# Perceptron
class PERCEPTRON:

    def __init__(self, X_trn, Y_trn, X_tst, Y_tst):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.name = "Perceptron"

    def hypothesis(self, theta, x):
    	return np.matmul(x,theta)

    def predict(self, h_theta):
    	return (np.sign(h_theta) + 1) / 2.0

    def gd(self, theta, x, err, l_rate, lmbda):
    	theta -= l_rate * np.squeeze(np.matmul(np.reshape(err, [1, -1]), x))
     
    def train(self, l_rate, n_epoch, batch, lmbda):
        m, n = self.X_trn.shape
        theta = np.zeros(n)
        I_xx = np.zeros((2, n_epoch))
        I_xy = np.zeros((2, n_epoch))
        Err = np.zeros((2, n_epoch))
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
        	p_tst = self.predict(self.hypothesis(theta, self.X_tst))
        	# mutual information with input
        	I_xx[0, k] = KDE_CD(self.X_trn[:,1:], pred)
        	I_xx[1, k] = KDE_CC(self.X_trn[:,1:], np.reshape(h, (-1,1)))
        	# mutual information with output
        	I_xy[0, k] = EMPIRICAL_DD(pred, self.Y_trn)
        	I_xy[1, k] = KDE_CD(np.reshape(h, (-1,1)), self.Y_trn)
        	# training error
        	Err[0, k] = np.sum(pred != self.Y_trn) / (m * 1.0)
        	Err[1, k] = np.sum(p_tst != self.Y_tst) / (self.X_tst.shape[0] * 1.0)
        self.I_xx = I_xx
        self.I_xy = I_xy
        self.Err = np.log2(Err)
        self.Epoch = np.arange(1, n_epoch + 1)
        self.Theta = theta

    def plot(self):
        info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        info_curve(self.Epoch, self.I_xx[0,:], self.I_xy[0,:], self.name)
        error_plane(self.Err[0,:], self.Err[1,:], self.Epoch, self.name)
        error_curve(self.Epoch, self.Err[0,:], self.Err[1,:], self.name)

# Logistic Regression
class LOGISTIC:

    def __init__(self, X_trn, Y_trn, X_tst, Y_tst):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.name = "Logistic Regression"

    def hypothesis(self, theta, x):
        return 1 / (1 + np.exp(-np.matmul(x,theta)))

    def predict(self, h_theta):
    	return (np.sign(h_theta - 0.5) + 1) / 2

    def gd(self, theta, x, err, l_rate, lmbda):
        theta -= l_rate * (np.squeeze(np.matmul(np.reshape(err, [1, -1]), x)) + lmbda * np.sign(theta)) #L1
        # theta -= l_rate * (np.squeeze(np.matmul(np.reshape(err, [1, -1]), x)) - 2 * lmbda * theta) 	#L2

    def train(self, l_rate, n_epoch, batch, lmbda):
    	m, n = self.X_trn.shape
    	theta = np.zeros(n)
    	I_xx = np.zeros((2, n_epoch))
    	I_xy = np.zeros((2, n_epoch))
    	Err = np.zeros((2, n_epoch))
    	for k in range(n_epoch):
    		# gradient descent
    		for i in range(0, m, batch):
    			x = self.X_trn[i:min(i + batch,m),:]
    			h = self.hypothesis(theta, x)
    			err = h - self.Y_trn[i:min(i + batch,m)]
    			self.gd(theta, x, err, (l_rate / float(batch)), lmbda)
    		# calculate hypothesis and prediction
    		h_trn = self.hypothesis(theta, self.X_trn)
    		p_trn = self.predict(h_trn)
    		p_tst = self.predict(self.hypothesis(theta, self.X_tst))
    		# mutual information with input
    		I_xx[0, k] = KDE_CD(self.X_trn[:,1:], p_trn)
    		I_xx[1, k] = KDE_CC(self.X_trn[:,1:], np.reshape(h_trn, (-1,1)))
    		# mutual information with output
    		I_xy[0, k] = EMPIRICAL_DD(p_trn, self.Y_trn)
    		I_xy[1, k] = KDE_CD(np.reshape(h_trn, (-1,1)), self.Y_trn)
    		# training and testing error
    		Err[0, k] = np.sum(p_trn != self.Y_trn) / (m * 1.0)
    		Err[1, k] = np.sum(p_tst != self.Y_tst) / (self.X_tst.shape[0] * 1.0)
    	self.I_xx = I_xx
    	self.I_xy = I_xy
    	self.Err = np.log2(Err)
    	self.Epoch = np.arange(1, n_epoch + 1)
    	self.Theta = theta


    def plot(self):
        info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        info_curve(self.Epoch, self.I_xx[0,:], self.I_xy[0,:], self.name)
        error_plane(self.Err[0,:], self.Err[1,:], self.Epoch, self.name)
        error_curve(self.Epoch, self.Err[0,:], self.Err[1,:], self.name)

# Softmax Logistic Regression (Syntax/Style needs to be upated to fit others)
class SOFTMAX:

    def __init__(self, X_trn, Y_trn, X_tst, Y_tst):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.n_class = np.unique(Y_tst).size
        self.name = "Softmax Regression"

    def hypothesis(self, theta, x):
        eta = np.matmul(x, theta)
    	temp = np.exp(eta - np.reshape(np.amax(eta, axis=1), [-1, 1]))
    	return (temp / np.reshape(np.sum(temp, axis=1), [-1, 1]))

    def predict(self, h_theta):
    	return np.argmax(h_theta, axis=1)

    def gd(self, theta, x, err, l_rate, lmbda):
    	theta -= l_rate * np.matmul(np.transpose(x), err)

    def train(self, l_rate, n_epoch, batch, lmbda):
    	# Setup Vectors
        Y_trn = np.zeros([len(self.Y_trn), self.n_class])
    	Y_trn[np.arange(len(self.Y_trn)), self.Y_trn.astype(int)] = 1
    	Y_tst = np.zeros([len(self.Y_tst), self.n_class])
    	Y_tst[np.arange(len(self.Y_tst)), self.Y_tst.astype(int)] = 1
    	# Setup Arrays
    	m, n = self.X_trn.shape
    	theta = np.zeros((n, self.n_class))
        I_xx = np.zeros((2, n_epoch))
        I_xy = np.zeros((2, n_epoch))
        Err = np.zeros((2, n_epoch))
        for k in range(n_epoch):
    		# gradient descent
    		for i in range(0, m, batch):
    			x = self.X_trn[i:min(i + batch,m),:]
    			h = self.hypothesis(theta, x)
    			err = h - Y_trn[i:min(i + batch,m)]
    			self.gd(theta, x, err, (l_rate / float(batch)), lmbda)
    		# calculate hypothesis and prediction
    		h_trn = self.hypothesis(theta, self.X_trn)
    		p_trn = self.predict(h_trn)
    		p_tst = self.predict(self.hypothesis(theta, self.X_tst))
    		# mutual information with input
    		I_xx[0, k] = KDE_CD(self.X_trn[:,1:], p_trn)
    		I_xx[1, k] = KDE_CC(self.X_trn[:,1:], np.reshape(np.max(h_trn, axis=1), (-1,1)))
    		# mutual information with output
    		I_xy[0, k] = EMPIRICAL_DD(p_trn, self.Y_trn)
    		I_xy[1, k] = KDE_CD(np.reshape(np.max(h_trn, axis=1), (-1,1)), self.Y_trn)
    		# training and testing error
    		Err[0, k] = np.sum(p_trn != self.Y_trn) / (m * 1.0)
    		Err[1, k] = np.sum(p_tst != self.Y_tst) / (self.X_tst.shape[0] * 1.0)
    	self.I_xx = I_xx
        self.I_xy = I_xy
        self.Err = np.log2(Err)
        self.Epoch = np.arange(1, n_epoch + 1)
        self.Theta = theta

    def plot(self):
        info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        info_curve(self.Epoch, self.I_xx[0,:], self.I_xy[0,:], self.name)
        error_plane(self.Err[0,:], self.Err[1,:], self.Epoch, self.name)
        error_curve(self.Epoch, self.Err[0,:], self.Err[1,:], self.name)


# Support Vector Machine
class SVM:
    
    def __init__(self, X_trn, Y_trn, X_tst, Y_tst):
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self.name = "Support Vector Machine"

    #def train(self, l_rate, n_epoch, batch, lmbda):
    def train(self, l_rate, n_epoch, batch, lmbda):

        I_xx = np.zeros((2, n_epoch))
        I_xy = np.zeros((2, n_epoch))
        # Err = np.zeros((2, n_epoch))

        matrix = self.X_trn
        category = self.Y_trn

        tau = 8.
        state = {}
        M, N = matrix.shape
        Y = 2 * category - 1 #is -1 or 1
        squared = np.sum(matrix * matrix, axis=1)
        gram = matrix.dot(matrix.T)
        K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

        alpha = np.zeros(M)
        alpha_avg = np.zeros(M)
        # L = 1. / (64 * M)
        # n_epoch = 40 #n_epochs

        for ii in np.arange(n_epoch):
            for indx, i in enumerate(np.random.permutation(np.arange(M))):
                margin = Y[i] * np.dot(K[i, :], alpha)
                grad = M * l_rate * K[:, i] * alpha[i]
                if (margin < 1):
                    grad -=  Y[i] * K[:, i]
                alpha -=  grad / np.sqrt(M * ii + indx + 1)
                alpha_avg += alpha
            # Determine MI for training
            alpha_avg /= (M * ii + indx + 1) * M
            preds = K.dot(alpha_avg)
            output = np.sign(preds)
            # print(np.sum(output == Y) / (1. * M))

            # mutual information with input
            I_xx[0, ii] = KDE_CD(matrix, output)
            I_xx[1, ii] = KDE_CC(matrix, np.reshape(preds, (-1,1)))
            # I_xx[1, ii] = KDE_CC(self.X_trn[:,1:], np.reshape(np.max(h_trn, axis=1), (-1,1)))

            # mutual information with output
            I_xy[0, ii] = EMPIRICAL_DD(output, Y)
            I_xy[1, ii] = KDE_CD(np.reshape(preds,[-1,1]), Y)
            # I_xy[1, k] = KDE_CD(np.reshape(np.max(h_trn, axis=1), (-1,1)), self.Y_trn)

            alpha_avg *= (M * ii + indx + 1) * M

        alpha_avg /= M * M * n_epoch

        state['alpha'] = alpha
        state['alpha_avg'] = alpha_avg
        state['Xtrain'] = matrix
        state['Sqtrain'] = squared

        print("Our Training Accuracy:")
        print(np.sum(output == Y) / (1. * M))

        print("SKLEARN Training Accuracy:")
        clf = SVC()
        clf.fit(matrix, Y)
        print(clf.score(matrix,Y))

        # tol = 0.05
        # print("Alpha Avg")
        # print(np.sum(alpha_avg > tol))
        # plt.hist(alpha_avg, bins=100)
        # plt.show()

        info_plane(I_xx[0,:], I_xy[0,:], np.arange(n_epoch), "SVM")
        info_curve(np.arange(n_epoch), I_xx[0,:], I_xy[0,:], "SVM")

        info_plane(I_xx[1,:], I_xy[1,:], np.arange(n_epoch), "SVM")
        info_curve(np.arange(n_epoch), I_xx[1,:], I_xy[1,:], "SVM")


        return state
  #   def svm_test(matrix, state):
		# M, N = matrix.shape
		# output = np.zeros(M)
		# Xtrain = state['Xtrain']
		# Sqtrain = state['Sqtrain']
		# matrix = 1. * (matrix > 0)
		# squared = np.sum(matrix * matrix, axis=1)
		# gram = matrix.dot(Xtrain.T)
		# K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
		# alpha_avg = state['alpha_avg']
		# preds = K.dot(alpha_avg)
		# output = np.sign(preds)
		# return output

