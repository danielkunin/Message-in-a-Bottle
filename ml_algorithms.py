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
        self.Err = Err
        self.Epoch = np.arange(1, n_epoch + 1)
        self.Theta = theta

    def plot(self):
        # info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        # info_plane2(self.I_xx[0,:], self.I_xy[0,:], self.Err[0,:], self.name)
        info_plane(self.I_xx[1,:], self.I_xy[1,:], self.Epoch, self.name)
        info_plane2(self.I_xx[1,:], self.I_xy[1,:], self.Err[0,:], self.name)
        # info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        # info_curve(self.Epoch, self.I_xx[0,:], self.I_xy[0,:], self.name)
        # error_plane(self.Err[0,:], self.Err[1,:], self.Epoch, self.name)
        # error_curve(self.Epoch, self.Err[0,:], self.Err[1,:], self.name)

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
        # theta -= l_rate * (np.squeeze(np.matmul(np.reshape(err, [1, -1]), x)) - 2 * lmbda * theta)    #L2

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
        self.Err = Err
        self.Epoch = np.arange(1, n_epoch + 1)
        self.Theta = theta


    def plot(self):
        # info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        # info_plane2(self.I_xx[0,:], self.I_xy[0,:], self.Err[0,:], self.name)
        info_plane(self.I_xx[1,:], self.I_xy[1,:], self.Epoch, self.name)
        info_plane2(self.I_xx[1,:], self.I_xy[1,:], self.Err[0,:], self.name)
        # info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        # info_curve(self.Epoch, self.I_xx[0,:], self.I_xy[0,:], self.name)
        # error_plane(self.Err[0,:], self.Err[1,:], self.Epoch, self.name)
        # error_curve(self.Epoch, self.Err[0,:], self.Err[1,:], self.name)

# Softmax Regression
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
            # I_xx[1, k] = KDE_CC(self.X_trn[:,1:], h_trn)
            I_xx[1, k] = KDE_CC(self.X_trn[:,1:], np.reshape(np.max(h_trn, axis=1), (-1,1)))
            # mutual information with output
            I_xy[0, k] = EMPIRICAL_DD(p_trn, self.Y_trn)
            # I_xy[1, k] = KDE_CD(h_trn, self.Y_trn)
            I_xy[1, k] = KDE_CD(np.reshape(np.max(h_trn, axis=1), (-1,1)), self.Y_trn)
            # training and testing error
            Err[0, k] = np.sum(p_trn != self.Y_trn) / (m * 1.0)
            Err[1, k] = np.sum(p_tst != self.Y_tst) / (self.X_tst.shape[0] * 1.0)
        self.I_xx = I_xx
        self.I_xy = I_xy
        self.Err = Err
        self.Epoch = np.arange(1, n_epoch + 1)
        self.Theta = theta

    def plot(self):
        info_plane(self.I_xx[1,:], self.I_xy[1,:], self.Epoch, self.name)
        info_curve(self.Epoch, self.I_xx[1,:], self.I_xy[1,:], self.name)
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

    def train(self, l_rate, n_epoch, batch, lmbda):
        # Setup Arrays
        I_xx = np.zeros((2, n_epoch))
        I_xy = np.zeros((2, n_epoch))
        Err = np.zeros((2, n_epoch))
        # Get Traning and Test Matrix
        X_trn = self.X_trn
        X_tst = self.X_tst
        # Convert response from (0,1) to (-1,1)
        Y_trn = 2 * self.Y_trn - 1
        Y_tst = 2 * self.Y_tst - 1
        # Constants
        M, N = X_trn.shape
        tau = 8.
        # Create Training Kernel
        sqr_trn = np.sum(X_trn * X_trn, axis=1)
        gram_trn = X_trn.dot(X_trn.T)
        K_trn = np.exp(-(sqr_trn.reshape((1, -1)) + sqr_trn.reshape((-1, 1)) - 2 * gram_trn) / (2 * (tau ** 2)))
        # Create Test Kernel
        sqr_tst = np.sum(X_tst * X_tst, axis=1)
        gram_tst = X_tst.dot(X_trn.T)
        K_tst = np.exp(-(sqr_tst.reshape((1, -1)) + sqr_trn.reshape((-1, 1)) - 2 * gram_tst) / (2 * (tau ** 2)))
        # Setup Coeficient Parameters
        alpha = np.zeros(M)
        alpha_avg = np.zeros(M)
        # gradient descent
        for k in np.arange(n_epoch):
            for indx, i in enumerate(np.random.permutation(np.arange(M))):
                margin = Y_trn[i] * np.dot(K_trn[i, :], alpha)
                grad = M * l_rate * K_trn[:, i] * alpha[i]
                if (margin < 1):
                    grad -=  Y_trn[i] * K_trn[:, i]
                alpha -=  grad / np.sqrt(M * k + indx + 1)
                alpha_avg += alpha
            # Determine MI for training
            alpha_avg /= (M * k + indx + 1) * M
            preds = K_trn.dot(alpha_avg)
            p_trn = np.sign(preds)
            p_tst = np.sign(K_tst.dot(alpha_avg))
            # mutual information with input
            I_xx[0, k] = KDE_CD(X_trn, p_trn)
            I_xx[1, k] = KDE_CC(X_trn, np.reshape(preds, (-1,1)))
            # mutual information with output
            I_xy[0, k] = EMPIRICAL_DD(p_trn, Y_trn)
            I_xy[1, k] = KDE_CD(np.reshape(preds,[-1,1]), Y_trn)
            # training and testing error
            Err[0, k] = np.sum(p_trn != Y_trn) / (M * 1.0)
            Err[1, k] = np.sum(p_tst != Y_tst) / (X_tst.shape[0] * 1.0)
            alpha_avg *= (M * k + indx + 1) * M
        # normalize
        alpha_avg /= M * M * n_epoch

        # Check SVM training accuracy against SKLEARN
        print("Final Training Accuracy: %f" % (np.sum(p_trn == Y_trn) / (1. * M)))
        clf = SVC()
        clf.fit(X_trn, Y_trn)
        print("SKLEARN Training Accuracy: %f" % clf.score(X_trn,Y_trn))


        self.I_xx = I_xx
        self.I_xy = I_xy
        self.Err = Err
        self.Epoch = np.arange(1, n_epoch + 1)

    def plot(self):
        # info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        # info_plane2(self.I_xx[0,:], self.I_xy[0,:], self.Err[0,:], self.name)
        info_plane(self.I_xx[1,:], self.I_xy[1,:], self.Epoch, self.name)
        info_plane2(self.I_xx[1,:], self.I_xy[1,:], self.Err[0,:], self.name)
        # info_plane(self.I_xx[0,:], self.I_xy[0,:], self.Epoch, self.name)
        # info_plane(self.I_xx[1,:], self.I_xy[1,:], self.Epoch, self.name)
        # info_curve(self.Epoch, self.I_xx[0,:], self.I_xy[0,:], self.name)
        # info_curve(self.Epoch, self.I_xx[1,:], self.I_xy[1,:], self.name)
        # error_plane(self.Err[0,:], self.Err[1,:], self.Epoch, self.name)
        # error_curve(self.Epoch, self.Err[0,:], self.Err[1,:], self.name)
        # ratio_curve(self.Epoch, self.I_xx[0,:] / Ixy, self.Err[0,:], self.name)
        # ratio_curve(self.Epoch, self.I_xy[0,:] / Hy, self.Err[0,:], self.name)
