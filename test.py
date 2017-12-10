import matplotlib.pyplot as plt
from generate_data import *
from info_measures import *
from ml_algorithms import *
from nn_starter import *
from plots import *

# generates mixture of gaussian dataset and plots results with contours
def sample():
    print("=== SAMPLE AND PLOT ===")
    # setup
    n = 300
    # pi, mu, cov = binary_paramters()
    pi, mu, cov = complex_paramters()
    # sample & mesh
    X, Y = sample_gaussian(pi, mu, cov, n)
    pos, cond = mesh_gaussian(mu, cov, [-2,-2], [12,12], 1000)
    # plot 2D data
    plot_2d(X, Y, pos, cond)


# tests consitency of mutual information estimators
def consitency():
    print("=== MUTUAL INFORMATION ESTIMATION ===")
    # setup
    n, step, m = 2000,50,10
    pi, mu, cov = binary_paramters()
    size = np.arange(step, n + step, step)
    kde = np.zeros((m, int(n / step)))
    for i in range(m):
    	print("== %d of %d ==" % (i + 1, m))
        # sample
        X, Y = sample_gaussian(pi, mu, cov, 4 * n)
        X_y0 = X[Y == 0]
        X_y1 = X[Y == 1]
        # calculate mutual information
        for j in size:
            kde[i, int((j - step) / step)] = KDE_CD(X[0:j], Y[0:j])
            # kde[i, int((j - step) / step)] = KDE_CC(X_y0[0:j,:], X_y1[0:j,:])
            # kde[i, int((j - step) / step)] = I_BIN(X[0:j], Y[0:j])
    mean = np.mean(kde, axis=0)
    std = np.std(kde, axis=0)
    mi = I_TRUTH(pi, mu, cov, 10^8)
    truth = [mi for i in size]
    # plot estimates
    # plot_line(np.tile(size,(m, 1)), kde, 'KDE I(X,Y) Estimator')
    plot_area(size, truth, mean, std, 'KDE with CV Method')

# Perceptron
def perceptron_test():
    # setup
    print("=== PERCEPTRON ===")
    m = 300
    pi, mu, cov = binary_paramters()
    # sample
    print(" == sample data == ")
    X_trn, Y_trn = sample_gaussian(pi, mu, cov, m)
    X_tst, Y_tst = sample_gaussian(pi, mu, cov, m)
    X_trn = add_ones(X_trn)
    X_tst = add_ones(X_tst)
    # perceptron
    perceptron = PERCEPTRON(X_trn, Y_trn, X_tst, Y_tst)
    # train
    print(" == train == ")
    n_epoch = 100
    l_rate = 0.001
    batch = m
    lmbda = 0
    perceptron.train(l_rate, n_epoch, batch, lmbda)
    # plot
    print(" == plot == ")
    perceptron.plot()


# Logistic Regression
def logistic_test():
    # setup
    print("=== LOGISTIC REGRESSION ===")
    m = 100
    pi, mu, cov = binary_paramters()
    # sample
    print(" == sample data == ")
    X_trn, Y_trn = sample_gaussian(pi, mu, cov, m)
    X_tst, Y_tst = sample_gaussian(pi, mu, cov, m)
    # update data
    X_trn = square(X_trn)
    X_trn = add_ones(X_trn)
    X_tst = square(X_tst)
    X_tst = add_ones(X_tst)
    # logistic
    logistic = LOGISTIC(X_trn, Y_trn, X_tst, Y_tst)
    # train
    print(" == train == ")
    n_epoch = 100
    l_rate = 0.01
    batch = m
    lmbda = 0
    logistic.train(l_rate, n_epoch, batch, lmbda)
    # plot
    print(" == plot == ")
    logistic.plot()

# Softmax Regression
def softmax_test():
    # setup
    print("=== SOFTMAX REGRESSION ===")
    m = 300
    pi, mu, cov = complex_paramters()
    # sample
    print(" == sample data == ")
    X_trn, Y_trn = sample_gaussian(pi, mu, cov, m)
    X_tst, Y_tst = sample_gaussian(pi, mu, cov, m)
    # update data
    X_trn = square(X_trn)
    X_trn = add_ones(X_trn)
    X_tst = square(X_tst)
    X_tst = add_ones(X_tst)
    # logistic regression
    softmax = SOFTMAX(X_trn, Y_trn, X_tst, Y_tst)
    # train
    print(" == train == ")
    n_epoch = 100
    l_rate = 0.01
    batch = m
    lmbda = 0
    softmax.train(l_rate, n_epoch, batch, lmbda)
    # plot
    print(" == plot == ")
    softmax.plot()


# Support Vector Machines
def svm_test():
	# setup
    print("=== SUPPORT VECTOR MACHINE ===")
    m = 800
    pi, mu, cov = binary_paramters()
    # sample
    print(" == sample data == ")
    X_trn, Y_trn = sample_gaussian(pi, mu, cov, m)
    X_tst, Y_tst = sample_gaussian(pi, mu, cov, m)
    # update data
    # X_trn = square(X_trn)
    # X_tst = square(X_tst)
    # logistic regression
    svm = SVM(X_trn, Y_trn, X_tst, Y_tst)
    # train
    print(" == train == ")
    n_epoch = 100
    l_rate = 0.01
    batch = m
    lmbda = 0
    svm.train(l_rate, n_epoch, batch, lmbda)
    # plot
    print(" == plot == ")
    Ixy = 0.6#I_TRUTH(pi, mu, cov, m)
    #H = np.vectorize(entropy)
    Hy = 1.0#np.sum(H(pi))
    svm.plot(Ixy, Hy)



# main function of tests to run
def main():
    # sample()
    # consitency()
    # perceptron_test()
    logistic_test()
    # softmax_test()
    # svm_test()

if __name__ == '__main__':
    main()