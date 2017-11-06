import matplotlib.pyplot as plt
from generate_data import *
from info_measures import *
from ml_algorithms import *
from IB import *


# plots first two dim dataset (data) and contour lines (pos, cond)
def plot_2d(data, pos, cond):
	fig, ax = plt.subplots()
	for p in cond:
		plt.contour(pos[:,:,0], pos[:,:,1], p)
	plt.scatter(data[:,0],data[:,1], c=data[:,-1])
	plt.axis('scaled')
	ax.grid(True)
	plt.legend(data[:,-1])
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('Data')
	plt.draw()

# plots each row of imput as line
def plot_lines(x,y,title):
    fig, ax = plt.subplots()
    plt.plot(x.T,y.T,'r-')
    ax.grid(True)
    plt.xlabel('Sample Size')
    plt.ylabel('Error')
    plt.title(title)
    plt.show()

# plots first two dim dataset (data) and contour lines (pos, cond)
def simple_paramters(n1, n2, y1, y2):
    # define y1   
    mu1 = np.array([0,0])
    sig1 = np.eye(2)
    # define y2
    mu2 = np.array([3,5])
    sig2 = np.array([[3,2],[2,10]])
    # set parameters   
    param = [{'mu': mu1, 'cov': sig1, 'n': n1, 'y': y1},
             {'mu': mu2, 'cov': sig2, 'n': n2, 'y': y2}]
    return param


# generates two gaussian datasets
def simple_test():
    # setup
    n = 300
    param = simple_paramters(n,n,0,1)
    # sample & mesh
    data = sample_gaussian(param, 2)
    pos, cond = mesh_gaussian(param, 2, [-2,-2], [12,12], 1000)
    # plot 2D data
    plot_2d(data, pos, cond)
    # calculate mutual information
    TRUTH(param, cond)
    KDE(data)
    # create IB object
    ds = dataset(coord = data[:,:-1], labels = data[:,-1])
    ds.s = 2.
    ds.smoothing_type = "uniform"
    ds.coord_to_pxy()
    ds.plot_pxy()


# tests consitency of mutual information estimates
def consitency():
    # setup
    n,step,m = 1000,10,10
    param = simple_paramters(n,n,0,1)
    size = np.arange(step,n+step,step)
    kde = np.zeros((m,int(n/step)))
    for i in range(m):
        # sample
        data = sample_gaussian(param, 2)
        Y = data[:,-1]
        X_y0 = data[Y == 0]
        X_y1 = data[Y == 1]
        # calculate mutual information
        for j in size:
            d = np.vstack([X_y0[0:j,:], X_y1[0:j,:]])
            kde[i,int((j-step)/step)] = KDE(d)
    # plot 2D data
    plot_lines(np.tile(size,(m, 1)),kde,'KDE I(X;Y) Estimator')

# Perceptron
def perceptron_test():
    # setup
    n = 300
    param = simple_paramters(n,n,0,1)
    # sample
    data = sample_gaussian(param, 2)
    # Perceptron
    perceptron = PERCEPTRON()
    # Train and Plot
    l_rate = 0.01
    n_epoch = 1000
    Ixx, Ixy, error = perceptron.info_train(data, l_rate, n_epoch)
    perceptron.plot_IPlane(Ixx,Ixy,np.arange(1,n_epoch+1),error)

# Logistic Regression
def logistic_test():
    # setup
    n = 300
    param = simple_paramters(n,n,0,1)
    # sample
    data = sample_gaussian(param, 2)
    # Perceptron
    logistic = LOGISTIC()
    # Train and Plot
    max_iter = 1000
    alpha = 0.00001
    X = np.concatenate((np.ones((data.shape[0],1)), data[:,0:-1]), axis=1)
    Ixx, Ixy, error = logistic.info_train(X, data[:,-1], max_iter, alpha)
    logistic.plot_IPlane(Ixx,Ixy,np.arange(1,max_iter+1),error)


# Logistic Regression
def logistic_test():
    # setup
    n = 300
    param = simple_paramters(n,n,0,1)
    # sample
    data = sample_gaussian(param, 2)
    # Perceptron
    logistic = LOGISTIC()
    # Train and Plot
    max_iter = 1000
    alpha = 0.0001
    lmbda = 0
    # X = np.concatenate((np.ones((data.shape[0],1)), data[:,0:-1]), axis=1)
    X = np.concatenate((np.ones((data.shape[0],1)), data[:,0:-1],np.square(data[:,0:-1])), axis=1)
    Ixx, Ixy, error = logistic.info_train(X, data[:,-1], max_iter, alpha, lmbda)
    logistic.plot_IPlane(Ixx,Ixy,np.arange(1,max_iter+1),error)

# Softmax Regression
def softmax_test():
    # setup
    n = 300
    param = simple_paramters(n,n,-1,1)
    # sample
    data = sample_gaussian(param, 2)
    # Perceptron
    softmax = SOFTMAX()
    # Train and Plot
    max_iter = 100
    alpha = 0.0001
    Ixx, Ixy, error = softmax.info_train(data[:,0:-1], data[:,-1], max_iter, alpha)
    softmax.plot_IPlane(Ixx,Ixy,np.arange(1,max_iter+1),error)

# K Means Clustering
def kmeans_test():
    # setup
    n = 300
    param = simple_paramters(n,n,0,1)
    # sample
    data = sample_gaussian(param, 2)
    # Perceptron
    kmeans = KMEANS()
    # Train and Plot
    n_clusters = 2
    n_epoch = 100
    Ixx, Ixy, error = kmeans.info_train(data[:,0:-1], data[:,-1], n_clusters, n_epoch)
    kmeans.plot_IPlane(Ixx,Ixy,np.arange(1,n_epoch+1),error)


# main function of tests to run
def main():
    # simple_test()
    # consitency()
    # perceptron_test()
    logistic_test()
    # softmax_test()
    # kmeans_test()

if __name__ == '__main__':
    main()