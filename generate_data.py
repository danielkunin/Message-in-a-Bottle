import numpy as np
from scipy.stats import multivariate_normal as mvn


# plots first two dim dataset (data) and contour lines (pos, cond)
def binary_paramters():
    pi = np.array([0.5, 0.5])
    mu = np.array([[0., 0.],[3., 5.]])
    cov = np.array([[[1.,0.],[0.,1.]],[[3.,2.],[2.,10.]]])
    return pi, mu, cov

# plots first two dim dataset (data) and contour lines (pos, cond)
def complex_paramters():
    pi = np.array([1./3, 1./3, 1./3])
    mu = np.array([[0., 0.],[3., 5.],[5., 2.]])
    cov = np.array([[[1.,0.],[0.,1.]],[[3.,2.],[2.,10.]],[[2,0.5],[0.5,2]]])
    return pi, mu, cov

# Sample data from multivariate gaussian conditionals
def sample_gaussian(pi, mu, cov, m):
	# generate data
	k = pi.size
	Y = np.random.choice(k, size=m, p=pi)
	X = np.array([np.random.multivariate_normal(mu[y], cov[y]) for y in Y])    
	return X, Y

# adds column of ones to features
def add_ones(X):
	return np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

# adds squared features to X
def square(X):
	return np.concatenate((X, np.square(X)), axis=1)

# Computes mesh from multivariate gaussian conditionals
def mesh_gaussian(mu, cov, mins, maxs, num):
	# define grid of X values
	X = [np.linspace(i,j,num) for i,j in zip(mins,maxs)]
	pos = np.array(np.meshgrid(*X)).T
	# generate conditional pdf
	cond = []
	for m,c in zip(mu,cov):
		rv = mvn(m,c)
		px_y = rv.pdf(pos)
		px_y = px_y/np.sum(px_y)
		cond.append(px_y)
	return pos, cond

	
