import numpy as np
from scipy.stats import multivariate_normal as mvn

# Sample data from multivariate gaussian conditionals
def sample_gaussian(param, dim):
	# generate data
	X = np.zeros((0, dim))
	Y = np.zeros(0)
	for p in param:
		coord = np.random.multivariate_normal(p['mu'],p['cov'],p['n'])
		label = np.full(p['n'], p['y'])
		X = np.concatenate([X, coord])
		Y = np.concatenate([Y, label])    
	return X,Y

# adds column of ones to features
def add_ones(X):
	return np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

# adds squared features to X
def square(X):
	return np.concatenate((X, np.square(X)), axis=1)

# Computes mesh from multivariate gaussian conditionals
def mesh_gaussian(param, dim, mins, maxs, num):
	# define grid of X values
	X = [np.linspace(i,j,num) for i,j in zip(mins,maxs)]
	pos = np.array(np.meshgrid(*X)).T
	# generate conditional pdf
	cond = []
	for p in param:
		rv = mvn(p['mu'],p['cov'])
		px_y = rv.pdf(pos)
		px_y = px_y/np.sum(px_y)
		cond.append(px_y)
	return pos, cond

	
