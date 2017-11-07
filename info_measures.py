import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal as mvn


# Computes mutual infromation I(X;Y) of multivariate gaussians
# THIS FUNCTION IS NOT CORRECT YET....
def TRUTH(param, cond):
	m = 0
	mi = 0
	joint = np.zeros(cond[0].shape)
	for p,c in zip(param,cond):
		# true H(X|Y) from definition normal 
		rv = mvn(p['mu'],p['cov'])
		Hx = rv.entropy()
		print("True H(X|%s) = %.3f" % (p['y'], Hx))
		# estimated H(X|Y) from mesh of p(X|Y)
		HatHx = np.sum(-c * np.log2(c))
		print("Mesh Estimated H(X|%s) = %.3f" % (p['y'], HatHx))
		# update terms
		m += p['n']
		mi -= Hx
		joint += p['n'] * c
	joint /= m
	mi += np.sum(-joint * np.log2(joint)) # What is formula for H(X) from \sum_Y P(X|Y)P(Y) 
	print("Mesh Estimated I(X;Y) = %.3f" % mi)


# Computes MI from eprical discrete joint distribution
def DISCRETE(X, Y):
	# Setup
	n = len(X)
	# Compute Empirical Distributions
	valX, countX = np.unique(X, return_counts=True)
	valY, countY = np.unique(Y, return_counts=True)
	valXY, countXY = np.unique(zip(X,Y), return_counts=True)
	# Compute Entropy
	Hx = np.sum(-countX/n * np.log2(countX/n))
	Hy = np.sum(-countY/n * np.log2(countY/n))
	Hxy = np.sum(-countXY/n * np.log2(countXY/n))
	# Compute Mutual Information
	MI = Hx + Hy - Hxy
	return MI

# DJ Strouse Partiotion Method (BIN)

# Edgeworth Expansion Method (EDGE)

# Kernel Density Estimator Method (KDE)
def KDE(data):
	# Setup
	X = data[:,0:-1]
	Y = data[:,-1]
	n,d = X.shape
	# H(X)
	kernel = stats.gaussian_kde(X.T)
	Hx = -kernel.logpdf(X.T).sum()
	MI = Hx
	# H(X|Y)
	for y in np.unique(Y):
		if X[Y == y].shape[0] >= d: #need to make sure there are at least as many obs as dim
			kernel = stats.gaussian_kde(X[Y == y].T)
			Hx_y = -kernel.logpdf(X[Y == y].T).sum()
			MI -= Hx_y
	# Normalize and Convert Base 2
	MI /= (n * np.log(2))
	return MI


# K-Nearest Neighbor Method (KNN)

# Maximum Likelihood Mutual Information (MLMI)
