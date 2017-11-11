import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal as mvn


# Computes mutual infromation I(X;Y) of multivariate gaussians
# # THIS FUNCTION IS NOT CORRECT YET....
# def TRUTH(param, cond):
# 	m = 0
# 	mi = 0
# 	joint = np.zeros(cond[0].shape)
# 	for p,c in zip(param,cond):
# 		# true H(X|Y) from definition normal 
# 		rv = mvn(p['mu'],p['cov'])
# 		Hx = rv.entropy()
# 		print("True H(X|%s) = %.3f" % (p['y'], Hx))
# 		# estimated H(X|Y) from mesh of p(X|Y)
# 		HatHx = np.sum(-c * np.log2(c))
# 		print("Mesh Estimated H(X|%s) = %.3f" % (p['y'], HatHx))
# 		# update terms
# 		m += p['n']
# 		mi -= Hx
# 		joint += p['n'] * c
# 	joint /= m
# 	mi += np.sum(-joint * np.log2(joint)) # What is formula for H(X) from \sum_Y P(X|Y)P(Y) 
# 	print("Mesh Estimated I(X;Y) = %.3f" % mi)


# Empirical Method for two discrete random variables
def EMPIRICAL_DD(X, Y):
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

# Kernel Density Estimator Method for a Continuous and Discrete Random Variable
def KDE_CD(X, Y):
	n, d = X.shape
	# entropy of X
	kernel = stats.gaussian_kde(X.T)
	Hx = -kernel.logpdf(X.T).sum()
	MI = Hx
	# conditional entropy of X|Y
	for y in np.unique(Y):
		if X[Y == y].shape[0] >= d: # check there are as many observations as feature dimensions
			kernel = stats.gaussian_kde(X[Y == y].T)
			Hx_y = -kernel.logpdf(X[Y == y].T).sum()
			MI -= Hx_y
	# normalize and convert to bits
	MI /= (n * np.log(2))
	return MI


# Kernel Density Estimator Method for two Continuous Random Variables
def KDE_CC(X, Y):
	try:
		# entropy of HXY
		XY = np.concatenate((X, Y), axis=1)
		Hxy = stats.gaussian_kde(XY.T).logpdf(XY.T).sum()
		# entropy of X
		Hx = stats.gaussian_kde(X.T).logpdf(X.T).sum()
		# entropy of Y
		Hy = stats.gaussian_kde(Y.T).logpdf(Y.T).sum()
		# mutual information
		I = Hxy - Hx - Hy
		# normalize and convert to bits
		I /= (X.shape[0] * np.log(2))
		return I
	except np.linalg.linalg.LinAlgError as err:
		return 0

# DJ Strouse Partiotion Method (BIN)

# Edgeworth Expansion Method (EDGE)

# K-Nearest Neighbor Method (KNN)

# Maximum Likelihood Mutual Information (MLMI)
