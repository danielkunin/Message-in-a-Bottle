import numpy as np
from scipy.stats import multivariate_normal as mvn


# Computes mutual infromation I(X;Y) of multivariate gaussians
def mi_gaussian(param, cond):
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



# DJ Strouse Partiotion Method (BIN)

# Edgeworth Expansion Method (EDGE)

# Kernel Density Estimator Method (KDE)

# K-Nearest Neighbor Method (KNN)

# Maximum Likelihood Mutual Information (MLMI)
