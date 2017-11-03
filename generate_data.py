import numpy as np
from scipy.stats import multivariate_normal as mvn

# Sample data from multivariate gaussian conditionals
# Input:
#  	* param = array of paramters for each conditional distribution
# 		- mu = mean vector
#  		- cov = covariance matrix
# 		- n = number of samples
# 		- y = class label
# 	* dim = dimension of X
# Output:
# A numpy array with data in tidy format
# 	- column (last) for class
#  	- column (1 to dim) for each x 
def sample_gaussian(param, dim):
      
	# generate data
	ds = np.zeros((0, dim + 1))
	for p in param:
		coord = np.random.multivariate_normal(p['mu'],p['cov'],p['n'])
		label = np.full((p['n'], 1), p['y'])
		ds = np.concatenate([ds, np.hstack((coord,label))])    
	return ds



# Computes mesh from multivariate gaussian conditionals
# Input:
#  	* param = array of paramters for each conditional distribution
# 		- mu = mean vector
#  		- cov = covariance matrix
# 		- n = number of samples
# 		- y = class label
# 	* dim = dimension of X
# 	* mins = 
# 	* maxs = 
# 	* nums = 
# Output:
# A numpy array with data in tidy format
# 	- column for class
#  	- column for each x 
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

	
