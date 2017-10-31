import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from IB import *

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
# 	- column for class
#  	- column for each x 
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




# Plot data
# Input:
#  	* ds = a dataset as a numpy array in tidy format
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



# Generates two gaussian datasets
def simple_test():
	# define y1   
    n1 = 30
    mu1 = np.array([0,0])
    sig1 = np.eye(2)
    # define y2
    n2 = 30
    mu2 = np.array([3,5])
    sig2 = np.array([[3,2],[2,10]])
    # set parameters   
    param = [{'mu': mu1, 'cov': sig1, 'n': n1, 'y': 0},
             {'mu': mu2, 'cov': sig2, 'n': n2, 'y': 1}]
    # sample & mesh
    data = sample_gaussian(param, 2)
    pos, cond = mesh_gaussian(param, 2, [-2,-2], [12,12], 100)
    # plot 2D data
    plot_2d(data, pos, cond)
    # create IB object
    ds = dataset(coord = data[:,:-1], labels = data[:,-1])
    ds.s = 1.
    ds.smoothing_type = "uniform"
    ds.coord_to_pxy()
    ds.plot_pxy()
