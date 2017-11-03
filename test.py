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



# generates two gaussian datasets
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
    pos, cond = mesh_gaussian(param, 2, [-2,-2], [12,12], 1000)
    # plot 2D data
    plot_2d(data, pos, cond)
    # calculate mutual information
    mi_gaussian(param, cond)
    # create IB object
    ds = dataset(coord = data[:,:-1], labels = data[:,-1])
    ds.s = 2.
    ds.smoothing_type = "uniform"
    ds.coord_to_pxy()
    ds.plot_pxy()


# main function of tests to run
def main():
	simple_test()

if __name__ == '__main__':
    main()