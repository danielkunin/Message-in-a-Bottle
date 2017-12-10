import matplotlib.pyplot as plt
import numpy as np


# plot to information plane
def info_plane(x, y, c, name):
    fig, ax = plt.subplots()
    plt.scatter(x, y, c=c, s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Epoch', rotation=270)
    ax.grid(True)
    plt.title(name + " in the Information Plane", fontsize="x-large")
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    plt.show()

# plot to error curve
def info_curve(x, y1, y2, name):
    fig, ax = plt.subplots()
    plt.plot(x, y1, color="#3CBEA3", label="I(X;T)")
    plt.plot(x, y2, color="#1189D5", label="I(T;Y)")
    plt.legend()
    ax.grid(True)
    plt.title(name + " in the Information Curve", fontsize="x-large")
    plt.xlabel('Epoch')
    plt.ylabel('Information')
    plt.show()

# plot to error plane
def error_plane(x, y, c, name):
    fig, ax = plt.subplots()
    plt.scatter(x, np.log(y), c=c, s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Epoch', rotation=270)
    ax.grid(True)
    plt.title(name + " in the Error Plane", fontsize="x-large")
    plt.xlabel('Log Training Error')
    plt.ylabel('Log Test Error')
    plt.show()

# plot to error curve
def error_curve(x, y1, y2, name):
    fig, ax = plt.subplots()
    plt.plot(x, np.log(y1), color="#3CBEA3", label="Training Error")
    plt.plot(x, np.log(y2), color="#1189D5", label="Test Error")
    plt.legend()
    ax.grid(True)
    plt.title(name + " in the Error Curve", fontsize="x-large")
    plt.xlabel('Epoch')
    plt.ylabel('Log Error')
    plt.show()

# plots ratio of mutual information against error
def ratio_curve(x, y1, y2, name):
    fig, ax = plt.subplots()
    plt.plot(x, y1, color="#3CBEA3", label="Ratio")
    plt.plot(x, y2, color="#1189D5", label="Error")
    plt.legend()
    ax.grid(True)
    plt.title(name + " Mutual Information Convergence", fontsize="x-large")
    plt.xlabel('Epoch')
    plt.ylabel('Information')
    plt.show()


def plot_decision_boundary(X, pred_func): 
	# from DENNY BRITZ
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], s=10, c=y, cmap=plt.cm.Spectral) 


# plots first two dim dataset (data) and contour lines (pos, cond)
def plot_2d(x, y, pos, cond):
	fig, ax = plt.subplots()
	for p in cond:
		plt.contour(pos[:,:,0], pos[:,:,1], p)
	plt.scatter(x[:,0],x[:,1], c=y)
	plt.axis('scaled')
	ax.grid(True)
	plt.legend(y)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('Data')
	plt.show()

# plot average of line plot with standard deviation
def plot_area(x, y, mean, std, title):
	fig, ax = plt.subplots()
	plt.fill_between(x, mean - std, mean + std, color="#3CBEA3") 
	plt.plot(x, mean, color="white", lw=2)  
	plt.plot(x, y, '--', color="black", lw=1)  
	ax.grid(True)
	plt.xlabel('Sample Size')
	plt.ylabel('I(X,Y)')
	plt.title(title)
	plt.show()

# plots each row of imput as line
def plot_line(x, y, title):
    fig, ax = plt.subplots()
    plt.plot(x.T,y.T,'r-')
    ax.grid(True)
    plt.xlabel('Sample Size')
    plt.ylabel('I(X,Y)')
    plt.title(title)
    plt.show()
