import matplotlib.pyplot as plt
import numpy as np


# plot to information plane
def info_plane(x, y, c, name):
    fig, ax = plt.subplots()
    plt.scatter(x, y, c=c, s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Epoch', rotation=270)
    ax.grid(True)
    # ax.set_ylim([1,2.1])
    # ax.set_xlim([0,1.2])
    plt.title(name, fontsize="x-large")
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    plt.show()

# plot to information plane
def info_plane2(x, y, c, name):
    fig, ax = plt.subplots()
    plt.scatter(x, y, c=c, s=20, cmap='plasma')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Training Error', rotation=270)
    ax.grid(True)
    # ax.set_ylim([1,2.1])
    # ax.set_xlim([0,1.2])
    plt.title(name, fontsize="x-large")
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


# plots first two dim dataset (data) and contour lines (pos, cond)
def plot_2d(x, y, pos, cond):
	fig, ax = plt.subplots()
	plt.scatter(x[y==0,0],x[y==0,1], c="#3CBEA3", s=10, label="0")
	plt.scatter(x[y==1,0],x[y==1,1], c="#009CDD", s=10, label="1")
	for p in cond:
		plt.contour(pos[:,:,0], pos[:,:,1], p, cmap=plt.get_cmap('OrRd'))
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
	plt.fill_between(x, mean - std, mean + std, color="#46C7B2") 
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
