import matplotlib.pyplot as plt


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
    plt.scatter(x, y, c=c, s=20, cmap='viridis')
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
    plt.plot(x, y1, color="#3CBEA3", label="Training Error")
    plt.plot(x, y2, color="#1189D5", label="Test Error")
    plt.legend()
    ax.grid(True)
    plt.title(name + " in the Error Curve", fontsize="x-large")
    plt.xlabel('Epoch')
    plt.ylabel('Log Error')
    plt.show()

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
