import matplotlib.pyplot as plt
# plot to information plane
def info_plane(self):
    fig, ax = plt.subplots()
    fig.suptitle(self.name + " in the Information Plane", fontsize="x-large")
    # information & epoch
    plt.subplot(2, 2, 1)
    plt.scatter(self.I_xx[0,:], self.I_xy[0,:], c=self.Epoch, s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Epoch', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # information & training error
    plt.subplot(2, 2, 2)
    plt.scatter(self.I_xx[0,:], self.I_xy[0,:], c=self.Err[0,:], s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Training Error', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # information & epoch
    plt.subplot(2, 2, 3)
    plt.scatter(self.I_xx[1,:], self.I_xy[1,:], c=self.Epoch, s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Epoch', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # information & training error
    plt.subplot(2, 2, 4)
    plt.scatter(self.I_xx[1,:], self.I_xy[1,:], c=self.Err[0,:], s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Training Error', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # show plots
    plt.show()


# plot to error plane
def accuracy_plane(self):
    fig, ax = plt.subplots()
    fig.suptitle(self.name + " in the Accuracy Plane", fontsize="x-large")
    # information & epoch
    plt.subplot(2, 2, 1)
    plt.scatter(self.I_xx[0,:], self.I_xy[0,:], c=self.Err[0,:], s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Training Error', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # information & training error
    plt.subplot(2, 2, 2)
    plt.scatter(self.I_xx[0,:], self.I_xy[0,:], c=self.Err[1,:], s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Test Error', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # information & epoch
    plt.subplot(2, 2, 3)
    plt.scatter(1 - self.Err[1,:], 1 - self.Err[0,:], c=self.I_xx[0,:], s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('I(X;X~)', rotation=270)
    ax.grid(True)
    plt.xlabel('Test Error')
    plt.ylabel('Training Error')
    # information & training error
    plt.subplot(2, 2, 4)
    plt.scatter(1 - self.Err[1,:], 1 - self.Err[0,:], c=self.I_xy[0,:], s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('I(X~;Y)', rotation=270)
    ax.grid(True)
    plt.xlabel('Test Error')
    plt.ylabel('Training Error')
    # show plots
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
	plt.fill_between(x, mean - std, mean + std, color="#3CBEA3")#1189D5")  
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
