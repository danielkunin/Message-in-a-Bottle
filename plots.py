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
    plt.scatter(self.I_xx[0,:], self.I_xy[0,:], c=self.Err_trn, s=20, cmap='viridis')
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
    plt.scatter(self.I_xx[1,:], self.I_xy[1,:], c=self.Err_trn, s=20, cmap='viridis')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Training Error', rotation=270)
    ax.grid(True)
    plt.xlabel('I(X;X~)')
    plt.ylabel('I(X~;Y)')
    # show plots
    plt.show()