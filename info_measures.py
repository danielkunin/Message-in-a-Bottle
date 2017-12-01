import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal as mvn
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


# Computes mutual infromation I(X;Y) of multivariate gaussians
def I_TRUTH(pi, mus, Sigs, m):
	MI = 0
	n = 0
	for i in range(100):
	    k = np.size(pi)
	    ys = np.random.choice(k, size=m, p=pi)
	    xs = np.array([np.random.multivariate_normal(mus[y_i], Sigs[y_i]) for y_i in ys])
	    log_p_xi_g_yi = np.array([mvn.logpdf(x_i, mus[y_i], Sigs[y_i]) for x_i, y_i in zip(xs, ys)])
	    log_p_xi = np.log(np.array([np.sum([mvn.pdf(x_i, mus[y], Sigs[y])*pi[y] for y in np.arange(k)]) for x_i in xs]))
	    I_ = log_p_xi_g_yi - log_p_xi
	    I = np.cumsum(I_)/(np.arange(m)+1)
	    outputs = {'ys': ys, 'xs': xs, 'I': I}
	    n += I[-1] > 0
	    MI += I[-1] * (I[-1] > 0)
	return MI / n


# Binning
def I_BIN(X, Y):

    n,d = X.shape
    # X = data[:,0:-1]
    # Y = data[:,-1]
    n, dX = X.shape
    dY = 1  # Need to fix this for general dimensions
    binsnumber = 100     # an alternative to use the Freeman-Diaconis Criteria which I will implement shortly.
    epsilon = 1e-20

    binX = int(binsnumber)*np.ones(dX)
    binX_y = binsnumber*np.ones(dX)
    for i in range(dX):
        bintemp = 2*stats.iqr(X[:,i])/n**(1/3) # Freeman-Diaconis Criteria
        binX[i] = np.ceil((max(X[:,i])-min(X[:,i]))/bintemp)
        #print(binX[1])

    binY = int(binsnumber)*np.ones(dY)
    for i in range(dY):
        bintemp = 2 * stats.iqr(Y[:]) / n ** (1 / 3)
        binY[i] = np.ceil((max(Y[:])-min(Y[:]))/bintemp) # Freeman-Diaconis Criteria # need to fix this for general dimensions
    Hx_ybin = np.zeros([dY,binsnumber])

    Hx,edgesx = np.histogramdd(X,binX)
    Hy,edgesy = np.histogramdd(Y,binY)

    for i in range(dY):
        edgetemp = edgesy[i]
        for aaa in range(int(binY[i])):
            temp = (edgetemp[aaa]<=Y) & (Y<edgetemp[aaa+1])
            X_Ybin = X[temp,:]
            ny,dim = X_Ybin.shape
            #print(ny, dim) #returns 10,2
            #print(X_Ybin[:,1])
            if ny >0:
                for k in range(dX):
                    bintemp = 2 * stats.iqr(X_Ybin[:,k]) / ny ** (1 / 3)
                    binX_y[k] = np.ceil((max(X_Ybin[:,k]) - min(X_Ybin[:,k])) / bintemp)  # Freeman-Diaconis Criteria
                Histx_ybin, edgesx_y = np.histogramdd(X_Ybin, binX_y)
                Px_ybin = Histx_ybin/ny
                #print(np.sum(Px_ybin==0))
                Hx_ybin[i,aaa] = np.nansum(Px_ybin*np.log2(Px_ybin+epsilon))

    pdfDatax = Hx / n

    MI = np.nansum(np.nansum((pdfDatax*np.log2(pdfDatax+epsilon))))-np.sum(Hx_ybin)


    MIcorrectedFO = MI + np.prod([binX]) / (2 * n * np.log(2))
    # Easiest resource for the first order correction is the SI of Jordan...Liebler PNAS, 2014
    #print(MIcorrectedFO, MI)

    return MIcorrectedFO

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



# Calculates KDE estimated entropy of X with cross validated bandwidth
def KDE_ENTROPY(X, cv):
	H = 0
	if cv == 0:
		H = -stats.gaussian_kde(X.T).logpdf(X.T).sum()
	else:
		# if X.shape[0] <= 1:
		# 	return 0
		bands = {'bandwidth': np.logspace(-1, 1, 2*cv)}
		grid = GridSearchCV(KernelDensity(), bands)
		grid.fit(X)
		kde = grid.best_estimator_
		H = -kde.score(X)
	return H


# Kernel Density Estimator Method for a Continuous and Discrete Random Variable
def KDE_CD(X, Y):
	n, d = X.shape
	# entropy of X
	MI = KDE_ENTROPY(X, 10)
	# conditional entropy of X|Y
	for y in np.unique(Y):
		if X[Y == y].shape[0] >= d: # check there are as many observations as feature dimensions
			MI -= KDE_ENTROPY(X[Y == y], 10)
	# normalize and convert to bits
	MI /= (n * np.log(2))
	return MI


# Kernel Density Estimator Method for two Continuous Random Variables
def KDE_CC(X, Y):
	try:
		# entropy of HXY
		XY = np.concatenate((X, Y), axis=1)
		Hxy = KDE_ENTROPY(XY, 20)
		# entropy of X
		Hx = KDE_ENTROPY(X, 20)
		# entropy of Y
		Hy = KDE_ENTROPY(Y, 20)
		# mutual information
		I = Hx + Hy - Hxy
		# normalize and convert to bits
		I /= (X.shape[0] * np.log(2))
		return I
	except np.linalg.linalg.LinAlgError as err:
		return 0



# Variational Bound Method (VAR)
