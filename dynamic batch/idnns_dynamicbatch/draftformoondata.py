# draft for moon data set

"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
import matplotlib.pyplot as plt
import numpy as np
import idnns.plots.utils as utils

def plotlearningcurve(x,y, z):
    fig, ax = plt.subplots(figsize=(10,5))
    fig.suptitle("the error Plane", fontsize="x-large")
    x = x+1e-8
    y = y+1e-8
    # information & epoch
    plt.subplot(1, 2, 1)
    plt.plot(np.log(1-x),np.log(1-x),'r-')
    plt.scatter(np.log(1-x), np.log(1-y), c=z, s=20, cmap='viridis')
    #ax.fill_between(np.log(1-x), np.log(1-y), np.log(1-x), facecolor='red',alpha = 0.5, interpolate=True)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Epoch', rotation=270)
    ax.grid(True)
    plt.ylabel('Test Error')
    plt.xlabel('Training error')
	
    plt.subplot(1,2,2)
    fig.suptitle("Learning Curve", fontsize="x-large")
    plt.plot(z,np.log(1-x),'r--')
    plt.plot(z,np.log(1-y),'k-')
    plt.legend('Training','Test')
	
    plt.show()
	
def extract_array(data, name):
    results = [[data[j,k][name] for k in range(data.shape[1])] for j in range(data.shape[0])]
    return results
	
def extract_arrayLastLayer(data, name):
    lastlayer = 5;
    results = [[data[j][k][lastlayer][name] for k in range(data.shape[1])] for j in range(data.shape[0])]
    return results
	
def plotinfo(IXT,ITY, trainerror, testerror, epochid):
    trainerror = np.reshape(trainerror+1e-8,[1,len(trainerror)])
    testerror = np.reshape(testerror+1e-8, [1,len(testerror)])
    epochid = np.reshape(epochid, [1,len(epochid)])
    #data_array = utils.get_data(name)
    #data  = np.squeeze(np.array(data_array['informationError'])) #had to add to the save file
    I_XT_array = np.array(IXT) #np.array(extract_array(data, 'local_IXT'))
    I_TY_array = np.array(ITY) #np.array(extract_array(data, 'local_ITY'))
    #print(I_XT_array.shape, testerror.shape)
    #I_XT_array = np.array(extract_array(data, 'IXT_vartional'))
    #I_TY_array = np.array(extract_array(data, 'ITY_vartional'))
    #epochsInds = data_array['params']['epochsInds']
    fig, ax = plt.subplots(figsize=(6,6))
    plt.subplot(2, 2, 1)
    plt.scatter(I_XT_array,np.log(1-testerror), c=epochid, s = 20, cmap='viridis')
    plt.xlabel('I(X;T)')
    plt.ylabel('log Test Errror')
	
    plt.subplot(2, 2, 2)
    plt.scatter(np.log(1-trainerror),np.log(1-testerror), c=epochid, s=20, cmap='viridis')
    plt.xlabel('log Train error')
    plt.ylabel('log Test Errror')

    plt.subplot(2, 2, 3)
    plt.scatter(I_XT_array,I_TY_array, c=epochid, s=20, cmap='viridis')
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
	
    plt.subplot(2, 2, 4)
    plt.scatter(np.log(1-trainerror),I_TY_array, c=epochid, s=20, cmap='viridis')
    plt.xlabel('log Train error')
    plt.ylabel('I(T;Y)')
	
    plt.show()
	

def main():
    import time
    t0 = time.time()
    #Bulid the netowrk
    print ('Building the network')
    net = inet.informationNetwork()
    net.print_information()
    print ('Start running the network')
    net.run_network() #replace this with our moon data set.
    temp = net.informationError
    #print('this is the IXT array:')
    #print(extract_arrayLastLayer(temp,'local_IXT'))#print(temp[0][0][5]['local_IXT']) #print(np.array(extract_array(temp, 'local_IXT')))
    IXT = extract_arrayLastLayer(temp,'local_IXT')
    ITY = extract_arrayLastLayer(temp,'local_ITY')
    print ('Saving data')
    net.save_data()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network()
    #print(net.testerrorsample)
    #print(net.trainerrorsample)
    plotlearningcurve(net.trainerrorsample, net.testerrorsample, net.epochs_indexes)
    plotinfo(IXT, ITY, net.trainerrorsample, net.testerrorsample, net.epochs_indexes)
    t1 = time.time()
    print(t1-t0)
if __name__ == '__main__':
    main()

