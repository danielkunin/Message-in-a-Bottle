"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
def main():
    import time
    t0 = time.time()
    #Bulid the netowrk
    print ('Building the network')
    net = inet.informationNetwork()
    net.print_information()
    print ('Start running the network')
    net.run_network()
	# MSB addtion
	#temp = net.test_error()
	#print(temp.shape)
    print ('Saving data')
    net.save_data()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network()
    print(net.testerrorsample)
    t1 = time.time()
    print(t1-t0)
if __name__ == '__main__':
    main()

