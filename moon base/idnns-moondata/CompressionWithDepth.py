import numpy as np
from matplotlib import pyplot as plt

temp = np.load('Ixt1010883HD2_{}.npy'.format(0))
#print(temp.shape)
IXT = np.zeros(temp.shape)
ITY = np.zeros(temp.shape)
epochs = np.load('epochcount200.npy')
temp = np.load('Testerror1010883HD2_{}.npy'.format(0))/20
TrainE = np.zeros(temp.shape)
TestE = np.zeros(temp.shape)
#print(np.arange(78).shape)
#print(np.squeeze(IXT).shape)
for i in range(1):
	IXT += np.load('Ixt1010883HD2_{}.npy'.format(i))/1
	ITY += np.load('Ity1010883HD2_{}.npy'.format(i))/1
	TestE += np.load('Testerror1010883HD2_{}.npy'.format(i))/1
	TrainE += np.load('Trainerror1010883HD2_{}.npy'.format(i))/1
	
temp = np.load('Ixt20103HD2_{}.npy'.format(0))
#print(temp.shape)
IXT3 = np.zeros(temp.shape)
ITY3 = np.zeros(temp.shape)
epochs3 = np.load('epochcount20103HD2.npy')
temp = np.load('Testerror20103HD2_{}.npy'.format(0))/1
TrainE3 = np.zeros(temp.shape)
TestE3 = np.zeros(temp.shape)
#print(np.arange(78).shape)
#print(np.squeeze(IXT).shape)
for i in range(1):
	IXT3 += np.load('Ixt20103HD2_{}.npy'.format(i))/1
	ITY3 += np.load('Ity20103HD2_{}.npy'.format(i))/1
	TestE3 += np.load('Testerror20103HD2_{}.npy'.format(i))/1
	TrainE3 += np.load('Trainerror20103HD_{}.npy'.format(i))/1

temp = np.load('Ixt10HD2_{}.npy'.format(0))	
IXT1 = np.zeros(temp.shape)
ITY1 = np.zeros(temp.shape)
epochs1 = np.load('epochcount10HD2.npy')
temp = np.load('Testerror10HD2_{}.npy'.format(0))/1
TrainE1 = np.zeros(temp.shape)
TestE1 = np.zeros(temp.shape)
#print(np.arange(78).shape)
#print(np.squeeze(IXT).shape)
for i in range(1):
	IXT1 += np.load('Ixt10HD2_{}.npy'.format(i))/1
	ITY1 += np.load('Ity10HD2_{}.npy'.format(i))/1
	TestE1 += np.load('Testerror10HD2_{}.npy'.format(i))/1
	TrainE1 += np.load('Trainerror10HD2_{}.npy'.format(i))/1

#ITY2 = np.load('Ity200_{}.npy'.format(0))	
#IXT2 = np.load('Ixt200_{}.npy'.format(0))

#ITY3 = np.load('Ity1000_{}.npy'.format(0))	
#IXT3 = np.load('Ixt1000_{}.npy'.format(0))	

#epochid = np.unique(np.logspace(np.log2(1), np.log2(2000), 100, dtype=int, base=2)) - 1
fig, ax = plt.subplots(figsize=(12,5))
plt.subplot(1,2,1)
#print(TestE)
plt.plot(epochs,TestE, 'r', label='test')
plt.plot(epochs, TrainE, 'k', label='train')
plt.scatter(epochs,TestE,c=epochs,s=20,cmap = 'viridis')
plt.scatter(epochs,TrainE,c=epochs,s=20,cmap = 'viridis')
plt.plot(epochs3,TestE3, 'r--', label='test')
plt.plot(epochs3, TrainE3, 'k--', label='train')
plt.scatter(epochs3,TestE3,c=epochs,s=20,cmap = 'cool')
plt.scatter(epochs3,TrainE3,c=epochs,s=20,cmap = 'cool')
plt.plot(epochs3,TestE1, 'r:', label='test')
plt.plot(epochs3, TrainE1, 'k:', label='train')
plt.scatter(epochs3,TestE1,c=epochs,s=20,cmap = 'Wistia')
plt.scatter(epochs3,TrainE1,c=epochs,s=20,cmap = 'Wistia')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('5 Layer NN(-), 3(--) and 1(:) Accuracy')


plt.subplot(1,2,2)
plt.scatter(np.squeeze(IXT3),np.squeeze(ITY3), c=epochs, s=20, cmap='cool')
plt.scatter(np.squeeze(IXT1),np.squeeze(ITY1), c=epochs, s=20, cmap='Wistia')
plt.scatter(np.squeeze(IXT),np.squeeze(ITY), c=epochs, s=20, cmap='viridis')
#plt.scatter(np.squeeze(IXT3),np.squeeze(ITY2), c=np.arange(IXT2.shape[1]), s=20, cmap='plasma')
plt.xlabel('I(X;T)')
plt.ylabel('I(T;Y)')
plt.colorbar()
plt.title('5 layer NN(viridis), 3(cool) and 1(Wistia) Information')
plt.show()
