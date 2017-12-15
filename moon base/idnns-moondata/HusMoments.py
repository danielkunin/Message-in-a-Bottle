# Hu's moments

from numpy import mgrid, sum

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

def humomentsMSB(image):
  """
  This function calculates the raw, centered and normalized moments
  for any image passed as a numpy array.
  """
  assert len(image.shape) == 2 # only for grayscale images        
  x, y = mgrid[:image.shape[0],:image.shape[1]]
  moments = {}
  momentsarray = []
  moments['mean_x'] = sum(x*image)/sum(image)
  moments['mean_y'] = sum(y*image)/sum(image)
          
  # raw or spatial moments
  moments['m00'] = sum(image)
  moments['m01'] = sum(x*image)
  moments['m10'] = sum(y*image)
  moments['m11'] = sum(y*x*image)
  moments['m02'] = sum(x**2*image)
  moments['m20'] = sum(y**2*image)
  moments['m12'] = sum(x*y**2*image)
  moments['m21'] = sum(x**2*y*image)
  moments['m03'] = sum(x**3*image)
  moments['m30'] = sum(y**3*image)
  
  # central moments
  # moments['mu01']= sum((y-moments['mean_y'])*image) # should be 0
  # moments['mu10']= sum((x-moments['mean_x'])*image) # should be 0
  moments['mu11'] = sum((x-moments['mean_x'])*(y-moments['mean_y'])*image)
  moments['mu02'] = sum((y-moments['mean_y'])**2*image) # variance
  moments['mu20'] = sum((x-moments['mean_x'])**2*image) # variance
  moments['mu12'] = sum((x-moments['mean_x'])*(y-moments['mean_y'])**2*image)
  moments['mu21'] = sum((x-moments['mean_x'])**2*(y-moments['mean_y'])*image) 
  moments['mu03'] = sum((y-moments['mean_y'])**3*image) 
  moments['mu30'] = sum((x-moments['mean_x'])**3*image) 

    
  # opencv versions
  #moments['mu02'] = sum(image*(x-m01/m00)**2)
  #moments['mu02'] = sum(image*(x-y)**2)

  # wiki variations
  #moments['mu02'] = m20 - mean_y*m10 
  #moments['mu20'] = m02 - mean_x*m01
        
  # central standardized or normalized or scale invariant moments
  moments['nu11'] = moments['mu11'] / sum(image)**(2/2+1)
  moments['nu12'] = moments['mu12'] / sum(image)**(3/2+1)
  moments['nu21'] = moments['mu21'] / sum(image)**(3/2+1)
  moments['nu20'] = moments['mu20'] / sum(image)**(2/2+1)
  moments['nu03'] = moments['mu03'] / sum(image)**(3/2+1) # skewness
  moments['nu30'] = moments['mu30'] / sum(image)**(3/2+1) # skewness
  momentsarray.append(moments['nu11'])
  momentsarray.append(moments['nu12'])
  momentsarray.append(moments['nu21'])
  momentsarray.append(moments['nu20'])
  momentsarray.append(moments['nu03'])
  momentsarray.append(moments['nu30'])
  moments['momentsarray'] = momentsarray
  return moments
  
# Next apply them to MNIST:
def load_data(name, random_labels=False):
	"""Load the data
	name - the name of the dataset
	random_labels - True if we want to return random labels to the dataset
	return object with data and labels"""
	print ('Loading Data...')
	C = type('type_C', (object,), {})
	data_sets = C()
	if True: #name.split('/')[-1] == 'MNIST':
		data_sets_temp = input_data.read_data_sets(os.path.dirname(sys.argv[0]) + "/data/MNIST_data/", one_hot=True)
		data_sets.data = np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis=0)
		data_sets.labels = np.concatenate((data_sets_temp.train.labels, data_sets_temp.test.labels), axis=0)
	else:
		d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
		F = d['F']
		y = d['y']
		C = type('type_C', (object,), {})
		data_sets = C()
		data_sets.data = F
		data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
	# If we want to assign random labels to the  data
	if random_labels:
		labels = np.zeros(data_sets.labels.shape)
		labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
		labels[np.arange(len(labels)), labels_index] = 1
		data_sets.labels = labels
	return data_sets
	
def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
	"""Divided the data to train and test and shuffle it"""
	perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
	C = type('type_C', (object,), {})
	data_sets = C()
	stop_train_index = perc(percent_of_train, data_sets_org.data.shape[0])
	start_test_index = stop_train_index
	if percent_of_train > min_test_data:
		start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
	data_sets.train = C()
	data_sets.test = C()
	if shuffle_data:
		shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
	else:
		shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
	data_sets.train.data = shuffled_data[:stop_train_index, :]
	data_sets.train.labels = shuffled_labels[:stop_train_index, :]
	data_sets.test.data = shuffled_data[start_test_index:, :]
	data_sets.test.labels = shuffled_labels[start_test_index:, :]
	return data_sets

	
randomlabels = False
name = 'data/' + 'MNIST'#'var_u'
data_sets_org = load_data(name, randomlabels)
percent_of_train = 80

# Apply Hu's moments to one image:
print(data_sets_org.data.shape, data_sets_org.labels.shape )

data_sets = data_shuffle(data_sets_org, percent_of_train)

plt.imshow(np.reshape(data_sets.train.data[1],[28,28]))
plt.show()

import cv2
image = np.reshape(data_sets.train.data[1],[28,28])
outputhumoments = cv2.HuMoments(cv2.moments(image)).flatten()

momentsoneimage = humomentsMSB(np.reshape(data_sets.train.data[1],[28,28]))

#print(data_sets.train.data[0].shape)
print(momentsoneimage['momentsarray'])
print(outputhumoments)

# convert the training set into moments

numberofexamples = len(data_sets.train.data)


momentsout = np.zeros([numberofexamples,len(outputhumoments)])
labelsout = np.zeros([numberofexamples,10])
for i in range(numberofexamples):
	image = np.reshape(data_sets.train.data[i],[28,28])
	outputhumoments = cv2.HuMoments(cv2.moments(image)).flatten()
	#momentstemp = humomentsMSB(np.reshape(data_sets.train.data[i],[28,28]))
	momentsout[i] = outputhumoments
	#print(momentsout[i])
	labelsout[i] = data_sets.train.labels[i]

x = momentsout.T[0]
y = momentsout.T[1]
plotlabels = np.argmax(labelsout, axis=1)

print(x.shape, y.shape, plotlabels.shape)

plt.scatter(x,y,s=20, c=plotlabels, cmap='viridis')
plt.show()	


"""	
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca2moments = pca.fit_transform(momentsout)

print(pca2moments.shape)
plotlabels = np.argmax(labelsout, axis=1)

print(plotlabels[0], plotlabels.shape)
x = pca2moments.T[0]
y = pca2moments.T[1]

print(x.shape, y.shape)

plt.scatter(x,y,s=20, c=plotlabels, cmap='viridis')
plt.show()
"""
#a quick alternative:
from matplotlib.mlab import PCA as mlabPCA

mlab_pca = mlabPCA(momentsout)

#print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)

plt.plot(mlab_pca.Y[0:numberofexamples,0],mlab_pca.Y[0:numberofexamples,1], 'o', markersize=7, color='black', alpha=0.5)
#plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.show()

plt.scatter(mlab_pca.Y[0:numberofexamples,0],mlab_pca.Y[0:numberofexamples,1], c=plotlabels, cmap='viridis')
plt.show()


