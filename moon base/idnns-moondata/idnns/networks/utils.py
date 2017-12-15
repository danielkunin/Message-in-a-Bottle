import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as sio
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import datasets as datasets2


def load_data(name, random_labels=False):
	"""Load the data
	name - the name of the dataset
	random_labels - True if we want to return random labels to the dataset
	return object with data and labels"""
	print ('Loading Data...')
	C = type('type_C', (object,), {})
	data_sets = C()
	if name.split('/')[-1] == 'MNIST':
		data_sets_temp = input_data.read_data_sets(os.path.dirname(sys.argv[0]) + "/data/MNIST_data/", one_hot=True)
		#fig = plt.figure(figsize=(5,5))
		X, y = datasets2.make_moons(1000, noise=0.5) #noise up from 0.2
		print(X.shape)
		data_sets.data = np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis=0)
		#plt.scatter(X[:,0], X[:,1])
		data_sets.labels = y#np.concatenate((data_sets_temp.train.labels, data_sets_temp.test.labels), axis=0)
		#plt.scatter(data_sets.data[:,0], data_sets.data[:,1])
		#plt.show()
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
	#plt.scatter(data_sets.data[:,0], data_sets.data[:,1])
	#plt.show()
	np.random.seed(0)
	X, y = datasets2.make_moons(100, noise=0.3) #noise down from 0.5
	# lets try adding redundant dimensions in feature space with gaussian noise.
	numdimensiontoadd = 100
	noise = 0.3
	#xnoise = np.zeros(X.shape)
	#for dim in range(numdimensiontoadd):
	#	Xnoisey = noise*np.random.randn(X.shape[0],X.shape[1])
	#	xnoise = np.concatenate((xnoise, Xnoisey),axis = 1)
	#Xhd = np.concatenate((xnoise, X),axis=1)
	#print(Xhd.shape)
	data_sets.data = X
	print(data_sets.data.shape)
	labels = np.zeros([len(y),2])
	labels[y==0,0] = 1
	labels[y==1,1] = 1
	data_sets.labels = labels #np.concatenate((data_sets_temp.train.images, data_sets_temp.test.images), axis=0)
	#plt.scatter(np.squeeze(data_sets.data[:,2]), np.squeeze(data_sets.data[:,3]), s=10, c=y, cmap=plt.cm.Spectral)
	#plt.show()
	return data_sets


def shuffle_in_unison_inplace(a, b):
	"""Shuffle the arrays randomly"""
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
	"""Divided the data to train and test and shuffle it"""
	perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
	C = type('type_C', (object,), {})
	data_sets = C()
	stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
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


def _convert_string_dtype(dtype):
	if dtype == 'float16':
		return tf.float16
	if dtype == 'float32':
		return tf.float32
	elif dtype == 'float64':
		return tf.float64
	elif dtype == 'int16':
		return tf.int16
	elif dtype == 'int32':
		return tf.int32
	elif dtype == 'int64':
		return tf.int64
	elif dtype == 'uint8':
		return tf.int8
	elif dtype == 'uint16':
		return tf.uint16
	else:
		raise ValueError('Unsupported dtype:', dtype)
