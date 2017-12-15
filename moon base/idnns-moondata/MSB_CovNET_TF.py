# MSB_CovNET_TF.py

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
from idnns.networks import model as mo
t0 = time.time()

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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
def getActivations(sess,layer,stimuli):
  units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
  
def exctract_activity(sess, batch_points_all, model, data_sets_org):
	"""Get the activation values of the layers for the input"""
	w_temp = []
	for i in range(0, len(batch_points_all) - 1):
		batch_xs = data_sets_org.data[batch_points_all[i]:batch_points_all[i + 1]]
		batch_ys = data_sets_org.labels[batch_points_all[i]:batch_points_all[i + 1]]
		feed_dict_temp = {x: batch_xs, y_: batch_ys}
		w_temp_local = sess.run(layer,feed_dict=feed_dict_temp)
		for s in range(len(w_temp_local[0])):
			if i == 0:
				w_temp.append(w_temp_local[0][s])
			else:
				w_temp[s] = np.concatenate((w_temp[s], w_temp_local[0][s]), axis=0)
	""""
	  infomration[k] = inn.calc_information_for_epoch(k, interval_information_display, ws_t, params['bins'],
										params['unique_inverse_x'],
										params['unique_inverse_y'],
										params['label'], estimted_labels,
										params['b'], params['b1'], params['len_unique_a'],
										params['pys'], py_hats_temp, params['pxs'], params['py_x'],
										params['pys1'])

	"""
	return w_temp

print('load data')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
  
print('init network')
#1st convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add a softmax readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and readout
print('training...')
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

numepochs = 250
displayfreq = 100
numberofsamples = 100
dropoutprob = 0.5 #changed from 0.5
randomlabels = 0
percent_of_train = 80
name = 'data/' + 'MNIST'#'var_u'
indexsample = np.unique(np.logspace(np.log2(1), np.log2(numepochs), numberofsamples, dtype=int, base=2)) - 1
trainaccout = np.zeros([len(indexsample),1])
testaccout = np.zeros([len(indexsample),1])
epochout = np.zeros([len(indexsample),1])
cts = 0

batch_size = 512
data_sets_org = load_data(name, randomlabels)
print(data_sets_org.data.shape[0])
data_sets = data_shuffle(data_sets_org, percent_of_train)
batch_size = np.min([batch_size, data_sets.train.data.shape[0]])
batch_points = np.rint(np.arange(0, data_sets.train.data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
batch_points_test = np.rint(np.arange(0, data_sets.test.data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
batch_points_all = np.rint(np.arange(0, data_sets_org.data.shape[0] + 1, batch_size)).astype(dtype=np.int32)
if data_sets_org.data.shape[0] not in batch_points_all:
   batch_points_all = np.append(batch_points_all, [data_sets_org.data.shape[0]])
if data_sets.train.data.shape[0] not in batch_points:
   batch_points = np.append(batch_points, [data_sets.train.data.shape[0]])
if data_sets.test.data.shape[0] not in batch_points_test:
   batch_points_test = np.append(batch_points_test, [data_sets.test.data.shape[0]])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for t in range(numepochs):
    tstart = time.time()
    acc_train_array = []
    acc_test_array = []
    for i in range(0, len(batch_points) - 1):
      if t in indexsample and i <len(batch_points_test)-1:
        batch_xs = data_sets.test.data[batch_points_test[i]:batch_points_test[i + 1]]
        batch_ys = data_sets.test.labels[batch_points_test[i]:batch_points_test[i + 1]]
        feed_dict_temp = {x: batch_xs, y_: batch_ys}
        acc = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})#sess.run([model.accuracy],feed_dict=feed_dict_temp)
        acc_test_array.append(acc)
      batch_xs = data_sets.train.data[batch_points[i]:batch_points[i + 1]]
      batch_ys = data_sets.train.labels[batch_points[i]:batch_points[i + 1]]
      feed_dict = {x: batch_xs, y_: batch_ys}
      if t in indexsample:
        tr_err = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        acc_train_array.append(tr_err)
        #print(np.matrix.flatten(getActivations(sess,2,batch_xs[0])))
        #print(i, tr_err)
      #batch = mnist.train.next_batch(200)    #need to loop over all the batches to define an epoch, will fix this later.
      #if i in indexsample: #% displayfreq == 0:
      #  train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      #  test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
      #  print('step %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))
      #  trainaccout[cts] = train_accuracy
      #  testaccout[cts] = test_accuracy
      #  epochout[cts] = i
      #  cts +=1
	  #train
      train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: dropoutprob})
    if t in indexsample:
      train_accuracy = np.mean(acc_train_array)
      test_accuracy = np.mean(acc_test_array)
      trainaccout[cts] = train_accuracy
      testaccout[cts] = test_accuracy
      epochout[cts] = t
      tfinish = time.time()
      print('step %d, training accuracy %g, test accuracy %g, time, %g' % (t, train_accuracy, test_accuracy, tfinish-tstart))
      cts +=1

  print('final test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	  
#First let's do the learning curve:
fig, ax = plt.subplots(figsize=(10,5))
fig.suptitle("the error Plane", fontsize="x-large")
xp = trainaccout+1e-8
yp = testaccout+1e-8
zp = epochout
# information & epoch
plt.subplot(1, 2, 1)
plt.plot(np.log(1-xp),np.log(1-xp),'r-')
plt.scatter(np.log(1-xp), np.log(1-yp), c=zp, s=20, cmap='viridis')
#ax.fill_between(np.log(1-x), np.log(1-y), np.log(1-x), facecolor='red',alpha = 0.5, interpolate=True)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Epoch', rotation=270)
ax.grid(True)
plt.ylabel('Test Error')
plt.xlabel('Training error')
	
plt.subplot(1,2,2)
fig.suptitle("Learning Curve", fontsize="x-large")
plt.plot(zp,np.log(1-xp),'r--')
plt.plot(zp,np.log(1-yp),'k-')
#plt.legend('Training','Test')
t1 = time.time()
print('time to finish is %g',t1-t0)
	
plt.show()