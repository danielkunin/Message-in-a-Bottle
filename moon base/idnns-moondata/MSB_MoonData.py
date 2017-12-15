# MSB intuition generating NN on simple data
# goal: overfitting in the info plane
# goal: high bias fit in the info plane

from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
#mport plotly.plotly as py

def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=plt.cm.Spectral) 

# Generate a dataset and plot it
np.random.seed(0)
fig = plt.figure(figsize=(5,5))
X, y = datasets.make_moons(100,noise=0.3) #noise up from 0.2
Xtest, ytest = datasets.make_moons(20, noise=0.3)

# plot up the data quickly
plt.scatter(X[:,0], X[:,1], s=10, c=y, cmap=plt.cm.Spectral)
plt.show()

## Start making models

# Train the logistic regression classifier
#clf = skl.linear_model.LogisticRegression()
#clf.fit(X, y)
 
# Plot the decision boundary
#plot_decision_boundary(lambda x: clf.predict(x))
#plt.title("Logistic Regression")

num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0 #0.01 # regularization strength

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss
	
	# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
	
def calc_acc(predict, labels):
	acc = np.sum(predict == labels)*1./len(labels)
	return acc
	
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations

def build_model(nn_hdim, num_passes=1000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
    cts = 0
    trainacc = np.zeros(200)
    testacc = np.zeros(200)
    timepts = np.zeros(200)
    # Gradient descent. For each batch...
    for i in range(num_passes):
 
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 100 == 0:
          #print("Loss after iteration %i: %f" %(i, calculate_loss(model)))
          testacc[cts] = calc_acc(predict(model,Xtest),ytest)
          trainacc[cts] = calc_acc(predict(model,X),y)
          timepts[cts]=i
          print('Iteration', timepts[cts], ' || train acc: ', trainacc[cts], '|| test acc: ', testacc[cts])
          cts+=1
		  
    fig3=plt.figure(figsize=(5,5)) 
    plt.plot(timepts, np.log10(1-trainacc), 'k', label='train')
    plt.plot(timepts, np.log10(1-testacc), 'r', label='test')
    #plt.fill(, np.log10(1-trainacc), 'k', 1, 1-testacc, 'r', alpha=0.3)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('log Accuracy')
    plt.show()

    return model
	
# Build a model with a 3-dimensional hidden layer
model = build_model(1, print_loss=True)
 
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 200")
plt.show()