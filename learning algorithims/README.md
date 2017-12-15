# Learning Algorithms in the Information Plane

This code base implements adaptive binning, KDE, and KDE with cross validation mutual information estimators. We also implement the perceptron, logistic regression, softmax regression, and SVM learning algorithm to use these mutual information estimators during training. We generate data from multivariate gaussians to experimentally evaluate our estimators and train our learning algorithms.


## File Structure

+ test.py - Main file that calls other functionality and generates plots
+ generate_data.py - Generate data from different distributions
+ info_measures.py - Estimates measures of information
+ ml_algorithms.py - Implements different learning algorithms
+ plots.py - Contains a variety of plotting functions

