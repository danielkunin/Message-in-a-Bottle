Authors: MSB
Date: Dec 14th,2017

----
This is a readme for our code built off of https://github.com/ravidziv/IDNNs
note: to upload to github, we had to remove the local copy of MNIST
---

Tasks of Interest:
=====================================
To train a model with 5 layer NN:
run mainMSB.py
This file will:
+take the architecture defined in network_parameters.py
+train it over the described number of epochs with fixed minibatch size
+plot out the information plane learning curve
+save all your network learning data to a file (in pickle) to the bin folder
+plot up the learning curve and the error plane (log error on each axis)
+plot up the 4 quadrants of the error plane being mapped to the information plane
===note: the first file saves automatically, but subsequent files are saved manually to avoid overwhelming our testing phase.
=================================================================
To plot the gradients:
run plot_gradients2.py
select the file from IDNNS-moondata\jobs in which you saved the gradients
note: if the file errors at the import that means that the trained network in mainMSB.py was not requested to save gradients. This setting can be changed in network_parameters.py
============================================================================
To run a convolutional NN on MNIST with partial implementation of the information plane (incomplete):
run MSB_CovNET_TF.py
This plots up the learning curve and easily achieves sub-1% error on MNIST
The information plane is in progress but has not been completed for submission of the final project by Dec 14th, 2017
==============================================================================
To run a simple softmax classifier which achieves 92% accuracy on MNIST:
run MSB_softmaxTF.py
This file is a development file but gives important intution for the difficulty of MNIST classification (i.e. its not the hardest thing to do reasonbly well on).
==================================================================================
To learn more about MNIST, we looked at Hu's Moment Invarients:
run HusMoments.py
you will find the application of Hu's Moment Invarients (which are measures that are invarient to scale, translation and rotation of the feature in the image). We use this to reduce the dimensionality of MNIST down from 784 to 8. We then study the classifications as a function of their position in this low dimensional space and plot up results projecting this data down to two principle components.
===============================================================================



How we built off this existing codebase from:
1. Replace MNIST with Moondata (comprehesive changes to the file utils.loaddata())
- importing scikit-learns moon data set
- converting labels to one-hots
- adding N features of Gaussian noise
- porting out

2. Overhaul of the InfoNetworks file
- Introduction of dev accuracy calculation and train accuracy calculation
- porting these values out to our learning curve from our TF session
- adding the 

3. modifications to fix error in mutual_info_estimation.estimate_IY_by_network
- data handling produced errors, had to improve this pipeline.

4. Adding the path to the calculation of the variational information.
Varitional information is included but not used in the code base.

5. Wrote new plotting output for everything we displayed.
-Learning curves
-Information Plane
-Four quadrants of log-error plane to information plane
-Decision Boundaries
-added the log-timespacing for calculational tractability (reduces run time to something managable)

6. IMplemented a dynamic batch size in network.py
- this samples the gradient mean/gradient std and applies a threshold to this ratio.
- once achieved the batch size shifts from the intial state to 128 within a TF session
- we do not have to initailze a interactive session because of our direct control of batch size in the current implementation
- run time is not significantly affected by this single step modification, but may be more adversely impacted by many updates.
- independent of batch size, over the final 1000 epochs, the STD drops rapidly

7. Improved the architecture selection in network_parameters.py
- Formerly only called preset defaults with special calculations associated with each
- Now it allows you to select arbitrary network shape with a simple change in the bottom of the file and a quick change to your MainMSB.py extractlastlayer function.

8. Extensive Modification of network.py (where the network is trained and the TF session is openned)
- fixed an error in extracting activities, was inputing incorrect output dimensions.

9. gradients of code where not initially plotting
- had to feed the gradients out and do our own analysis and saving (via pickle) to successfully plot the mean and std.

10. Extenively explored architecture space in this model
- some interesting notes that didn't make it into the final report:
== If you add a wide layer (large # of neurons) in the middle of your network, the layers preceeding it are not active in the information plane meaning. We believe this means that they are acting in the linear regime of the activation function and simply reshuffling the input before you hit this wide layer.
== the final layer dynamics in the error minimiazation phase is largely independent of the depth of the network, but you see noticeable differences in the compression/diffusion phase
== On MNIST the speed of compression scales with the depth of the layer of study (as predicted by Tishby et al as well)

11. Implemented an L2 regularization for use on the ReLU activation functions
- ICLR double blind paper observes no compression with ReLU (and example of good generalization without compression in each layer, which we also show the inverse of that on our moon data trials)
- Tishby claims that is because of lack of reguliarzation
- We implemented a Regularization which is added into Network parameters
- Too few observations to state unequiviolically for the paper, but there may be slight compression with regularized ReLU

12. Implemented the Freemon-Diaconais critierion for binsize selection (the original code didn't have a maximum entropy method for binsizing, which is ironic because that is what TISHby calls out in the ICLR paper)
- ended up not using this anyway, because of the Varitional MI calculation.



LICENSE CONDITIONS


Copyright (2016) Ravid Shwartz-Ziv
All rights reserved.


For details, see the paper:
Ravid Shwartz-Ziv, Naftali Tishby,

Opening the Black Box of Deep Neural Networks via Information

Arxiv, 2017

Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and non-commercial purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice and this paragraph appear in all copies, modifications, and distributions.


Any commercial use or any redistribution of this software requires a license. 
For further details, contact Ravid Shwartz-Ziv (ravidziv@gmail.com).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

- 
