# Message-in-a-Bottle

The fundamental challenge of supervised learning is to navigate a trade off between compressing the input features and preserving the meaningful information required for prediction of the output labels. Tishby et al. describe this process as squeezing ``the information that X provides about Y through a bottleneck”, which they termed the Information Bottleneck (IB) Method \cite{Tishby}. This method has provided a new perspective on the recent successes of Deep Learning \cite{2017}. Specifically, they find two phases of learning (a rapid error minimization and a compression phase), provide insight into the value of depth in a neural network and make analogies between generalization and compression of the input.

In this project we set out to explore the Information Bottleneck Method and address the following four major gaps in the current discourse:
\begin{enumerate}
    \item Discuss mutual information (MI) estimates in the context of supervised learning.
    \item Explore a range of learning algorithms in the information plane and investigate the relationship between MI and training error.
    \item Investigate bias vs. variance and generalization vs. compression in the information plane.
    \item Use the information bottleneck to improve neural network performance by adapting the learning strategy to the learning phase
\end{enumerate}


## File Structure
Our code is split into three folders each with their own associated README:

+ learning algorithims - this code base implements our first two goals (MI Estimates and learning algorithms)
+ moon base - this code base is built on https://github.com/ravidziv/IDNNs and the moon dataset from Scikit learn to achieve our third goal (bias/variance and generalization/compression)
+ dynamic batch - this code base is built on https://github.com/ravidziv/IDNNs to achieve our fourth goal (dynamic batch learning algorithm)

Additionally we added our report and two pdfs with additional material:
+ Message in a Bottle - our final report
+ mutual information - our literature search of mutual information estimators
+ information and error - an analytic exploration of the relationship between mutual information and error 