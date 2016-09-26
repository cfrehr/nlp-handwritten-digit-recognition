
NEURAL NETWORK FOR NATURAL LANGUAGE PROCESSING
==============================================

I. ABOUT
--------
This is a neural network for handwritten numeric digit recognition. Data was pulled from UCI's 
Machine Learning Depository (https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit).
The full problem statement can be viewed in "problem-statement.pdf"

This NN was implemented using:

- 1 hidden layer
- fully connected, feed-forward network
- back propogation algo with stochastic gradient descent
- ReLU activation function


II. RUN ARGS
------------

<numHidden> | Integer that controls number of hidden layer nodes
<learnRate> | Double that controls amount of change in arc weights
<maxEpoch>  | Maximum number of epochs for training
<train>	    | Training data for network
<test>      | Testing data for network

Recommended args: 
<numHidden> : "12" 
<learnRate> : "0.001"
<maxEpoch>  : "100"
<train>	    : "train.txt"
<test>      : "test.txt"


III. FILES
----------
NNImpl.java       	(Cody Frehr)
HW4.java		(Chuck Dyer)	
Instance.java		(Chuck Dyer)
Node.java		(Chuck Dyer)
NodeWeightPair.java	(Chuck Dyer)
train.txt
test.txt
view.py
