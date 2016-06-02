///////////////////////////////////////////////////////////////////////////////
// Main Class File:	HW4.java
//
// File:       		NNImpl.java
//
//
// Description: 	Deep learning NN for handwritten digit classification, implemented using:
//						- 2 layer, feed-forward, fully connected network (1 hidden layer)
//						- back-propogation algo
//						- ReLU activation function
//
// Run Args:		<numHidden> : Integer that controls number of hidden layer nodes
//					<learnRate>	: Double that controls amount of change in arc weights
//					<maxEpoch>	: Maximum number of epochs for training
//					"train.txt"
//					"test.txt"
//
// Author:          Cody Frehr
// Course:        	CS 540: Intro to Artificial Intelligence
///////////////////////////////////////////////////////////////////////////////

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 * 
 */

import java.util.*;


public class NNImpl {
	
	public ArrayList<Node> inputNodes=null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes=null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes=null;// list of the output layer nodes
	
	public ArrayList<Instance> trainingSet=null;//the training set
	
	int inputCount = 0;
	int hiddenCount = 0;
	int outputCount = 0;
	int trainingCount = 0;
	Double learningRate=0.001; // variable to store the learning rate
	int maxEpoch=100; // variable to store the maximum number of epochs
	
	
	/**
 	* This constructor creates the nodes necessary for the neural network
 	* Also connects the nodes of different layers
 	* After calling the constructor the last node of both inputNodes and  
 	* hiddenNodes will be bias nodes. 
 	*/
	public NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights) {
		
		this.trainingSet=trainingSet;
		this.trainingCount = trainingSet.size();
		this.hiddenCount = hiddenNodeCount;
		this.learningRate=learningRate;
		this.maxEpoch=maxEpoch;
		
		//input layer nodes
		inputNodes = new ArrayList<Node>();
		int inputNodeCount=trainingSet.get(0).attributes.size();
		this.inputCount = inputNodeCount;
		int outputNodeCount=trainingSet.get(0).classValues.size();
		this.outputCount = outputNodeCount;
		for(int i=0;i<inputNodeCount;i++)
		{
			Node node = new Node(0);
			inputNodes.add(node);
		}
		
		//bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);
		
		//hidden layer nodes
		hiddenNodes = new ArrayList<Node> ();
		for(int j=0;j<hiddenNodeCount;j++)
		{
			Node node = new Node(2);
			//Connecting hidden layer nodes with input layer nodes
			for(int i=0;i<inputNodes.size();i++)
			{
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(i),hiddenWeights[j][i]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		
		//bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);
			
		//Output node layer
		outputNodes = new ArrayList<Node> ();
		for(int k=0;k<outputNodeCount;k++)
		{
			Node node = new Node(4);
			//Connecting output layer nodes with hidden layer nodes
			for(int j=0;j<hiddenNodes.size();j++)
			{
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[k][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
		}	
		
	}
	
	
	/**
	 * Get the output from the neural network for a single instance
	 * Return the idx with highest output values. For example if the outputs
	 * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
	 * The parameter is a single instance
	 */
	public int calculateOutputForInstance(Instance inst) {
		
		Double[] instOutputs = getInstanceOutputs(inst);
		Double maxOutput = instOutputs[0];
		int classification = 0;
		for(int i=0; i<outputCount; i++) {
			if(instOutputs[i] > maxOutput) {
				maxOutput = instOutputs[i];
				classification = i;
			}
		}
		return classification;	
	}
	
	
	// Forward pass
	public Double[] getInstanceOutputs(Instance inst) {
		
		Double[] instOutputs = new Double[outputCount];
		
		// Initialize input values
		Iterator<Double> attributeItr = inst.attributes.iterator();
		Iterator<Node> inputItr = inputNodes.iterator();
		for(int i=0; i<inputCount; i++) {
			inputItr.next().setInput(attributeItr.next());
		}
		
		// Compute hidden layer output values
		Iterator<Node> hiddenItr = hiddenNodes.iterator();
		while(hiddenItr.hasNext()) {
			hiddenItr.next().calculateOutput();
		}
		
		// Compute output layer output values
		Iterator<Node> outputItr = outputNodes.iterator();
		int count = 0;
		Node currNode = null;
		for(int i=0; i<outputCount; i++) {
			currNode = outputItr.next();
			currNode.calculateOutput();
			instOutputs[count] = currNode.getOutput();
			count++;
		}
		//System.out.println("i0=" + instOutputs[0] + ", i1=" + instOutputs[1] + ", i2=" + instOutputs[2]);
		
		return instOutputs;
	}
	
	
	/**
	 * Train the neural networks with the given parameters
	 *         
	 * The parameters are stored as attributes of this class
	 */
	public void train() {
		
		// do maxEpoch loops
		int currEpoch = 1;
		do {
			
			
			// compute weight change for each instance
			Iterator<Instance> instanceItr = trainingSet.iterator();
			Instance currInstance = null;
			while(instanceItr.hasNext()) {
				
				// forward pass
				currInstance = instanceItr.next();
				Double[] trainingOutput = getInstanceOutputs(currInstance);
				int[] actualOutput = currInstance.getClassValues();
				
				// compute error
				Double[] error = new Double[outputCount];
				for(int i=0; i<outputCount; i++) {
					error[i] = actualOutput[i] - trainingOutput[i];
				}
				
				// compute all local outputDeltas
				Double[] localOutputDeltas = getOutputDeltas(currInstance,error);
				
				// compute all local hiddenDeltas
				Double[] localHiddenDeltas = getHiddenDeltas(currInstance,localOutputDeltas);
				
				// get set of weight changes and add to existing sum
				Double[][] outputWeightDeltas = getOutputWeightDeltas(localOutputDeltas);
				Double[][] hiddenWeightDeltas = getHiddenWeightDeltas(localHiddenDeltas);

				
				// update all weights
				updateWeights(outputWeightDeltas,hiddenWeightDeltas);	
			}
			
			currEpoch++;
			
		} while(currEpoch <= maxEpoch);
	}

	
	/**
	 * Compute 2-D array of hidden-->output weight deltas.
	 * @param: A single Instance "inst"
	 * @return: 2-D array of Doubles "outputDeltas"
	 */
	public Double[] getOutputDeltas(Instance inst, Double[] error) {
		
		Double[] outputDeltas = new Double[outputCount];
		Iterator<Node> outputItr = outputNodes.iterator();
		Node currNode = null;
		int k = 0;
		while(k < outputCount) {
			currNode = outputItr.next();
			Double sum = currNode.getSum();
			Double derivative = 0.0;
			if(sum > 0.0) {
				derivative = 1.0;
			}
			outputDeltas[k] = error[k] * derivative;
			k++;
		}
		return outputDeltas;
	}
	
	
	/**
	 * Compute 2-D array of input-->hidden weight deltas.
	 * @param: A single Instance "inst"
	 * @return: 2-D array of Doubles "hiddenDeltas"
	 */
	public Double[] getHiddenDeltas(Instance inst, Double[] outputDeltas) {
		
		Double[] hiddenDeltas = new Double[hiddenCount];
		Iterator<Node> hiddenItr = hiddenNodes.iterator();
		Node currNode = null;
		int j = 0;
		while(j < hiddenCount) {
			currNode = hiddenItr.next();
			Double sum = currNode.getSum();
			Double derivative = 0.0;
			if(sum > 0.0) {
				derivative = 1.0;
			}
			Iterator<Node> outputItr = outputNodes.iterator();
			Double summation = 0.0;
			int k = 0;
			while(k < outputCount) {
				Double weight = outputItr.next().parents.get(j).weight;
				summation += weight*outputDeltas[k];
				k++;
			}
			hiddenDeltas[j] = derivative*summation;
			j++;
		}
		return hiddenDeltas;
	}

	
	public Double[][] getOutputWeightDeltas(Double[] outputDeltas) {
		
		Double[][] weightSet = new Double[outputCount][hiddenCount+1];
		// get output-->hidden weights
		Iterator<Node> outputItr = outputNodes.iterator();
		Node currNode = null;
		int k = 0;
		while(k < outputCount) {
			currNode = outputItr.next();
			Iterator<NodeWeightPair> nwpItr = currNode.parents.iterator();
			NodeWeightPair currPair = null;
			int j=0;
			while(j < hiddenCount+1) {
				currPair = nwpItr.next();
				weightSet[k][j] = learningRate * currPair.node.getOutput() * outputDeltas[k];
				j++;
			}
			k++;
		}
		return weightSet;
	}
	
	
	public Double[][] getHiddenWeightDeltas(Double[] hiddenDeltas) {
		
		Double[][] weightSet = new Double[hiddenCount][inputCount+1];
		// get hidden-->input weights
		Iterator<Node> hiddenItr = hiddenNodes.iterator();
		Node currNode = null;
		int j = 0;
		while(j < hiddenCount) {
			currNode = hiddenItr.next();
			Iterator<NodeWeightPair> nwpItr = currNode.parents.iterator();
			NodeWeightPair currPair = null;
			int i = 0;
			while(i < inputCount+1) {
				currPair = nwpItr.next();
				weightSet[j][i] = learningRate * currPair.node.getOutput() * hiddenDeltas[j];
				i++;
			}
			j++;
		}
		return weightSet;
	}
		
	
	public void updateWeights(Double[][] outputWeightDeltas, Double[][] hiddenWeightDeltas) {
		
		// update output-->hidden weights
		Iterator<Node> outputItr = outputNodes.iterator();
		Node currNode = null;
		int k = 0;
		while(k < outputCount) {
			currNode = outputItr.next();
			Iterator<NodeWeightPair> nwpItr = currNode.parents.iterator();
			NodeWeightPair currPair = null;
			int j = 0;
			while(nwpItr.hasNext()) {
				currPair = nwpItr.next();
				currPair.weight += outputWeightDeltas[k][j];
				j++;
			}
			k++;
		}
		// update hidden-->input weights
		Iterator<Node> hiddenItr = hiddenNodes.iterator();
		int j = 0;
		while(j < hiddenCount) {
			currNode = hiddenItr.next();
			Iterator<NodeWeightPair> nwpItr = currNode.parents.iterator();
			NodeWeightPair currPair = null;
			int i = 0;
			while(nwpItr.hasNext()) {
				currPair = nwpItr.next();
				currPair.weight += hiddenWeightDeltas[j][i];
				i++;
			}
			j++;
		}
		
	}
}
