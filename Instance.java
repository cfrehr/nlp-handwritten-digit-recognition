///////////////////////////////////////////////////////////////////////////////
// Main Class File:	HW4.java
//
// File:       		Instance.java
// Author:          Chuck Dyer
// Course:        	CS 540: Intro to Artificial Intelligence
///////////////////////////////////////////////////////////////////////////////

import java.util.*;
/**
 * Holds data for a particular instance.
 * Attributes are represented as an ArrayList of Doubles
 * Class labels are represented as an ArrayList of Integers. For example,
 * a 3-class instance will have classValues as [0 1 0] meaning this 
 * instance has class 1.
 * Do not modify
 */
 

public class Instance{
	public ArrayList<Double> attributes;
	public ArrayList<Integer> classValues;
	
	public Instance()
	{
		attributes=new ArrayList<Double>();
		classValues=new ArrayList<Integer>();
	}
	
	/**
	 * Get integer representation of Instance classification
	 * @param: void
	 * @return: Integer "classification" between 0 and size of classValues ArrayList
	 */
	 
	public int getClassValue() {
		int numClasses = classValues.size();
		int classification = 0;
		for(int i=0; i<numClasses; i++) {
			if(classValues.get(i) == 1) {
				classification = i;
			}
		}
		return classification;
	}
	
	public int[] getClassValues() {
		int classSize = classValues.size();
		int[] classVals = new int[classSize];
		for(int i=0; i<classSize; i++) {
			classVals[i] = classValues.get(i);
		}
		return classVals;
	}
}
