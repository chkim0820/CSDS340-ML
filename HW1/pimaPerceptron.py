# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Class Perceptron for running the Perceptron algorithm
class Perceptron:

    # Initializing class variables for Perceptron
    def __init__(self):
        self.features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"]
        self.weights = [0] * (len(self.features) - 1) # Exclude "Class"
        self.bias = 0
        self.learningRate = 0.1

    # Initial prediction; z = w^T * x + b
    def prediction(self, data):
        net = 0 # The prediction value (z)
        # For each feature; exclude the actual output ("Class")
        for j in (range(len(self.features) - 1)): # w^T * x
            net += self.weights[j] * (data[j]) # jth feature
        net = net + self.bias # Add bias
        return net

    # For executing the threshold function
    def threshold(self, prediction):
        # For each training sample
        if (prediction >= 0): # Perceptron fires
            return 1
        else: # Perceptron does not fire
            return 0
    
    # For updating the weights
    def updateWeights(self, result, data):
        newWeights = self.weights.copy()
        # For each feature; exclude the actual output column
        for j in (range(len(self.features) - 1)):
            actual = data[len(self.features) - 1] # actual output for ith sample
            change = self.learningRate * (actual - result) * data[j]
            newWeights[j] = newWeights[j] + change # update for jth feature
        return newWeights
    
    # For updating the bias
    def updateBias(self, predicted, actual):
        change = self.learningRate * (actual - predicted)
        newBias = self.bias + change
        return newBias
    
    # Learning algorithm; for updating parameters
    def learningAlgorithm(self, prediction, data):
        weights = self.updateWeights(prediction, data)
        bias = self.updateBias(prediction, data["Class"])
        return weights, bias
    
    # Calculates the rate of correct classification out of all
    def accuracyRate(self, actual, calculated):
        num = 0 # Stores the number of correct predictions
        # For each sample
        for i in range(len(actual)):
            if (actual[i] == calculated[i]):
                num += 1
        return (num / len(actual)) # ratio of correct predictions out of entire data

    # Main function; running this method once is one epoch
    def main(self, epoch):
        data = pd.read_csv('pima-indians-diabetes.csv', names=self.features)
        accuracy = 0
        while (epoch >= 0):
            outputs = []
            # For each sample
            for i in (range(len(data))):
                prediction = self.prediction(data.loc[i]) # contains output to z = wx + b
                output = self.threshold(prediction)
                self.weights, self.bias = self.learningAlgorithm(output, data.loc[i])
                outputs.append(output) # Add to the list of predicted outputs
            accuracy = self.accuracyRate(data["Class"], outputs)
            epoch -= 1
        print(accuracy)

# Main function; for running the Perceptron class
# Input the desired number of epochs
if __name__ == "__main__":
    training = Perceptron()
    training.main(100) # Specify the number of epochs