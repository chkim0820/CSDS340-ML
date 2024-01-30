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
        predictions = [] # Stores the predictions for each sample
        # For each training sample
        for i in (range(len(data))):
            net = 0 # The z value; z = w^T * x + b
            # For each feature; exclude the actual output ("Class")
            for j in (range(len(self.features) - 1)): # w^T * x
                net += self.weights[j] * (data.iloc[i, j]) # jth feature
            net = net + self.bias # Add bias
            predictions.append(net)
        return predictions

    # For executing the threshold function
    def threshold(self, data):
        results = [] # Store the predicted outputs for each sample
        # For each training sample
        for i in (range(len(data))):
            if (data[i] >= 0): # Perceptron fires
                results.append(1)
            else: # Perceptron does not fire
                results.append(0)
        return results
    
    # For updating the weights
    def updateWeights(self, results, data):
        newWeights = self.weights.copy()
        # For each training sample
        for i in (range(len(data))):
            # For each feature; exclude the actual output column
            for j in (range(len(self.features) - 1)):
                actual = data.iloc[i, len(self.features) - 1] # actual output for ith sample
                change = self.learningRate * (actual - results[i]) * data.iloc[i, j]
                newWeights[j] = newWeights[j] + change # update for jth feature
        return newWeights
    
    # For updating the bias
    def updateBias(self, results, data):
        newBias = self.bias
        # For each training sample
        for i in (range(len(data))): # for each sample
            change = self.learningRate * (data[i] - results[i])
            newBias = newBias + change
        return newBias
    
    # Learning algorithm; for updating parameters
    def learningAlgorithm(self, predictions, data):
        weights = self.updateWeights(predictions, data)
        bias = self.updateBias(predictions, data["Class"])
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
    def main(self, iter=None):
        accuracy = 0
        data = pd.read_csv('pima-indians-diabetes.csv', names=self.features)
        while (iter >= 0):
            predictions = self.prediction(data) # contains output to z = wx + b
            outputs = self.threshold(predictions)
            self.weights, self.bias = self.learningAlgorithm(outputs, data)
            accuracy = self.accuracyRate(data["Class"], outputs)
            iter -= 1
            print(accuracy)

# Main function; for running the Perceptron class
# Input the desired number of epochs
if __name__ == "__main__":
    training = Perceptron()
    training.main(20) # Specify the number of epochs