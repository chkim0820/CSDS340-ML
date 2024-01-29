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
        self.weights = [random.random() * 0.1] * len(self.features) 
        self.bias = random.random() * 0.1
        self.learningRate = 0.1

    # Initial prediction; z = w^T * w + b
    def prediction(self, data):
        predictions = [] # Prediction for each sample
        # For each sample
        for i in (range(len(data))):
            net = 0 # The z value; z = w^T * w + b
            # For each feature
            for j in (range(len(self.features))):
                net += self.weights[j] * (data.iloc[i, j])
            predictions.append(net + self.bias) # add bias at the end of the 
        return predictions

    # For executing the threshold function
    def threshold(self, data):
        results = []
        for i in (range(len(data))):
            if (data[i] >= 0):
                results.append(1)
            else:
                results.append(0)
        return results
    
    # For updating the weights
    def updateWeights(self, results, data):
        newWeights = [0] * len(self.features)
        for i in range(len(data)):
            for j in range(len(data.columns)):
                actual = data.iloc[i, len(self.features)-1] # actual output for ith sample
                change = self.learningRate * (actual - results[i]) * data.iloc[i, j]
                newWeights[j] = newWeights[j] + change # update for jth feature
        return newWeights
    
    # For updating the bias
    def updateBias(self, results, data):
        newBias = 0
        for i in range(len(results)): # for each sample
            change = self.learningRate * (data[i] - results[i])
            newBias = newBias + change
        return newBias
    
    # Learning algorithm; for updating parameters
    def learningAlgorithm(self, results, data):
        weights = self.updateWeights(results, data) # outputs are the class labels
        bias = self.updateBias(results, data["Class"])
        return weights, bias
    
    # Calculates the rate of correct classification out of all
    def accuracyRate(self, actual, calculated):
        num = 0
        for i in range(len(actual)):
            if (actual[i] == calculated[i]):
                num += 1
        return (num / len(actual))

    # Main function; running this method once is one epoch
    def main(self, iter):
        accuracy = 0
        data = pd.read_csv('pima-indians-diabetes.csv', names=self.features)
        while (accuracy != 1):
            predictions = self.prediction(data) # contains output to z = wx + b
            outputs = self.threshold(predictions)
            self.weights, self.bias = self.learningAlgorithm(outputs, data)
            accuracy = self.accuracyRate(data["Class"], outputs)

# Main function; for running the Perceptron class
#Input the desired number of epochs
if __name__ == "__main__":
    training = Perceptron()
    training.main(1)