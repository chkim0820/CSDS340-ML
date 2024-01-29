# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Perceptron:

    def __init__(self):
        self.features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"]
        self.weights = [random.random()] * len(self.features)
        self.bias = random.random() * 0.1
        self.learningRate = random.random() * 0.1

    def prediction(self, data):
        predictions = []
        for i in (range(len(data))):
            y = 0
            for j in (range(len(self.features))):
                net = self.weights[j] * (data.iloc[i, j]) + self.bias
                y += net
            predictions.append(y)
        return predictions

    def threshold(self, data):
        results = []
        for i in (range(len(data))):
            if (data[i] >= 0):
                results.append(1)
            else:
                results.append(0)
        return results
    
    def updateWeights(self, results, data):
        newWeights = [0] * len(self.features)
        for i in range(len(data)):
            for j in range(len(data.columns)):
                actual = data.iloc[i, len(self.features)-1]
                change = self.learningRate * (actual - results[i]) * data.iloc[i, j]
                newWeights[j] = newWeights[j] + change
        return newWeights
    
    def updateBias(self, results, data):
        newBias = 0
        for i in range(len(results)):
            change = self.learningRate * (data[i] - results[i])
            newBias = newBias + change
        return newBias
    
    def learningAlgorithm(self, results, data):
        weights = self.updateWeights(results, data) # outputs are the class labels
        bias = self.updateBias(results, data["Class"])
        return weights, bias

    def main(self):
        # Running the main method once is one epoch
        data = pd.read_csv('pima-indians-diabetes.csv', names=self.features)
        predictions = self.prediction(data)
        outputs = self.threshold(predictions)
        self.weights, self.bias = self.learningAlgorithm(outputs, data)

if __name__ == "__main__":
    training = Perceptron()
    training.main()