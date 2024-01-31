# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Class Perceptron for running the Perceptron algorithm
class Perceptron:

    # Initializing class variables for Perceptron
    def __init__(self):
        self.features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Class"]
        self.weights = [0.5] * (len(self.features) - 1) # Exclude "Class"
        self.bias = 0.1
        self.learningRate = 0.000019

    # Initial prediction; z = w^T * x + b
    def prediction(self, sample):
        net = 0 # The prediction value (z)
        # For each feature; exclude the actual output ("Class")
        for j in (range(len(self.features) - 1)): # w^T * x
            net += self.weights[j] * (sample[j]) # jth feature
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
        newWeights = self.weights.copy() # Take current weights
        actual = data[len(self.features) - 1] # actual output for ith sample
        # For each feature; exclude the actual output column
        for j in (range(len(self.features) - 1)):
            change = self.learningRate * (actual - result) * data[j]
            newWeights[j] = newWeights[j] + change # update weight for jth feature
        return newWeights
    
    # For updating the bias
    def updateBias(self, predicted, actual):
        newBias = self.learningRate * (actual - predicted) + self.bias
        return newBias
    
    # Learning algorithm; for updating parameters
    def learningAlgorithm(self, prediction, data):
        weights = self.updateWeights(prediction, data)
        bias = self.updateBias(prediction, data["Class"])
        return weights, bias
    
    # Calculates the rate of correct classification out of all
    def accuracyRate(self, actual, calculated):
        accurate = 0 # Stores the number of correct predictions
        # For each sample
        for i in range(len(actual)):
            if (actual[i] == calculated[i]): # Predicted == actual
                accurate += 1
        misclassified = len(actual) - accurate
        return misclassified, (accurate / len(actual)) # ratio of correct predictions out of entire data

    # Plot the number of misclassification errors for each epoch
    def plotGraph(self, accuracyRates):
        plt.plot(accuracyRates)
        plt.xlabel('Epochs')
        plt.ylabel('# Misclassifications')
        plt.title("Number of Misclassification Errors over Epochs")
        plt.show()

    # Main function; running this method once is one epoch
    def main(self, epoch):
        data = pd.read_csv('pima-indians-diabetes.csv', names=self.features)
        misclassificationList = []
        while (epoch > 0):
            outputs = []
            # For each sample
            for i in (range(len(data))):
                prediction = self.prediction(data.loc[i]) # contains output to z = wx + b
                output = self.threshold(prediction)
                self.weights, self.bias = self.learningAlgorithm(output, data.loc[i])
                outputs.append(output) # Add to the list of predicted outputs
            misclassified, accuracyRate = self.accuracyRate(data["Class"], outputs) # Calculate the accuracy of this epoch
            misclassificationList.append(misclassified)
            epoch -= 1
        self.plotGraph(misclassificationList) # Plot the accuracy rate for each epoch

# Main function; for running the Perceptron class
# Input the desired number of epochs
if __name__ == "__main__":
    training = Perceptron()
    training.main(10) # Specify the number of epochs