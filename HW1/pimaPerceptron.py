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
        self.bias = random.random()
        self.learningRate = random.random()

    def prediction(self, data):
        predictions = pd.DataFrame(columns=self.features)
        for i in (range(len(data))):
            calculations = []
            for j in (range(len(self.features))):
                net = self.weights[j] * (data.iloc[i, j]) + self.bias
                calculations.append(net)
            predictions.loc[i] = calculations
        return predictions

    def threshold(self, data):
        print("hello from threshold function")

    def main(self):
        # Running the main method once is one epoch
        inputs = pd.read_csv('pima-indians-diabetes.csv', names=self.features)
        predictions = self.prediction(inputs)
        self.threshold(predictions)

if __name__ == "__main__":
    training = Perceptron()
    training.main()