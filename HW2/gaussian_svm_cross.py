# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 2

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Class for running the SVM algorithm
class SVMClassifier:
    # For plotting the decision region
    def plotDecision(self, model, features, output):
        # Plot the decision region
        h = 0.02
        x_min, x_max = features['First'].min() - 1, features['First'].max() + 1
        y_min, y_max = features['Second'].min() - 1, features['Second'].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(features['First'], features['Second'], c=output)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title('Gaussian Kernel SVM Decision Region')
        plt.show()

    # Main function; running this method once is one epoch
    def main(self, features, output):
        SVMModel = SVC(kernel='rbf') # Creating an SVM model
        SVMModel.fit(features, output) # Training the model
        # SVMPred = SVMModel.predict(features) # Making predictions
        # print("Accuracy:", accuracy_score(output, SVMPred)) # Calculating accuracy
        # print("Classification Report:\n", classification_report(output, SVMPred))
        self.plotDecision(SVMModel, features, output)
    

# Main function; for running the different types of models for the Wine Dataset
if __name__ == "__main__":
    # Creating dataset
    outputClass = [1, 1, 1, 1, 2, 2, 2, 2]
    firstFeature = [-2, 0, 0, 2, -1, 1, 0, 0]
    secondFeature = [0, 2, -2, 0, 0, 0, 1, -1]
    dataFor4 = pd.DataFrame({'First': firstFeature, 'Second': secondFeature, 'Output': outputClass})
    # Specifying input & output
    features = dataFor4.iloc[:, :-1] # Both features
    classes = dataFor4.iloc[:, -1] # Output class
    # Creating an instance of the SVM classifier
    classifier = SVMClassifier()
    classifier.main(features, classes)