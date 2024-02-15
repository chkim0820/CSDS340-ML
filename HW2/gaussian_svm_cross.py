# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 2

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Class for running the SVM algorithm
class SVM:

    # Main function; running this method once is one epoch
    def main(self, features, output):
        SVMModel = SVC(kernel='rbf') # Creating an SVM model
        SVMModel.fit(features, output) # Training the model
        SVMPred = SVMModel.predict(features) # Making predictions
        print("Accuracy:", accuracy_score(output, SVMPred)) # Calculating accuracy
        print("Classification Report:\n", classification_report(output, SVMPred))

# Main function; for running the different types of models for the Wine Dataset
if __name__ == "__main__":
    # For problem 4
    outputClass = [1, 1, 1, 1, 2, 2, 2, 2]
    firstFeature = [-2, 0, 0, 2, -1, 1, 0, 0]
    secondFeature = [0, 2, -2, 0, 0, 0, 1, -1]
    dataFor4 = pd.DataFrame({'First': firstFeature, 'Second': secondFeature, 'Output': outputClass})
    features = dataFor4.iloc[:, :-1] # Both features
    classes = dataFor4.iloc[:, -1] # Output class
    SVM(features, classes)