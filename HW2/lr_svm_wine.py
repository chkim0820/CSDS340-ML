# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 2

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Class LogisticRegression for running the Logistic Regression algorithm
class LogisticRegressionProcessor:

    # Main function; running this method once is one epoch
    def main(self, X_train, X_test, y_train, y_test):
        LRModel = LogisticRegression() # Creating a logistic regression model
        LRModel.fit(X_train, y_train) # Training the model
        LRPred = LRModel.predict(X_test) # Making predictions
        print("Accuracy:", accuracy_score(y_test, LRPred)) # Calculating accuracy
        print("Classification Report:\n", classification_report(y_test, LRPred))

# Class SVM for running the SVM algorithm
class SVMProcessor:

    # Main function; running this method once is one epoch
    def main(self, X_train, X_test, y_train, y_test):
        SVMModel = SVC(kernel='linear') # Creating an SVM model
        SVMModel.fit(X_train, y_train) # Training the model
        SVMPred = SVMModel.predict(X_test) # Making predictions
        print("Accuracy:", accuracy_score(y_test, SVMPred)) # Calculating accuracy
        print("Classification Report:\n", classification_report(y_test, SVMPred))


# Main function; for running the different types of models for the Wine Dataset
if __name__ == "__main__":
    # Process the dataset
    dataset = pd.read_csv("wine_dataset.csv") # Import dataset
    features = dataset.iloc[:, :-1] # All features
    classes = dataset.iloc[:, -1] # Actual output classes (red vs. white wine)
    # For dividing into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.5, random_state=1)

    # For running the models with the dataset above
    SVMTraining = SVMProcessor(X_train, X_test, y_train, y_test)
    LRTraining = LogisticRegressionProcessor(X_train, X_test, y_train, y_test)
    SVMTraining.main()
    LRTraining.main() 