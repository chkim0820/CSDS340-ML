# For CSDS 340 HW 3 Problem 2; Logistic regression code copied from HW 2
# Written by Chaehyeon Kim (cxk445)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Class for running the Logistic Regression algorithm
class LogisticRegressionClassifier:

    # Main function; running this method once is one epoch
    def main(self, X_train, X_test, y_train, y_test):
        # Fitting a logistic regression model with no regularization
        LRNone = LogisticRegression(penalty='none')
        LRNone.fit(X_train, y_train) # Training the model
        LRPredN = LRNone.predict(X_test) # Making predictions
        print("No Regularization:")
        print("Accuracy:", accuracy_score(y_test, LRPredN)) # Calculating accuracy
        # print("Classification Report:\n", classification_report(y_test, LRPredN))

        # Fitting a logistic regression model with L1 regularization
        LRl1 = LogisticRegression(solver='liblinear', penalty='l1', C=100) # Trying 0.01, 1, 100
        LRl1.fit(X_train, y_train) # Training the model
        LRPred1 = LRl1.predict(X_test) # Making predictions
        print("L1 Regularization with C=100:")
        print("Accuracy:", accuracy_score(y_test, LRPred1)) # Calculating accuracy
        # print("Classification Report:\n", classification_report(y_test, LRPred1))

        # Fitting a logistic regression model with L2 regularization
        LRl2 = LogisticRegression(solver='liblinear', penalty='l2', C=100) # Trying 0.01, 1, 100
        LRl2.fit(X_train, y_train) # Training the model
        LRPred2 = LRl2.predict(X_test) # Making predictions
        print("L2 Regularization with C=100:")
        print("Accuracy:", accuracy_score(y_test, LRPred2)) # Calculating accuracy
        # print("Classification Report:\n", classification_report(y_test, LRPred2))
        
        # Calculate the L2 norm of each input model
        self.calculateL2Norm(LRNone)
        self.calculateL2Norm(LRl1)
        self.calculateL2Norm(LRl2)
        # Calculate the number of zero weights for each input model
        self.calculateZeroWeights(LRNone)
        self.calculateZeroWeights(LRl1)
        self.calculateZeroWeights(LRl2)

    # For calculating the L2 norm of the given
    def calculateL2Norm(self, model):
        weights = model.coef_
        l2_norm = np.linalg.norm(weights)
        print("The calculated L2 norm of the trained weights: ", l2_norm)
    
    # For calculating the number of zero weights
    def calculateZeroWeights(self, model):
        weights = model.coef_
        numZeroWeights = np.sum(np.abs(weights) < 0.1) # Counting numbers smaller than 0.01 as zeros
        print("Number of zero weights: ", numZeroWeights)


# Main function; for running the different types of models for the Wine Dataset
if __name__ == "__main__":
    # Process the dataset
    dataset = pd.read_csv("wine_dataset.csv") # Import dataset
    features = dataset.iloc[:, :-1] # All features
    classes = dataset.iloc[:, -1] # Actual output classes (red vs. white wine)

    # For dividing into training/test sets (70/30 ratio for training and testing)
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.3, random_state=1)

    # For running the models with the dataset above
    LRTraining = LogisticRegressionClassifier()
    LRTraining.main(X_train, X_test, y_train, y_test) # Highest accuracy achieved: 0.9883040935672515