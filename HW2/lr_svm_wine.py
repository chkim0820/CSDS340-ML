# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 2

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Class for running the SVM algorithm
class SVMClassifier:

    # Main function; running this method once is one epoch
    def main(self, X_train, X_test, y_train, y_test):
        SVMModel = SVC(kernel='linear') # Creating an SVM model
        SVMModel.fit(X_train, y_train) # Training the model
        SVMPred = SVMModel.predict(X_test) # Making predictions
        print("Accuracy:", accuracy_score(y_test, SVMPred)) # Calculating accuracy
        print("Classification Report:\n", classification_report(y_test, SVMPred))


# Class for running the Logistic Regression algorithm
class LogisticRegressionClassifier:

    # Main function; running this method once is one epoch
    def main(self, X_train, X_test, y_train, y_test):
        LRModel = LogisticRegression(penalty='l2', C=50, solver='liblinear') # Creating a logistic regression model
        LRModel.fit(X_train, y_train) # Training the model
        LRPred = LRModel.predict(X_test) # Making predictions
        print("Accuracy:", accuracy_score(y_test, LRPred)) # Calculating accuracy
        print("Classification Report:\n", classification_report(y_test, LRPred))


def svmProb4(features, output):
    SVMModel = SVC(kernel='rbf') # Creating an SVM model
    SVMModel.fit(features, output) # Training the model
    SVMPred = SVMModel.predict(features) # Making predictions
    print("Accuracy:", accuracy_score(output, SVMPred)) # Calculating accuracy
    print("Classification Report:\n", classification_report(output, SVMPred))


# Main function; for running the different types of models for the Wine Dataset
if __name__ == "__main__":
    # Process the dataset
    dataset = pd.read_csv("wine_dataset.csv") # Import dataset
    features = dataset.iloc[:, :-1] # All features
    classes = dataset.iloc[:, -1] # Actual output classes (red vs. white wine)
    # For dividing into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.5, random_state=1)

    # For running the models with the dataset above
    SVMTraining = SVMClassifier()
    LRTraining = LogisticRegressionClassifier()
    SVMTraining.main(X_train, X_test, y_train, y_test) # Highest accuracy achieved: 0.9858417974761465
    LRTraining.main(X_train, X_test, y_train, y_test) # Highest accuracy achieved: 0.9883040935672515

    # For problem 4
    outputClass = [1, 1, 1, 1, 2, 2, 2, 2]
    firstFeature = [-2, 0, 0, 2, -1, 1, 0, 0]
    secondFeature = [0, 2, -2, 0, 0, 0, 1, -1]
    dataFor4 = pd.DataFrame({'First': firstFeature, 'Second': secondFeature, 'Output': outputClass})
    features = dataFor4.iloc[:, :-1] # Both features
    classes = dataFor4.iloc[:, -1] # Output class
    svmProb4(features, classes)