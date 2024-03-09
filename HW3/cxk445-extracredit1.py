# For CSDS 340 extra credit 1 (due 03/08/2024)
# Written by Chaehyeon Kim (cxk445)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Answering the question posted on slide 59 of L14
# Plotting the ROC curve for a random classifier

# Random Classifier for simple classification
def randomClassifier(data):
    outputs = [] # Storing the output classes
    # Classifying based on a threshold value of 0.5
    for pt in data:
        if (pt > 0.5): # If random number is above threshold
            outputs.append(1) # Class 1
        else:
            outputs.append(0) # Class 0 otherwise
    return outputs

# Plotting the ROC curve for a random classifier
def ROCcurve(pred, actual):
    # Calculate ROC curve w/ scikit-learn
    fpr, tpr, _ = roc_curve(actual, pred)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr) # Plotting the ROC curve
    plt.plot([0, 1], [0, 1]) # Straight middle line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for a Random Classifier')
    plt.show()

# Main method
if __name__ == "__main__":
    # Generating 1000 random points between 0 and 1 & classifying
    # Using more data points would lead to a straighter middle line
    data = np.random.rand(1000)
    predictions = randomClassifier(data)
    # 1000 random true outcomes (0 or 1)
    actual = np.random.randint(0, 2, size=1000)
    # Generate a ROC curve
    ROCcurve(predictions, actual)