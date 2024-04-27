import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#Handle data
def preprocess_dataset(trainingData):
    #Check for any null rows
    if np.any(trainingData.isnull()):
           print('!!!!null entries!!!!')
           print(trainingData[trainingData.isnull()])
    vectors = trainingData.iloc[:, :-1] # All data vectors
    labels = trainingData.iloc[:, -1] # Actual output labels (quality 0 = bad or 1 = good)
    #Standardize data
    st_scaler = StandardScaler()
    vectors = st_scaler.fit_transform(vectors)
    return vectors, labels

# Return accuracy score
# Format: "Test Accuracy: xx.xx%"
def report_accuracy(classifier, vectors, labels):
    return "Test Accuracy: %.2f%%" % (accuracy_score(labels, classifier.predict(vectors)) * 100)

if __name__ == '__main__':
    trainingData = pd.read_csv('./Data/train.csv')
    vectors, labels = preprocess_dataset(trainingData)
    svm = SVC(C=1.65455, gamma=0.548413, kernel='rbf', probability=True, random_state=1).fit(vectors, labels)
    testData =  pd.read_csv('./Data/test.csv')
    testVectors, testLabels = preprocess_dataset(testData)
    print(report_accuracy(svm, testVectors, testLabels))
