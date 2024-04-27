# For CSDS 340 Case Study 1
# By Chae Kim (cxk445), Wesley Miller (wgm20)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import LearningCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
import matplotlib.pyplot as plt

# For preprocessing the data
def preprocess_dataset(trainingData):
    #Check for any null rows
    if np.any(trainingData.isnull()):
           print('null entries')
           print(trainingData[trainingData.isnull()])
    vectors = trainingData.iloc[:, :-1] # All data vectors
    labels = trainingData.iloc[:, -1] # Actual output labels (quality 0 = bad or 1 = good)
    #Standardize data
    st_scaler = StandardScaler()
    vectors = st_scaler.fit_transform(vectors)
    #Normalizing data didn't make any changes
    return vectors, labels

# For extracting features using PCA or LDA
def feature_extraction(model, x_train, x_test, y_train, mode='PCA'):
    if (mode=='PCA'):
        pca = PCA(n_components=6)
        pca.fit(x_train)
        new_x_train = pca.transform(x_train)
        new_x_test = pca.transform(x_test)
        model.fit(new_x_train, y_train)
    elif (mode=='LDA'):
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        new_x_train = lda.transform(x_train)
        new_x_test = lda.transform(x_test)
        model.fit(new_x_train, y_train)
    else:
        new_x_test = x_test
        new_x_train = x_train
    return model, new_x_train, new_x_test

# Train SVM model
def train_svm(x, y):
    svm = SVC(probability=True, kernel = 'rbf', random_state=1)
    hyperparams = {
        'C': uniform(1,2),
        'gamma':uniform(0,1),
    }
    grid_search = RandomizedSearchCV(svm, hyperparams, cv=5, n_jobs=-1)
    grid_search.fit(x, y)
    print("best parameters for SVM:", grid_search.best_params_)
    return grid_search.best_estimator_

# Train Logistic Regression model
def train_logit(x, y):
    logit = LogisticRegression(random_state=1, solver='saga', penalty = None, n_jobs=-1)
    hyperparams = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': uniform(0, 20),
        'l1_ratio': uniform(0,1)
    }
    grid_search = RandomizedSearchCV(logit, hyperparams, cv=5, n_jobs=-1)
    grid_search.fit(x, y)
    best_lr = grid_search.best_estimator_
    print("best parameters for Logistic Regression: ", grid_search.best_params_)
    return best_lr

# Train Decision Tree model
def train_decision_tree(x, y):
    dectr = DecisionTreeClassifier(random_state = 1)
    hyperparams = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [3, None],
        'min_samples_leaf': np.arange(20, 25),
        'min_samples_split': np.arange(2, 10),
    }
    grid_search = GridSearchCV(dectr, hyperparams, cv=5, n_jobs=-1)
    grid_search.fit(x, y)
    best_dectr = grid_search.best_estimator_
    print(f"best parameters for Decision Tree", grid_search.best_params_)
    return grid_search.best_estimator_

# Return accuracy score
# Format: "Test Accuracy: xx.xx%"
def report_accuracy(classifier, vectors, labels):
    return "Test Accuracy: %.2f%%" % (accuracy_score(labels, classifier.predict(vectors)) * 100)

def plot_validation_curve(classifier, vectors, labels):
    curve = LearningCurveDisplay.from_estimator(classifier, vectors, labels, shuffle=True)
    curve.figure_.show()
    input()

if __name__ == "__main__":
    #Change to True to use test.csv 
    GRADING_MODE = False
    
    ### For grading ###
    if GRADING_MODE:
        testingData = pd.read_csv('./Data/test.csv')
        x, y = preprocess_dataset(testingData)
        best_model = SVC(C=1.65455, gamma=0.548413, kernel='rbf', probability=True, random_state=1).fit(x, y)
        print("Pre-selected model: SVM")
        print(report_accuracy(best_model, x, y))
        exit()

    #Deal with data 
    trainingData = pd.read_csv('./Data/train.csv')
    vectors, labels =preprocess_dataset(trainingData)
    x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3)
    
    #Train models
    try:
        if False:
            svm = train_svm(x_train, y_train)
            svm, x_train, x_test = feature_extraction(svm, x_train, x_test, y_train)
        else:
            svm = SVC(C=1.65455, gamma=0.548413, kernel='rbf', probability=True, random_state=1).fit(x_train, y_train)
    except Exception as e:
        print(e)
    try:
        if False:
            logit = train_logit(x_train, y_train)
            logit, x_train, x_test = feature_extraction(logit, x_train, x_test, y_train)
        else:
           logit = LogisticRegression(penalty='l2', C=1).fit(x_train,y_train)
    except Exception as e:
        print(e)
    try:
        if False:
            decision_tree = train_decision_tree(x_train, y_train)
            decision_tree, x_train, x_test = feature_extraction(decision_tree, x_train, x_test, y_train)
        else:
            decision_tree = DecisionTreeClassifier(min_samples_leaf=23, min_samples_split= 2, random_state=1, criterion='entropy').fit(x_train, y_train)
    except Exception as e:
        print(e)
    
    #Change testing set if grading mode
    if GRADING_MODE:
        testingData = pd.read_csv('./Data/test.csv')
        x_test, y_test = preprocess_dataset(testingData)
    
    #Report models accuracy
    models = [svm, logit, decision_tree]
    names = ['SVM', "Logistic Regression", "Decision Tree"]
    for model, name in zip(models, names):
        if model is not None:
            print(name)
            print('test set accuracy', report_accuracy(model, x_test, y_test))
            print('training set accuracy', report_accuracy(model, x_train, y_train))
            if not GRADING_MODE:
                try:
                    plot_validation_curve(model, vectors, labels)
                except Exception as e:
                    print(e)