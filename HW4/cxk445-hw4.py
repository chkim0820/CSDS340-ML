# Name: Chaehyeon Kim (cxk445)
# Description: For CSDS 340 HW 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score


# For homework problem 2; ensemble model bagging technique
def prob2(features, outputs):
    # a) Generating a random dataset with 20 samples (2 features & 1 output (0 or 1))
    # The dataset is randomly generated inside the main method (for easy reuse)
    dataset = pd.DataFrame(features, columns=['Feature1', 'Feature2']) # Specifying the names of feature columns
    dataset['Label'] = outputs # dataset contains the generated random datset
    print("A random dataset with 20 samples:\n", dataset)
    
    # b) Generating 10 training datsets (20 samples each) by sampling with repetition
    matrix = np.zeros((10, 20), dtype=int) # As specified, the output would be a 10 x 20 matrix
    for i in range(10):
        indices = np.random.choice(a=20, size=20, replace=True) # Picking 10 numbers out of 0-19 with replacements
        matrix[i, :] = indices
    print("Matrix of generated samples:\n", matrix)
    
    # c) Highlighting all duplicated entries in each training dataset
    matrix_highlighted = np.array([[np.sum(row == value) > 1 for value in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(matrix_highlighted, cmap="Greys", interpolation='nearest', aspect='auto')
    for i in range(matrix_highlighted.shape[0]):
        for j in range(matrix_highlighted.shape[1]):
            text = ax.text(j, i, matrix[i, j], # Print each index of the above matrix
                        ha="center", va="center", color="red" if matrix_highlighted[i, j] else "black")
    ax.set_title('Highlighted Duplicates in Each Dataset')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Dataset Number')
    plt.show()

    # d) Training 10 classifier models on the 10 datasets
    models = []
    accuracies = []
    for i in range(10):
    # Create a new model
        model = DecisionTreeClassifier()
        X_train, X_test, y_train, y_test = train_test_split(features[matrix[i]], outputs[matrix[i]], test_size=0.2, random_state=42)
        # Train the model on the dataset
        model.fit(X_train, y_train)
        # Add the trained model to the list
        models.append(model)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # e) Using majority voting to combine the results
    predictions = [model.predict(X_test[[0]])[0] for model in models] # Generate predictions for each classifier
    final_prediction = np.argmax(np.bincount(predictions)) # Select the class of highest frequency (voting)
    print("Final prediction: ", final_prediction)
    
    # returning the random generated dataset to be used for later problems
    return features, outputs


# For homework problem 3; ensemble model boosting technique (AdaBoost)
def prob3(features, outputs):
    # a) a random dataset with 20 samples with relabeled outputs (-1 and 1)
    outputs[outputs == 0] = -1 # Reusing the classes from problem 4 w/ relabeled classes
    print("Random dataset for problem 3:")
    print("Feature columns:\n", features)
    print("Output column:\n", outputs)

    # b, c, d) training 10 total weak learners; output features, thresholds, and αj
    learners = [] # to store 9 weak learners
    coefficients = [] # to store 9 coefficients
    for i in range(10):
        # Specifies the index of the current classifier out of 10
        print(f"Learner {i + 1}:")

        # b) training a weak learner with uniform weights for samples
        # Using scikit learn's Decision Tree Classifier
        stump = DecisionTreeClassifier(max_depth=1, random_state=42) # depth = 1 for weak learner   
        # Initializing uniform weight vectors
        weights = np.ones(len(outputs)) / len(outputs) # w=0.05 for each of 20 samples
        stump.fit(features, outputs, sample_weight=weights) # Training the weak learner
        feature = stump.tree_.feature[0] # First and only split since depth = 1
        threshold = stump.tree_.threshold[0] # Threshold also stored at index 1
        print("The feature of the split:", feature)
        print("The threshold of the split:", threshold)

        # c) calculating the coefficient and updated weights
        # Initializing weight vectors to uniform weights
        predictions = stump.predict(features) # class label predictions
        error = np.sum(weights * (predictions != outputs)) # Calculating weighted error rates
        coefficient = 0.5 * np.log((1 - error) / error)  # coefficient αj
        print("Coefficient:", coefficient)
        weights = weights * np.exp(-coefficient * outputs * predictions) # Updating weights
        weights = weights / np.sum(weights) # normalizing weights to sum of 1
        print("Updated weights:\n", weights)
        
        # appending results for later use
        coefficients.append(coefficient) # for storing the coefficient
        learners.append(stump)

    # e) predicting using each weak learner & combining results using coefficients
    example_sample = features[0] # using the first example as a sample
    final_prediction = 0 # to store the final prediction
    # Final prediction using each weak learner and coefficient
    for learner, alpha in zip(learners, coefficients):
        model = learner
        prediction = model.predict([example_sample])[0]
        final_prediction += alpha * prediction
    # determines the final prediction with the sign of aggregate
    final_prediction = np.sign(final_prediction) 
    print("Final prediction using all weak learners:", final_prediction)


# For homework problem 4; gradient boosting technique of ensemble model
def prob4():
    # Defining variables for part g
    learning_rate = 0.1 # as given in the problem
    gamma_example = [] # saves gamma values for the first sample through iterations
    y_new = 0 # the first predicted output for the first example

    # a) a random dataset with 20 samples with correct outputs (0 and 1)
    np.random.seed(42)
    features = np.random.rand(20, 2)
    outputs = np.random.randint(0, 2, 20)
    print("Random dataset for problem 3:")
    print("Feature columns:\n", features)
    print("Output column:\n", outputs)

    # b) outputting log-odds of the dataset; yi
    # Calculating log-odds: log(num true samples / num all)
    num_true = np.sum(outputs==1)
    prob = np.sum(outputs==1) / len(outputs) # pi
    log_odds = np.log(num_true / (len(outputs) - num_true)) # yi
    y_new = log_odds # saving log-odds as the first predicted output
    print("Log-odds of the dataset:", log_odds)

    # c) calculating the residuals; yi - pi (actual - predicted)
    # Set all of the samples' predicted outputs as the log odds
    residuals = []
    for i in range(20):
        residual = outputs[i] - prob
        residuals.append(residual)
    residuals = np.array(residuals)
    print("Residual terms:", residuals)

    # d) fitting to the residuals & outputting γj1 for each leaf node
    tree = DecisionTreeRegressor(max_depth=2, random_state=1) # max depth set to 2
    tree.fit(features, residuals) # fitting the model to residuals
    # Calculating γj1
    leaf_nodes = tree.apply(features)
    gamma_values = []
    for leaf in leaf_nodes:
        num_samples = np.sum(tree.apply(features)==leaf) # number of samples in leaf
        sum_residuals = np.sum(residuals[tree.apply(features)==leaf])
        gamma = sum_residuals / (prob * (1 - prob)) * num_samples
        gamma_values.append(gamma)
    gamma_example.append(gamma_values[0])
    print("γj1 values:\n", gamma_values)

    # e) choosing at least two samples from each leaf node; output predicted values
    leaf_samples = {}
    for leaf in set(leaf_nodes): # set method for running once per leaf node
        leaf_samples[leaf] = features[leaf_nodes == leaf][:2]  # Selecting 2 samples
        print(f"Two selected samples for leaf{leaf}: ", leaf_samples[leaf])
        predictions = tree.predict(np.array(leaf_samples[leaf])) # all predictions
        print("Predictions:", predictions)
    predictions = tree.predict(features) # setting up for part f
    # Updating with proper learning rates
    for i in range(len(predictions)):
        predictions[i] += learning_rate * gamma_values[leaf_nodes[i]]

    # f) Train 9 more decision trees and output γjk values
    for i in range(2, 11):
        # Residuals; yi - pi
        residuals = outputs - predictions # defining residual terms
        tree.fit(features, residuals) # fitting new model to residuals
        # Calculating γj1
        leaf_nodes = tree.apply(features) # getting leaf indeces
        gamma_values = []
        for leaf in leaf_nodes:
            num_samples = np.sum(tree.apply(features)==leaf) # number of samples in leaf
            sum_residuals = np.sum(residuals[tree.apply(features)==leaf]) # sum of residuals
            sum_probs = np.sum(predictions[tree.apply(features==leaf)])
            gamma = sum_residuals / sum_probs * num_samples # gamma of current leaf
            gamma_values.append(gamma)
        print("γj1 values:\n", gamma_values)
        gamma_example.append(gamma_values[0])
        # Updating with proper learning rates
        predictions = tree.predict(features)
        for i in range(len(predictions)):
            predictions[i] += learning_rate * gamma_values[leaf_nodes[i]]

    # g) Predict using the decision trees and combine their results
    # Choosing the first sample as an example
    final_prediction = y_new + learning_rate * np.sum(gamma_example)
    print("The final prediction after combining results: ", final_prediction)


# For homework problem 5; OLS (linear regression) model using iris dataset
def prob5():
    # loading iris dataset from sklearn.datasets & initializing dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # a) outputting the correlation matrix & selecting 2 features with highest correlation
    # Calculate correlation matrix and two highest features
    correlation_matrix = df.corr()
    top_features = correlation_matrix['target'].abs().nlargest(2).index.tolist()
    # Output the correlation matrix and selected features
    pd.set_option('display.max_rows', None) # To show all rows and columns
    pd.set_option('display.max_columns', None)
    print("Correlation Matrix:\n", correlation_matrix)
    print("\nTwo top features:", top_features)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

    # b) training an OLS model using the selected top 2 features; report MAE
    # Split dataframe into training/testing (80:20)
    X = df[top_features] # Only selected features
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train linear regression model
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    y_pred = ols_model.predict(X_test)
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error (MAE):", mae)

    # c) training a quadratic model by transforming 2 features in polynomial features & training LR model
    # Transform features into polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False) # degree=2 for polynomial
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    # Train a linear regression model on polynomial features
    quadratic_model = LinearRegression()
    quadratic_model.fit(X_train_poly, y_train)
    # Make predictions on testing dataset
    y_pred_poly = quadratic_model.predict(X_test_poly)
    # Calculate Mean Absolute Error (MAE)
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    print("Mean Absolute Error (MAE) for quadratic model: ", mae_poly)


# For calculating SSE; helper function
def SSE(X, y):
    # Calculate the cluster centers
    cluster_centers = []
    for cluster_label in np.unique(y): # For each cluster label,
        cluster_centers.append(np.mean(X[y == cluster_label], axis=0))
    # Calculate the sum of squared errors
    sse = 0
    for i in range(len(X)):
        cluster_label = y[i]
        center = cluster_centers[cluster_label]
        sse += np.linalg.norm(np.array(X.iloc[i]) - np.array(center))**2
    return sse

# For homework problem 6; using clustering algorithms for two moons dataset
# Input is the dataframe of the two moons dataset
def prob6(dataset):
    # Variables for features and outputs
    X = dataset.iloc[:, :2] # features
    y = dataset.iloc[:, 2]  # outputs; true cluster

    # a) performing k-Means clustering
    kmeans = KMeans(n_clusters=2) # specify 2 clusters
    kmeans_pred = kmeans.fit_predict(X)
    # calculate SSE and misclassification rate
    # SSE (distortion):
    sse_kmeans = kmeans.inertia_
    print("K-means clustering SSE: ", sse_kmeans)
    # Cluster misclassification rate (silhouette score):
    silhouette_kmeans = silhouette_score(X, kmeans_pred) # misclassification rate
    accuracy_kmeans = accuracy_score(kmeans_pred, y)      # accuracy score
    print("K-means clustering silhouette score: ", silhouette_kmeans)
    print("K-means clustering accuracy score: ", accuracy_kmeans)

    # b) Performing agglomerative hierarchical clustering
    agg = AgglomerativeClustering(n_clusters=2)
    agg_pred = agg.fit_predict(X)
    # Calculate SSE and misclassification rate
    # SSE:
    sse_agg = SSE(X, agg_pred)
    print("Agglomerative hierarchical clustering SSE: ", sse_agg)
    # Misclassification rate (accuracy score):
    accuracy_agg = accuracy_score(agg_pred, y)
    silhouette_agg = silhouette_score(X, agg_pred)
    print("Agglomerative hierarchical clustering silhouette score: ", silhouette_agg)
    print("Agglomerative hierarchical clustering accuracy score: ", accuracy_agg)


if __name__ == "__main__":
    features, outputs = make_classification(n_samples=20, n_features=2, 
                                            n_redundant=0, n_classes=2, random_state=8)
    # Each method calls lead to designated problems
    prob2(features, outputs) # Problem 2
    prob3(features, outputs) # Problem 3
    prob4()                  # Problem 4
    prob5()                  # Problem 5
    # For problem 6; importing twomoons dataset
    twomoons_dataset = pd.read_csv("twomoons.csv")
    prob6(twomoons_dataset)     # Problem 6