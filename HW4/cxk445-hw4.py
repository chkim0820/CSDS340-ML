# Name: Chaehyeon Kim (cxk445)
# Description: For CSDS 340 HW 4 problems

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
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score, accuracy_score


# For homework problem 2; ensemble model bagging technique
def prob2():
    # a) Generating a random dataset with 20 samples (2 features & 1 binary output)
    features, outputs = make_classification(n_samples=20, n_features=2, 
                                            n_redundant=0, n_classes=2, random_state=8)
    dataset = pd.DataFrame(features, columns=['Feature1', 'Feature2'])
    dataset['Label'] = outputs # dataset contains the generated random datset
    
    # b) Generating 10 training datsets (20 samples each) by sampling with repetition
    matrix = np.zeros((10, 20), dtype=int) # As specified, the output would be a 10 x 20 matrix
    for i in range(10):
        indices = np.random.choice(a=20, size=20, replace=True)
        matrix[i, :] = indices
    
    # c) Highlighting all duplicated entries in each training dataset
    matrix_highlighted = np.array([[np.sum(row == value) > 1 for value in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(matrix_highlighted, cmap="Greys", interpolation='nearest', aspect='auto')
    for i in range(matrix_highlighted.shape[0]):
        for j in range(matrix_highlighted.shape[1]):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="red" if matrix_highlighted[i, j] else "black")
    ax.set_title('Highlighted Duplicates in Each Dataset')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Dataset Number')
    # plt.show()

    # d) Training 10 classifier models on the 10 datasets
    models = []
    accuracies = []
    for i in range(10):
    # Create a new model
        model = DecisionTreeClassifier(random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(features[matrix[i]], outputs[matrix[i]], test_size=0.2, random_state=42)
        # Train the model on the dataset
        model.fit(X_train, y_train)
        # Add the trained model to the list
        models.append(model)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    # e) Using majority voting to combine the results
    predictions = [model.predict(X_test[[0]])[0] for model in models]
    final_prediction = max(set(predictions), key=predictions.count)
    ensemble_predictions = [max(set([model.predict([sample])[0] for model in models]), key=[model.predict([sample])[0] for model in models].count) for sample in X_test]
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    print("Ensemble model accuracy:", ensemble_accuracy)
    
    # returning the random generated dataset for later problems
    return features, outputs


# For homework problem 3; ensemble model boosting technique
def prob3(features, outputs):
    # a) a random dataset with 20 samples with relabeled outputs (-1 and 1)
    outputs[outputs == 0] = -1

    # b) training a weak learner with uniform weights for samples
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(features, outputs)
    feature_index = stump.tree_.feature[0]
    threshold = stump.tree_.threshold[0]

    # c) calculating the coefficient and updated weights
    # Initialize weights (uniform)
    weights = np.ones(len(outputs)) / len(outputs)
    print("Initial weights:", weights)
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(features, outputs, sample_weight=weights)
    predictions = stump.predict(features)
    error = np.sum(weights * (predictions != outputs))
    alpha = 0.5 * np.log((1 - error) / error)  # coefficient αj
    # Updating weights
    weights *= np.exp(-alpha * outputs * predictions)
    weights /= np.sum(weights)
    print("Updated weights:", weights)

    # d) training 9 more weak learners; output feature, threshold, and αj
    n_learners = 10
    learners = []
    alphas = []
    for _ in range(n_learners):
        # training a weak learner on the weighted dataset
        stump = DecisionTreeClassifier(random_state=42)
        print("Training stump with weights:", weights)
        stump.fit(features, outputs, sample_weight=weights)
        predictions = stump.predict(features)
        # Calculating error and coefficient
        error = np.sum(weights * (predictions != outputs))
        alpha = 0.5 * np.log((1 - error) / error)
        alphas.append(alpha)
        # Updating weights
        weights *= np.exp(-alpha * outputs * predictions)
        weights /= np.sum(weights)
        print("Updated weights:", weights)
        learners.append((stump, alpha))
        # Output: feature, threshold of split, and coefficient αj
        feature_index = stump.tree_.feature[0]
        threshold = stump.tree_.threshold[0]
        print(f"Learner {_ + 1}:")
        print("  Feature:", feature_index)
        print("  Threshold:", threshold)
        print("  Alpha:", alpha)
        print()

    # e) predicting using each weak learner & combining results
    # Sample test data
    sample_test_data = features[0]  # Assuming you want to use the first sample for testing

    # Initialize final prediction
    final_prediction = 0

    # Predict using each weak learner and combine with alpha
    for learner, alpha in zip(learners, alphas):
        model, _ = learner
        prediction = model.predict([sample_test_data])[0]
        final_prediction += alpha * prediction

    # Apply sign function to get the final prediction
    final_prediction = np.sign(final_prediction)

    print("Final prediction using all weak learners:", final_prediction)


# For homework problem 4; gradient boosting technique of ensemble model
def prob4(features, outputs):
    # a) a random dataset with 20 samples; reusing from previous problems
    
    # b) outputting the log-odds of the dataset
    log_odds = np.log(np.divide(outputs, 1 - outputs))
   
    # c) calculating & outputting the residual terms for each training data
    predictions_0th_tree = np.log(np.divide(outputs, 1 - outputs))
    residuals = outputs - (1 / (1 + np.exp(-predictions_0th_tree)))
    print("Residuals for each training data point:")
    print(residuals)
   
    # d) fitting a decision tree to the residuals; output: γj1 for each leaf node
    # As the problem stated, the max depth of tree is fixed to 2
    tree_residuals = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_residuals.fit(features, residuals)
    leaf_nodes = tree_residuals.apply(features)
    leaf_values = {leaf: tree_residuals.tree_.value[leaf][0][0] for leaf in np.unique(leaf_nodes)}
    print("Leaf nodes and their corresponding γj1 values:")
    for leaf, value in leaf_values.items():
        print(f"Leaf node {leaf}: γj1 = {value}")
    
    # e) choosing at least 2 samples from each leaf node & outputting predicted values
    # Choose at least two samples from each leaf node of decision tree 1
    samples_per_leaf = {}
    for leaf in np.unique(leaf_nodes):
        samples_per_leaf[leaf] = []
    for i, x in enumerate(features):
        leaf = tree_residuals.apply([x])[0]
        samples_per_leaf[leaf].append(i)
    print("Chosen samples and their predicted values for each leaf node:")
    for leaf, samples in samples_per_leaf.items():
        chosen_samples = samples[:2]  # Choose at least two samples from each leaf node
        for sample_idx in chosen_samples:
            sample = features[sample_idx]
            predicted_value = tree_residuals.predict([sample])[0]
            print(f"Leaf node {leaf}, Sample {sample_idx}: Predicted value = {predicted_value}")

    # f) continuing the process & training 9 more decision trees
    # for each decision tree, output the values corresponding to the leaf node
    # Initialize residuals for the first iteration
    residuals_k = residuals.copy()

    # Train 9 more decision trees
    for k in range(1, 10):
        # Fit a decision tree to the residuals
        tree_residuals_k = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_residuals_k.fit(X, residuals_k)

        # Get the leaf nodes and their corresponding values (γjk)
        leaf_nodes_k = tree_residuals_k.apply(X)
        leaf_values_k = {leaf: tree_residuals_k.tree_.value[leaf][0][0] for leaf in np.unique(leaf_nodes_k)}

        # Output the leaf nodes and their corresponding values for decision tree k
        print(f"Decision Tree {k} - Leaf nodes and their corresponding γjk values:")
        for leaf, value in leaf_values_k.items():
            print(f"  Leaf node {leaf}: γjk = {value}")

        # Update residuals for the next iteration
        residuals_k = residuals_k - tree_residuals_k.predict(X)

        # If residuals are all zeros, break the loop
        if np.all(residuals_k == 0):
            print("All residuals are zero. Stopping the process.")
            break

    # g) predicting using the deciison trees combined
    # Sample test data (example sample)
    sample_test_data = features[0]
    final_prediction = 0
    for k in range(10):
        tree_residuals_k = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree_residuals_k.fit(features, residuals_k)
        prediction_k = tree_residuals_k.predict([sample_test_data])[0]
        final_prediction += prediction_k
    final_prediction = np.sign(final_prediction)


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


# For calculating SSE
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
    # For problem 6; importing twomoons dataset
    twomoons_dataset = pd.read_csv("twomoons.csv")
    # Each method calls lead to designated problems
    # features, outputs = prob2() # Problem 2
    # prob3(features, outputs)    # Problem 3
    # prob4(features, outputs)    # Problem 4
    prob5()                     # Problem 5
    # prob6(twomoons_dataset)     # Problem 6