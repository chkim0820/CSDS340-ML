# For CSDS 340 HW 3 Problem 3/4; applying dimensionality reduction using PCA & LDA
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# For applying dimensionality reduction using PCA
class PCA:
    # For completing each task
    def main(self, data):
        # 3a) Dropping 'style' column; now excludes output
        data = data.drop('style', axis=1) # Style contains the outputs
        print("Updated dimension of the data: ", len(data.columns))
        # 3b) Standardizing the dataset
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        # 3c) Constructing the covariance matrix
        covMatrix = np.cov(data, rowvar=False)
        print("Covariance matrix: ", covMatrix)
        # 3d) Performing eigen-decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
        print("Reporting the eigenvalues:", eigenvalues, '\n', 
              "Reporting the eigenvectors:", eigenvectors) # 12 total
        # 3e) Sorting the eigenvalues and selecting k=d/2 largest eigenvalues
        ind = np.argsort(eigenvalues)[::-1]
        sortedVals = eigenvalues[ind]
        sortedVectors = eigenvectors[:, ind]
        # Selecting k = d/2 largest eigenvalues
        k = int(len(eigenvalues) / 2)
        selectedVals = sortedVals[:k]
        selectedVectors = sortedVectors[:, :k]
        print("Largest eigenvalues: ", selectedVals, '\n',
              "Largest eigenvectors: ", selectedVectors)
        # Construct the projection matrix
        projMatrix = selectedVectors
        print("Projection Matrix: ", projMatrix)

# For applying dimensionality reduction using LDA
class LDA:
    def main(self, data, outputs):
        # 4a) Standardizing the d-dimensional dataset
        scaler = StandardScaler()
        scaledData = scaler.fit_transform(data)
        # 4b) Computing the d-dimensional mean vector for each class
        meanVectors = []
        classes = np.unique(outputs)
        for cls in classes:
            classMean = scaledData[outputs == cls].mean(axis=0)
            meanVectors.append(classMean)
        print("\nMean Vectors:")
        for i, mean in enumerate(meanVectors):
            print(f"Class {i+1} Mean Vector:")
            print(mean)
        # 4c) Constructing the within-class scatter matrix (Sw)
        Sw = np.zeros((data.shape[1], data.shape[1]))
        for c, meanVec in zip(classes, meanVectors):
            classScatter = np.zeros((data.shape[1], data.shape[1]))
            for x in data[outputs == c]:
                x = x.reshape(-1, 1)  # make column vector
                meanVec = meanVec.reshape(-1, 1)  # make column vector
                classScatter += (x - meanVec).dot((x - meanVec).T)
            Sw += classScatter
        # Constructing the between-class scatter matrix (Sb)
        overallMean = scaledData.mean(axis=0)
        Sb = np.zeros((data.shape[1], data.shape[1]))
        for c, meanVec in zip(classes, meanVectors):
            n = data[outputs == c].shape[0]
            meanVec = meanVec.reshape(-1, 1)  # make column vector
            overallMean = overallMean.reshape(-1, 1)  # make column vector
            Sb += n * (meanVec - overallMean).dot((meanVec - overallMean).T)
        # 4d) Computing the eigenvalues and vectors of the matrix
        SwInv = np.linalg.inv(Sw) # Computing the inverse of Sw
        SwInvSb = np.dot(SwInv, Sb) # Computing the matrix
        eigenvalues, eigenvectors = np.linalg.eig(SwInvSb)
        print("\nEigenvalues:")
        print(eigenvalues)
        print("\nEigenvectors:")
        print(eigenvectors)
        # 4e) Sorting the eigenvalues and selecting d/2 largest eigenvalues
        ind = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[ind]
        eigenvectors = eigenvectors[:, ind]
        print("\nSorted Eigenvalues:")
        print(eigenvalues)
        print("\nSorted Eigenvectors:")
        print(eigenvectors)
        # 4f) Constructing the projection matrix
        k = len(eigenvalues) // 2
        projectionMatrix = eigenvectors[:, :k]
        print("\nProjection Matrix:")
        print(projectionMatrix)


if __name__ == "__main__":
    # Process the wine dataset
    dataset = pd.read_csv("wine_dataset.csv") # Import dataset
    features = dataset.iloc[:, :-1] # All features
    classes = dataset.iloc[:, -1] # Output classes
    # Number 3
    pca = PCA()
    pca.main(dataset)
    # Number 4
    # lda = LDA()
    # lda.main(features, classes) # excluding the target label
