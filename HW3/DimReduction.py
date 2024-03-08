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
        
        # 3f) Construct the projection matrix
        projMatrix = selectedVectors
        print("Projection Matrix: ", projMatrix)

# For applying dimensionality reduction using LDA
class LDA:
    def main(self, data, outputs):
        # 4a) Standardizing the d-dimensional dataset
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        # 4b) Computing the d-dimensional mean vector for each class
        meanVectors = []
        classes = np.unique(outputs)
        for cls in classes:
            classMean = data[outputs == cls].mean(axis=0)
            meanVectors.append(classMean)
        print("\nMean Vectors:")
        for i, mean in enumerate(meanVectors):
            print(f"Class {i+1} Mean Vector:")
            print(mean)

        # 4c) Constructing the within/between-class scatter matrices
        d = data.shape[1]
        Sw = np.zeros((d, d))
        for cl, mean in zip(classes, meanVectors):
            classScatter = np.zeros((d, d))
            for x in data[outputs == cl]:
                x = x.reshape(-1, 1)  # make column vector
                mean = mean.reshape(-1, 1)  # make column vector
                classScatter += (x - mean).dot((x - mean).T)
            Sw += classScatter
        print("Within-class Scatter Matrix", Sw)
        # Constructing the between-class scatter matrix (Sb)
        overallMean = data.mean(axis=0)
        Sb = np.zeros((d, d))
        for cl, meanVec in zip(classes, meanVectors): # FIX?
            n = data[outputs == cl].shape[0] # Number of samples in current class
            meanVec = meanVec.reshape(-1, 1)  # make column vector
            overallMean = overallMean.reshape(-1, 1)  # make column vector
            Sb += n * (meanVec - overallMean).dot((meanVec - overallMean).T)
        print("Between class scatter matrix:", Sb)

        # 4d) Computing the eigenvalues and vectors of the matrix
        SwInv = np.linalg.inv(Sw) # Computing the inverse of Sw
        SwInvSb = np.dot(SwInv, Sb) # Computing the matrix
        eigenvalues, eigenvectors = np.linalg.eig(SwInvSb)
        print("\nEigenvalues of SwInvSb:")
        print(eigenvalues)
        print("\nEigenvectors of SwInvSb:")
        print(eigenvectors)

        # 4e) Sorting the eigenvalues and selecting d/2 largest eigenvalues
        k = int(len(eigenvalues) / 2) # k = (d / 2)
        ind = np.argsort(eigenvalues)[::-1]
        sortedVals = eigenvalues[ind]
        sortedVectors = eigenvectors[:, ind]
        # Selecting k = d/2 largest eigenvalues
        k = int(len(eigenvalues) / 2)
        selectedVals = sortedVals[:k]
        selectedVectors = sortedVectors[:, :k]
        print("Largest eigenvalues: ", selectedVals, '\n',
              "Largest eigenvectors: ", selectedVectors)

        # 4f) Constructing the projection matrix
        projMatrix = selectedVectors
        print("Projection Matrix: ", projMatrix)


if __name__ == "__main__":
    # Process the wine dataset
    dataset = pd.read_csv("wine_dataset.csv") # Import dataset
    features = dataset.iloc[:, :-1] # All features
    classes = dataset.iloc[:, -1] # Output classes
    # Number 3
    # pca = PCA()
    # pca.main(dataset)
    # Number 4
    lda = LDA()
    lda.main(features, classes) # excluding the target label
