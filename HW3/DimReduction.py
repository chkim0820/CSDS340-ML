# For CSDS 340 HW 3 Problem 3/4; applying dimensionality reduction using PCA & LDA
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# For applying dimensionality reduction using PCA
class PCA:
    # For completing each task
    def main(self, data):
        # 3a) Dropping 'style' column
        data = data.drop('style', axis=1)
        # 3b) Standardizing the dataset
        stdsc = StandardScaler()
        stdsc.fit(data)
        data = stdsc.transform(data)
        # 3c) Constructing the covariance matrix
        matrix = np.cov(data, rowvar=False)
        # 3d) Performing eigen-decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        print("Reporting the eigenvalues:", eigenvalues, '\n', 
              "Reporting the eigenvectors:", eigenvectors) # 12 total
        # 3e) Sorting the eigenvalues and selecting d/2 largest eigenvalues
        ind = np.argsort(eigenvalues)[::-1]
        sortedVals = eigenvalues[ind]
        sortedVectors = eigenvectors[:, ind]
        k = len(eigenvalues) - 2
        selectedVals = sortedVals[:k]
        selectedVectors = sortedVectors[:, :k]
        print("Largest eigenvalues: ", selectedVals, '\n',
              "Largest eigenvectors: ", selectedVectors)
        # Construct the projection matrix
        projMatrix = selectedVectors
        print("Projection Matrix: ", projMatrix)

# For applying dimensionality reduction using LDA
class LDA:
    # For completing each task
    def main(self, data):
        

if __name__ == "__main__":
    # Process the wine dataset
    dataset = pd.read_csv("wine_dataset.csv") # Import dataset
    # Number 3
    pca = PCA()
    pca.main(dataset)
    # Number 4
    lda = LDA()
    lda.main(dataset)
