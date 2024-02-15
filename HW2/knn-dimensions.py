# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 2

import numpy as np
import matplotlib.pyplot as plt

# Generating the specified number of random data points in d dimensions
def generateRandomData(numData, d):
    data = np.random.uniform(low=-1, high=1, size=(numData, d))
    return data

# Returns the fraction of data points within <=1 Euclidean distance from origin
def hypersphereFraction(data, numPoints):
    # calculate the distance of each data point from the origin
    distances = np.linalg.norm(data, axis=1)
    withinSphere = np.sum(distances <= 1)
    sphereFraction = withinSphere / numPoints
    return sphereFraction

# Measures how close a nearest neighbor is relative to a randomly selected data point
# Mean dist. b/w a data point and its 1-nearest neighbor divided by mean dist. b/w any pair
def meanDistance(data):
    # Calculate pairwise distances between all data points
    pairDistances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    # Mean distance between a data point and its 1-nearest neighbor
    meanDistNN = np.mean(np.partition(pairDistances, 1, axis=1)[:, 1])
    # Mean distance between any pair of data points
    meanPairDist = np.mean(pairDistances[pairDistances != 0])
    # Mean distance between a data point and its 1-nearest neighbor divided by the mean distance between any pair of data points
    distance = meanDistNN / meanPairDist
    return distance

# Plot the calculated fractions within hypersphere
def plotFractions(dimensions, fractions):
    plt.figure()
    # Specifications
    plt.plot(dimensions, fractions, marker='o')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Fraction of Data Points within Unit Hypersphere')
    plt.title('Fraction of Data Points within Unit Hypersphere vs. Number of Dimensions')
    # Display
    plt.tight_layout()
    plt.show()

# Plot the mean distances between data points 
def plotDistances(dimensions, distances):
    plt.figure()
    # Specifications
    plt.plot(dimensions, distances, marker='o')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Mean NN Distance Ratio')
    plt.title('Mean NN Distance Ratio vs. Number of Dimensions')
    # Display
    plt.tight_layout()
    plt.show()

# def visualize(data, ratio):
#     # Plot the data points
#     plt.figure()
#     plt.scatter(data[:, 0], data[:, 1], label='Data Points')

#     # Plot the unit hypersphere
#     circle = plt.Circle((0, 0), 1, fill=False, label='Unit Hypersphere')
#     plt.gca().add_artist(circle) # Add circle on top of the plot above

#     # Add legend and title
#     plt.legend()
#     plt.title(f'Fraction of Points Within Unit Hypersphere: {ratio:.2f}')
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     # Display the plot
#     plt.axis('equal')
#     plt.show()
#     exit()

# Main method; calculate both fractions and mean distances
if __name__ == "__main__":
    # Generate data points w/ specified dimensions
    dimensions = range(2, 11) # Dimensions: 2 to 10
    numPoints = 1000 # Number of data points to generate
    fractions = [] # Stores the fractions within hypersphere
    meanDistances = [] # Stores mean NN distances
    for d in dimensions:
        # Generate 1000 random data points in d dimensions
        data = generateRandomData(numPoints, d)
        # Calculate fraction & mean distance
        fraction = hypersphereFraction(data, numPoints)
        meanDist = meanDistance(data)
        # Append to the lists to keep track
        fractions.append(fraction)
        meanDistances.append(meanDist)
    plotFractions(dimensions, fractions)
    plotDistances(dimensions, meanDistances)
    