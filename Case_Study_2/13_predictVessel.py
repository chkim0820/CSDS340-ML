import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))


def predictor_baseline(csv_path):
    # load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardization 
    X = preprocessing.StandardScaler().fit(X).transform(X)
    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    return labels_pred


def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = './Data/' + file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')


def evaluate():
    csv_path = './Data/set3.csv'
    labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
    labels_pred = predictor(csv_path)
    rand_index_score = adjusted_rand_score(labels_true, labels_pred)
    print(f'Adjusted Rand Index Score of set3.csv: {rand_index_score:.4f}')


def predictor(csv_path):
    # fill your code here
    return labels_pred

# BELOW CODES ARE EDITED

def split_dataset(df1, df2):
    # No VID duplicates across different days for this particular dataset
    # So skip the step of reassigning VIDs for the combined data
    combined_df = pd.concat([df1, df2], axis=0)
    # Extracting input features and output labels
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    X = combined_df[selected_features].to_numpy()
    y = combined_df['VID'].to_numpy()
    # Creating a train/test data while maintaining same vessels in one or the other
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    train_ind, test_ind = next(splitter.split(X, y, y)) # keep same VIDs together
    X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
    return X_train, X_test, y_train, y_test
    
def agglomerative(X, y):
    # Standardize data
    X = preprocessing.StandardScaler().fit(X).transform(X)
    X = preprocessing.Normalizer().fit(X).transform(X)
    # Range of cluster numbers to try
    cluster_range = range(1, 50)
    rand_scores = []
    for n_clusters in cluster_range:
        # Perform agglomerative clustering
        cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_pred = cluster.fit_predict(X)
        rand_scores.append(adjusted_rand_score(y, cluster_pred))
    # Find the optimal number of clusters
    optimal_n_clusters = cluster_range[np.argmax(rand_scores)]
    print("Optimal number of clusters:", optimal_n_clusters)
    # Final clustering with optimal number of clusters
    cluster_final = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
    predictions = cluster_final.fit_predict(X)
    print("Rand Score for Agglomerative Clustering: ", adjusted_rand_score(y, predictions))
    return cluster_final

def train_model():
    # train with 'set1.csv'
    set1_df = pd.read_csv('./Data/set1.csv', converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    set2_df = pd.read_csv('./Data/set2.csv', converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # combine and split the dataset into training and testing data
    X_train, X_test, y_train, y_test = split_dataset(set1_df, set2_df)    

# for testing agglomeartive clusterin
def testing_agglomerative():
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    # testing with set1
    print("Testing set 1 with agglomerative clustering algorithm")
    set1_df = pd.read_csv('./Data/set1.csv', converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    X = set1_df[selected_features].to_numpy()
    y = set1_df['VID'].to_numpy()
    agglomerative(X, y)
    # testing with set2
    print("Testing set 2 with agglomerative clustering algorithm")
    set2_df = pd.read_csv('./Data/set2.csv', converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    X = set2_df[selected_features].to_numpy()
    y = set2_df['VID'].to_numpy()
    agglomerative(X, y)
    # on combined dataset
    X_train, X_test, y_train, y_test = split_dataset(set1_df, set2_df)
    agglomerative(X_test, y_test)
    agglomerative(X_train, y_train)

def hdbscan(x,y):
    invCov = np.linalg.inv(np.cov(x.T)) + 0.0053 * np.eye(x.shape[1])
    #X = preprocessing.Normalizer().fit(X).transform(X)
    classifiers = {}
    neighbor_classifier = NearestNeighbors(metric = 'mahalanobis', metric_params={'VI': invCov})
    neighbor_classifier.fit(x)
    neighbor_distances, _ = neighbor_classifier.kneighbors(x)
    quantiles = mquantiles(neighbor_distances)
    iqr = quantiles[2]-quantiles[0]
    estimated_eps = quantiles[2] + 1.65 * iqr
    minSize = int(0.0005 * len(x))
    min_samples_estimate = minSize if minSize >= 10 else 10
    cluster = HDBSCAN(metric = 'mahalanobis', metric_params={'VI': invCov}, 
                      min_cluster_size = 20, cluster_selection_epsilon = estimated_eps, 
                      min_samples=min_samples_estimate, algorithm='brute', n_jobs=-1)
    classifiers[cluster] = adjusted_rand_score(y, cluster.fit_predict(x))
    best_cluster = max(classifiers, key = classifiers.get)
    print('best rand score:', classifiers[best_cluster])
    return best_cluster

def test_hdbscan():
    #https://www.mdpi.com/2077-1312/9/6/566 
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    # testing with set1
    print("Testing set 1 with DBSCAN clustering algorithm")
    set1_df = pd.read_csv('./Data/set1.csv', converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    X = set1_df[selected_features].to_numpy()
    y = set1_df['VID'].to_numpy()
    normalizer = preprocessing.StandardScaler()
    x = normalizer.fit_transform(X)
    # class1 = hdbscan(x, y)
    # plot_clusters(x, y, class1)
    # testing with set2
    print("Testing set 2 with DBSCAN clustering algorithm")
    set2_df = pd.read_csv('./Data/set2.csv', converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    X2 = set2_df[selected_features].to_numpy()
    y2 = set2_df['VID'].to_numpy()
    normalizer2 = preprocessing.StandardScaler()
    x2 = normalizer2.fit_transform(X2)
    # class2 = hdbscan(x2, y2)
    # plot_clusters(x2, y2, class2)

    # testing with combined set
    X3, y3 = combine_datasets(False)
    normalizer3 = preprocessing.RobustScaler()
    x3 = normalizer3.fit_transform(X3)
    clusterer = hdbscan(x3, y3)
    plot_clusters(x, y, clusterer)
    plot_clusters(x2, y2, clusterer)
    plot_clusters(x3, y3, clusterer)

def combine_datasets(include_3rd=False):
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    files = ['Data/set1.csv', 'Data/set2.csv']
    if include_3rd:
        files.append('Data/set3noVID.csv')
    list_df = []
    for file in files:
        data = pd.read_csv(file, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
        list_df.append(data)
    combined_df = pd.concat(list_df, axis=0)
    X = combined_df[selected_features].to_numpy()
    y = combined_df['VID'].to_numpy()
    return X, y

def plot_clusters(X, y, classifier):
    plt.scatter(X[:,1], X[:,2])
    labels = classifier.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(y)
    print(labels)
    print(f"Adjusted Rand Index: {adjusted_rand_score(y, labels):.3f}")
    unique_labels = set(labels)
    #core_samples_mask = np.zeros_like(labels, dtype=bool)
    #core_samples_mask[classifier.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k
        xy = X[class_member_mask]# & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        ) 
        """   
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        ) """

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()
    #plot true clusters


if __name__=="__main__":
    get_baseline_score()
   # testing_agglomerative()
    test_hdbscan()
    # train_model()
    # evaluate() ; UNCOMMENT for submission