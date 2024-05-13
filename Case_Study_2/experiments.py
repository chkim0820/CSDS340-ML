import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats.mstats import mquantiles
#you will have to pip install seaborn and colorcet
from sklearn.manifold import TSNE   
import seaborn as sns
import colorcet as cc

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))

def get_data(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM' : hh_mm_ss2seconds})
    # select features 
    selected_features = ['LAT', 'LON', 'SEQUENCE_DTTM', 'SPEED_OVER_GROUND' ,'COURSE_OVER_GROUND']
    # print(df[selected_features].head())
    X = df[selected_features].to_numpy()
    y = df['VID'].to_numpy()
    if not np.any(y):
        return X, None
    else:
        return X, y
    
def do_plots(datasets):
    fig, ax = plt.subplots(3,1)
    ax[0].scatter(datasets[0][:,0], datasets[0][:,1])
    ax[0].set_title("Dataset 1")
    ax[1].scatter(datasets[1][:,0], datasets[1][:,1])
    ax[1].set_title("Dataset 2")
    ax[2].scatter(datasets[2][:,0], datasets[2][:,1])
    ax[2].set_title("Dataset 3")
    plt.show()

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

def predictor_baseline_combined(X, y):
    # k-means with K = number of unique VIDs of set1
    K = 20 
    model = KMeans(n_clusters=K, random_state=123, n_init='auto').fit(X)
    # predict cluster numbers of each sample
    labels_pred = model.predict(X)
    rand_index_score = adjusted_rand_score(y, labels_pred)
    print(f'Get above this score for combined dataset: {rand_index_score:.4f}')

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
        print(f'Get above score {rand_index_score:.4f} for {file_name}')
        

def measure_performance(clusterer, x, y):
    if y is not None:
        print(adjusted_rand_score(y, clusterer.fit_predict(x)))
    else:
        clusterer.fit(x)

def measure_performance_sets(clusterer):
    datasets = []
    expected = []
    for file in ['Data/set1.csv', 'Data/set2.csv']:
        X, y = get_data(file)
        datasets.append(X)
        expected.append(y)
    for i, set in enumerate(datasets):
        scaler = preprocessing.RobustScaler()
        datasets[i] = scaler.fit_transform(set)
    for i, (set, output) in enumerate(zip(datasets, expected)):
        if output is not None:
            print(f"test performance for set {i+1}")
            print(adjusted_rand_score(output, clusterer.fit_predict(set)))
    # for i, set in enumerate(datasets):
    #     do_tsne(datasets[i], clusterer = clusterer, expected = expected[i])

def do_tsne(dataset, clusterer= None, expected=None):
    color_palette = sns.color_palette(cc.glasbey, n_colors=20)
    proj = TSNE().fit_transform(dataset)
    if expected is not None:
        plt.suptitle("Ground truth clusters")
        cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in expected]
        plt.scatter(*proj.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
        plt.figure()
    if clusterer is not None:

        color_palette = sns.color_palette(cc.glasbey, n_colors=len(np.unique(clusterer.labels_)))
        plt.suptitle("Clusterer clusters")
        cluster_colors = [color_palette[x] if x >= 0
                  else (0.5, 0.5, 0.5)
                  for x in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                         zip(cluster_colors, clusterer.probabilities_)]
        plt.scatter(*proj.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.show()



############################# Tune parameters ################################
def hdbscan(x):
    invCov = np.linalg.inv(np.cov(x.T)) + 0.0053 * np.eye(x.shape[1])
    # X = preprocessing.Normalizer().fit(X).transform(X)
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
    # out = cluster.fit_predict(x)
    return cluster



# main method
if __name__ == '__main__':
    DO_TSNE = True
    get_baseline_score()

    datasets = []
    expected = []
    clusterers= [] 
    for file in ['Data/set1.csv', 'Data/set2.csv', 'Data/set3noVID.csv']:
        X, y = get_data(file)
        datasets.append(X)
        expected.append(y)
    for i, set in enumerate(datasets):
        scaler = preprocessing.RobustScaler()
        datasets[i] = scaler.fit_transform(set)
    for i, labels in enumerate(expected):
        if labels is not None:
            label_enc = preprocessing.LabelEncoder()
            expected[i] = label_enc.fit_transform(labels)
    for i, (set, output) in enumerate(zip(datasets, expected)):
        clusterer = hdbscan(set)
        clusterers.append(clusterer)
        print(f"test performance for set {i+1}")
        measure_performance(clusterer, set, output)

    if DO_TSNE:
        for i, set in enumerate(datasets):
            do_tsne(datasets[i], clusterer = clusterers[i], expected = expected[i])

    do_plots(datasets)
    print("\n")
    print("-------Try combined datasetes-------")

    # trying different combinations of training sets
    datasets_combined = []
    expected_combined = []
    clusterers_combined = []
    for combo in [False, True]: # for combined datasets with or without 3rd dataset
        X, y = combine_datasets(combo)
        datasets_combined.append(X)
        expected_combined.append(y)
        predictor_baseline_combined(X, y)
    print("\n---------Performances-----------\n")
    for i, set in enumerate(datasets_combined):
        scaler = preprocessing.RobustScaler()
        datasets_combined[i] = scaler.fit_transform(set)
    for i, (set, output) in enumerate(zip(datasets_combined, expected_combined)):
        clusterer = hdbscan(set)
        clusterers_combined.append(clusterer)
        if output is not None:
            print(f"test performance for {i+1}th combined dataset:")
            measure_performance(clusterer, set, output) # performance w/in dataset
            print("test performance on separate sets:")
            measure_performance_sets(clusterer) # performance on set 1
            print("\n")