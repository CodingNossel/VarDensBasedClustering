import numpy as np
import copy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.decomposition import PCA

'''
This file performs the Edge Quantile Cut - HDBScan on a dataset of choice.
'''


def loading_extern_datasets(file: str) -> np.ndarray:
    """
    loads a dataset from file given in string

    returns the dataset as a list
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    coords_list = [list(map(float, line.split())) for line in lines]
    data_list = np.array(coords_list)

    return data_list


def do_hdbscan(default_data):
    """
    performs hdbscan on a given list

    returns the data and linkage matrix
    """
    hdb = HDBSCAN(store_centers='both', min_cluster_size=10, min_samples=1)
    hdb.fit(default_data)

    data = [dict(index=index, label=hdb.labels_[index], coord=value) for index, value in enumerate(default_data)]

    data.sort(key=lambda x: x['label'])
    linkage = hierarchy.linkage(hdb.centroids_, method='single')

    return data, linkage


def get_cutlist(linkage) -> list:
    """
    calculates the height differences between elements of the linkage list

    returns the cutlist
    """

    cut = []

    for i in range(len(linkage) - 1):
        height = linkage[i][2]
        height_child = linkage[i + 1][2]
        height_diff = height - height_child
        cut.append((height_diff, height))

    return cut


def get_biggest_height_change(cut: list, quantile: float = 0.55) -> list:
    """
    "sorts the cutlist based on height

    finds the biggest density change in the cutlist based on quantile, default = 0.55,
    and cuts the points after"

    ^ yeah no to this either, it now simply calculates the threshold for the dendrogram to cut the clusters and start
    coloring them differently

    returns the cutlist
    """

    cut.sort(key=lambda tup: tup[1])
    # cut from the point where quantile is reached
    tmp = cut[len(cut) - 1][1] * quantile

    return tmp


def compare_changes(shortend_cut: list, linkage, debug_mode: bool = False) -> dict:
    """
    "takes the cutlist after the density changes and performs them on the linkage list forming clusters"

    ^ yeah it doesn't do that anymore, now the dendrogram provides us with the information about when we cut the clusters
    doing so by color

    returns changes
    """

    return hierarchy.dendrogram(Z=linkage, color_threshold=shortend_cut)


def change_label(changes_in_data: dict, result: list, debug_mode: bool = False) -> None:
    """
    changes the label of the data and saves them in a new list
    """

    """
    want to know what this does? well it's the most horrible compatibility layer I could humanly think of to make this
    do what it's supposed to do and even then I'm sure it only does it by chance.

    Since HDSCAN, Linkage strip away way to much information and we realized this way to late, I had to somehow think of
    another way to get the information back. 

    Now the dendrogram provides us with the information of when we cut off the clusters and update their labels.

    basically all of this is somehow taking the information in the dendrogram and frankensteining them together in a way
    that allows us to change the cluster labels based on the colors the dendrogram assigns them. clusters with C0
    are always their own cluster and therefore are not combined. 

    The only reason we reassign all of them again is so that output image is correctly scaled. since we basically blow a 
    huge hole into our scale when we combine multiple clusters together. So we have to adjust the scale.

    Why this way you ask? Well because it's 01:04 on a tuesday, 6 hours before the project is due and I don't know even 
    more.  
    """

    tmp = []
    for i in range(len(changes_in_data['ivl'])):
        tmp.append((changes_in_data['ivl'][i], changes_in_data['leaves_color_list'][i]))

    values = set(map(lambda x: x[1], tmp))
    values.remove('C0')
    newlist = [[y[0] for y in tmp if y[1] == x] for x in values]

    hold = [[y[0] for y in tmp if y[1] == 'C0']]

    color_used = len(changes_in_data['leaves_color_list']) + 1
    print(result)

    for element in hold:
        for cluster in element:
            for point in result:
                if point['label'] == int(cluster):
                    point['label'] = color_used
            color_used += 1
            print('color changed to: ', color_used)

    for element in newlist:
        for cluster in element:
            for point in result:
                if point['label'] == int(cluster):
                    point['label'] = color_used


def plotting(data_values: list, data_labels: list) -> None:
    """
    plots the data
    """
    if np.shape(data_values)[1] >= 2:
        projected = np.array(data_values)

        plt.scatter(projected[:, 0], projected[:, 1], c=data_labels, edgecolor='none', alpha=0.8,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))
        plt.xlabel('X Coords')
        plt.ylabel('Y Coords')
        plt.colorbar()
        plt.title('2D Projection of Dataset using PCA')
        plt.show()


''' 
-------------------------------------------------------------------
 From here on forward we load data sets and visualize our results.
-------------------------------------------------------------------
'''
# blobies
"""
centers = [[1, 1], [-1, -1], [1, -1], [20, 20], [20, 21], [21, 20], [21, 21]]
# blobies
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
# )
"""

# digits
"""
# data, labels_true = datasets.load_digits(
#     return_X_y=True
# )
"""

# iris
"""
# X, labels_true = datasets.load_iris(
#     return_X_y=True
# )
"""

data = loading_extern_datasets('R15.txt')
data_values = [data[i][0:2] for i in range(len(data))]
data_labels = [data[i][2] for i in range(len(data))]

X, z_linkage = do_hdbscan(data)
dp_data = copy.deepcopy(X)
shortend_cutlist = get_biggest_height_change(get_cutlist(z_linkage), 0.55)  # compramised_cutlist
result = copy.deepcopy(dp_data)

changes = compare_changes(shortend_cutlist, z_linkage, True)
change_label(changes, result, True)
# sorted(shortend_cutlist, key=lambda tup: tup[1])

plotting(data_values, data_labels)

# BRO IDK WHY IT DOESN'T HAVE THE SAME PLOT AS THE ONE ABOVE
data_labels = [result[i]['label'] for i in range(len(result))]
data_values = [result[i]['coord'] for i in range(len(result))]

plotting(data_values, data_labels)

# Dendrogram for Christian DO NOT USE OTHERWISE
dendrogram = hierarchy.dendrogram(z_linkage)
plt.show()
