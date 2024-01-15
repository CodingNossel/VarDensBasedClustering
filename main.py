import numpy as np
import copy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.decomposition import PCA

def loading_extern_datasets(file : str) -> np.ndarray:

    with open(file, 'r') as f:
        lines = f.readlines()

    coords_list = [list(map(float, line.split())) for line in lines]
    data_list = np.array(coords_list)


    return data_list


def do_hdbscan(default_data):
    hdb = HDBSCAN(store_centers='both', min_cluster_size=10, min_samples=1)
    hdb.fit(default_data)

    data = [dict(index=index, label=hdb.labels_[index], coord=value) for index, value in enumerate(default_data)]

    data.sort(key=lambda x: x['label'])
    linkage = hierarchy.linkage(hdb.centroids_, method='single')


    return data, linkage


def get_cutlist(linkage) -> list:

    cut = []

    for i in range(len(linkage) - 1):
        height = linkage[i][2]
        height_child = linkage[i + 1][2]
        height_diff = height - height_child
        cut.append((height_diff, height))


    return cut


def get_biggest_density_change(cut : list, quantile : float = 0.55) -> list:

    cut.sort(key=lambda tup: tup[0])
    # cut from the point where quantile is reached
    cut = cut[:int(len(cut) * quantile)]


    return cut


def compare_changes(shortend_cut : list, linkage, debug_mode : bool = False) -> list:
    changes = []
    for i in range(len(shortend_cut)):
        val_identifier = shortend_cut[i][1]

        for j in range(len(linkage)):
            compare_to = linkage[j][2]

            if val_identifier == compare_to:
                change_to = linkage[j][0]
                change_only_me = linkage[j][1]
                changes.append((change_only_me, change_to))
                if debug_mode:
                    print('changeTo: ', change_to, 'changeOnlyMe: ', change_only_me, 'valIdentifier: ', val_identifier,
                          'compareTo: ', compare_to)
                break


    return changes


def change_label(changes_in_data : list, result : list, debug_mode : bool = False) -> None:
    print(changes_in_data)
    for k in range(len(changes_in_data)):
        for element in result:
            if element['label'] == changes_in_data[k][0]:
                element['label'] = changes_in_data[k][1]
                if debug_mode:
                    print('change from: ', changes_in_data[k][0], '->', 'changed to: ', changes_in_data[k][1], 'point: ', element['coord'])


def plotting(data_values : list, data_labels : list) -> None:
    
    if np.shape(data_values)[1] >= 2:
        pca = PCA(n_components=2)
        projected = pca.fit_transform(data_values)
        plt.scatter(projected[:, 0], projected[:, 1], c=data_labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('nipy_spectral', 10))
        plt.xlabel('X Coords')
        plt.ylabel('Y Coords')
        plt.colorbar()
        plt.title('2D Projection of Dataset using PCA')
        plt.show()


#blobies
"""
centers = [[1, 1], [-1, -1], [1, -1], [20, 20], [20, 21], [21, 20], [21, 21]]
X, labels_true = make_blobs(
     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)
"""
# digits
"""
X, labels_true = datasets.load_digits(
    return_X_y=True
)
"""

# iris
"""
X, labels_true = datasets.load_iris(
    return_X_y=True
)
"""

data = loading_extern_datasets('R15.txt')
data_values = [data[i][0:2] for i in range(len(data))]
data_labels = [data[i][2] for i in range(len(data))]


X, z_linkage = do_hdbscan(data)
dp_data = copy.deepcopy(X)
shortend_cutlist = get_biggest_density_change(get_cutlist(z_linkage), 0.95) #compramised_cutlist
result = copy.deepcopy(dp_data)

changes = compare_changes(shortend_cutlist, z_linkage, True)
change_label(changes, result)
sorted(shortend_cutlist, key=lambda tup: tup[1])

plotting(data_values, data_labels)

#BRO IDK WHY IT DOESNT HAVE THE SAME PLOT AS THE ONE ABOVE
data_labels = [result[i]['label'] for i in range(len(result))]
data_values = [result[i]['coord'] for i in range(len(result))]
plotting(data_values, data_labels)


# Dendrogram for Christian DO NOT USE OTHERWISE
dendrogram = hierarchy.dendrogram(z_linkage)
plt.show()