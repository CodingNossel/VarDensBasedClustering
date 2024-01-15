import numpy as np
import copy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.decomposition import PCA

def loading_datasets(file : str):

    with open(file, 'r') as f:
        lines = f.readlines()

    coords_list = [list(map(float, line.split())) for line in lines]
    data_list = np.array(coords_list)

    return data_list


def do_hdbscan(default_data : list):
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


def compare_changes(shortend_cut : list, linkage, debug_mode = False):
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


def change_label(changes_in_data : list, result : list, debug_mode = False):
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


# TODO: add way to load different datasets
centers = [[1, 1], [-1, -1], [1, -1], [20, 20], [20, 21], [21, 20], [21, 21]]
# blobies
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
# )
# digits
X, labels_true = datasets.load_digits(
    return_X_y=True
)
# iris
# X, labels_true = datasets.load_iris(
#     return_X_y=True
# )

data, z_linkage = do_hdbscan(X)
dp_data = copy.deepcopy(data)
shortend_cutlist = get_biggest_density_change(get_cutlist(z_linkage), 0.55) #compramised_cutlist
result = copy.deepcopy(dp_data)

changes = compare_changes(shortend_cutlist, z_linkage, True)
change_label(changes, result, True)
sorted(shortend_cutlist, key=lambda tup: tup[1])

# hierarchyTreeCentroids = hierarchy.to_tree(z)

# Create hierarchy from centroids and pass to tree for further use. Possible to change parameters here
# h = hierarchy.linkage(hdb.medoids_, method='single')

# hierarchyTreeMedoids = hierarchy.to_tree(h)

# TODO: use whichever fits get for getting the coordinates
# https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#module-scipy.cluster.hierarchy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_tree.html#scipy.cluster.hierarchy.to_tree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html#scipy.cluster.hierarchy.cut_tree
# this exists as well, maybe it will held

# TODO: implement Edge Quantile Cut

# TODO: implement Alpha Shape Cut


for i in range(len(data)):
    print('data: ', data[i], 'compare_result: ', data[i]['label'] == result[i]['label'])


data_labels = [result[i]['label'] for i in range(len(result))]
data_values = [result[i]['coord'] for i in range(len(result))]

default_labels = [data[i]['label'] for i in range(len(data))]
default_values = [data[i]['coord'] for i in range(len(data))]

print(np.shape(data_values), np.shape(data_labels))
print(data_values)
#plotting(default_values, default_labels)
#plotting(data_values, data_labels)

# Dendrogram for Christian DO NOT USE OTHERWISE
#dendrogram = hierarchy.dendrogram(z_linkage)
#plt.show()

data = loading_datasets('R15.txt')
data_values = [data[i][0:2] for i in range(len(data))]
data_labels = [data[i][2] for i in range(len(data))]
print()
print(type(X), np.shape(X))

print(type(data), np.shape(data))
print(np.shape(data_values), np.shape(data_labels))
#print(data)
print(data_values)

X, z_linkage = do_hdbscan(data)
dp_data = copy.deepcopy(X)
shortend_cutlist = get_biggest_density_change(get_cutlist(z_linkage), 0.55) #compramised_cutlist
result = copy.deepcopy(dp_data)

changes = compare_changes(shortend_cutlist, z_linkage, True)
change_label(changes, result, True)
sorted(shortend_cutlist, key=lambda tup: tup[1])



plotting(data_values, data_labels)

data_labels = [result[i]['label'] for i in range(len(result))]
data_values = [result[i]['coord'] for i in range(len(result))]
plotting(data_values, data_labels)