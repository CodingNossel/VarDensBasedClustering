from typing import Any

import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

# TODO: add way to load different datasets
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

# run HDBSCAN on Data. Possible to change parameters here
hdb = HDBSCAN(store_centers='both')
hdb.fit(X)

data = []
index = 0

for value in X:
    data.append(dict(index=index, label=hdb.labels_[index], coord=value))
    index = index + 1

data.sort(key=lambda x: x['label'])

# Create hierarchy from centroids and pass to tree for further use. Possible to change parameters here
z = hierarchy.linkage(hdb.centroids_, method='single')

hierarchyTreeCentroids = hierarchy.to_tree(z)

# Create hierarchy from centroids and pass to tree for further use. Possible to change parameters here
h = hierarchy.linkage(hdb.medoids_, method='single')

hierarchyTreeMedoids = hierarchy.to_tree(h)

# TODO: use whichever fits get for getting the coordinates
# https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html#module-scipy.cluster.hierarchy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_tree.html#scipy.cluster.hierarchy.to_tree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html#scipy.cluster.hierarchy.cut_tree
# this exists as well, maybe it will held

# TODO: implement Edge Quantile Cut

# TODO: implement Alpha Shape Cut

# Dendrogram for Christian DO NOT USE OTHERWISE
dendrogram = hierarchy.dendrogram(z)
plt.show()
