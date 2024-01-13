from typing import Any

import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn import datasets
# TODO: add way to load different datasets
centers = [[1, 1], [-1, -1], [1, -1], [20, 20], [20, 21], [21, 20], [21, 21]]
supa = datasets.load_iris()
print(len(supa.data))
print(len(supa['target']))
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

# run HDBSCAN on Data. Possible to change parameters here
hdb = HDBSCAN(store_centers='both')
hdb.fit(supa.data)

data = [dict(index=index, label=hdb.labels_[index], coord=value) for index, value in enumerate(X)]

data.sort(key=lambda x: x['label'])

# Create hierarchy from centroids and pass to tree for further use. Possible to change parameters here
z = hierarchy.linkage(hdb.centroids_, method='single')

#linkage to IDFKIKMS
# get height of parent and child cluster
#TODO: figure out how to get points for cluster (z[i][0] and z[i][1])
print(z)
cutlist = []
for i in range(len(z) - 1):
    height = z[i][2]
    heightChild = z[i+1][2]
    heightDiff = height - heightChild
    #TODO GET POINTS OF U
    uPoints = []
    
    cutlist.append((heightDiff, uPoints,  height))
#quantile = 0.95
#[0, 0, 1, 505, 10]
print(cutlist)
# quantile must be up 1.0
def get_biggest_density_change(cutlist, quantile):
    cutlist.sort(key=lambda tup: tup[0])
    cutlist = cutlist[int(len(cutlist) * quantile):]
    return cutlist
list = get_biggest_density_change(cutlist, 0.95)
for i in range(len(list)):
    #TODO: set label to the first I dunno pls help
    #FIX PIPELINE
    print(list[i][0])

hierarchyTreeCentroids = hierarchy.to_tree(z)

# Create hierarchy from centroids and pass to tree for further use. Possible to change parameters here
#h = hierarchy.linkage(hdb.medoids_, method='single')

#hierarchyTreeMedoids = hierarchy.to_tree(h)

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