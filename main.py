from typing import Any

import numpy as np
import copy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn import datasets
# TODO: add way to load different datasets
centers = [[1, 1], [-1, -1], [1, -1], [20, 20], [20, 21], [21, 20], [21, 21]]
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
# )
X, labels_true = datasets.load_digits(
    return_X_y=True
)
# run HDBSCAN on Data. Possible to change parameters here
#hdb = HDBSCAN(store_centers='both')
hdb = HDBSCAN(store_centers='both', min_cluster_size=10, min_samples=1)
hdb.fit(X)

data = [dict(index=index, label=hdb.labels_[index], coord=value) for index, value in enumerate(X)]

data.sort(key=lambda x: x['label'])

newData = copy.deepcopy(data)

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
    #uPoints = [data.map(lambda x: x['label']).filter(lambda x: x == z[i][0] or x == z[i][1])]
    uPoints = []
    
    cutlist.append((heightDiff, uPoints,  height))
#quantile = 0.95
#[0, 0, 1, 505, 10]
#[505, 10, 1, 0, 0]
print(cutlist)
# quantile must be up 1.0
def get_biggest_density_change(cutlist, quantile):
    cutlist.sort(key=lambda tup: tup[0])
    #cut from the point where quantile is reached
    cutlist = cutlist[:int(len(cutlist) * quantile)]
    return cutlist
list = get_biggest_density_change(cutlist, 0.55)
result = copy.deepcopy(newData)
for i in range(len(list)):
        points = []
        for element in newData:
            if (element['label'] == list[i][0]):
                result['label'] = list[i][1]

sorted(list, key=lambda tup: tup[2])
for i in range(len(list)):
    print(list[i][2])
#hierarchyTreeCentroids = hierarchy.to_tree(z)

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
# outPutlist = []
# for element in result:
#     plt.scatter(element['coord'][0], element['coord'][1], c=element['label'])

for a in result:
     print(a['coord'])

plt.scatter([newList['coord'][:] for newList in result], y=[newList['coord'][:] for newList in result], c=[newList['label'] for newList in result])
#dendrogram = hierarchy.dendrogram(z)
plt.show()
