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

# run HDBSCAN on Data. Possible to change parameters here
# hdb = HDBSCAN(store_centers='both')
hdb = HDBSCAN(store_centers='both', min_cluster_size=10, min_samples=1)
hdb.fit(X)
# print iris for the test
print(hdb.labels_)

print("------------------")
data = [dict(index=index, label=hdb.labels_[index], coord=value) for index, value in enumerate(X)]

data.sort(key=lambda x: x['label'])

newData = copy.deepcopy(data)

# Create hierarchy from centroids and pass to tree for further use. Possible to change parameters here
z = hierarchy.linkage(hdb.centroids_, method='single')

# linkage to IDFKIKMS
# get height of parent and child cluster
# TODO: figure out how to get points for cluster (z[i][0] and z[i][1])
# print(z)
cutlist = []
for i in range(len(z) - 1):
    height = z[i][2]
    heightChild = z[i + 1][2]
    heightDiff = height - heightChild

    cutlist.append((heightDiff, height))


# quantile = 0.95
# [0, 0, 1, 505, 10]
# [505, 10, 1, 0, 0]
# print(cutlist)
# quantile must be up 1.0
def get_biggest_density_change(cutlist, quantile):
    cutlist.sort(key=lambda tup: tup[0])
    # cut from the point where quantile is reached
    cutlist = cutlist[:int(len(cutlist) * quantile)]
    return cutlist


list2 = get_biggest_density_change(cutlist, 0.55)
result = copy.deepcopy(newData)

# debug why label assigment is wrong

# for element in result:
#     changeTo = 0
#     comp = 0
#     changeOnlyMe = 0
#     val = 0
#     for i in range(len(list2)):
#         changeTo = z[i][1]
#         changeOnlyMe = z[i][0]
#         val = list2[i][1]
#         print('changeTo: ', changeTo, ' changeOnlyMe: ', changeOnlyMe)
#         for j in range(len(z)):
#             comp = z[j][2]
#             print('compare: ', comp)
#             for k in range(len(z)):
#                 if z[k][0] == changeOnlyMe:
#                     if val == comp:
#                         cluster = z[k][0]
#                         print('changed from: ', element['label'], '->', 'to: ', cluster, )
#                         element['label'] = cluster

changeTo = 0
changeOnlyMe = 0
valIdentifier = 0
compareTo = 0
tmp = []

for i in range(len(list2)):
    valIdentifier = list2[i][1]

    for j in range(len(z)):
        compareTo = z[j][2]

        if valIdentifier == compareTo:
            changeTo = z[j][0]
            changeOnlyMe = z[j][1]
            tmp.append((changeOnlyMe, changeTo))
            print('changeTo: ', changeTo, 'changeOnlyMe: ', changeOnlyMe, 'valIdentifier: ', valIdentifier,
                  'compareTo: ', compareTo)
            break

for k in range(len(tmp)):
    for element in result:
        if element['label'] == tmp[k][0]:
            element['label'] = tmp[k][1]
            print('change from: ', tmp[k][0], '->', 'changed to: ', tmp[k][1], 'point: ', element['coord'])

sorted(list2, key=lambda tup: tup[1])
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

# Dendrogram for Christian DO NOT USE OTHERWISE
# outPutlist = []
# for element in result:
#     plt.scatter(element['coord'][0], element['coord'][1], c=element['label'])
# count = 0
# for a in result:
#      print(a['label'], "coord:",  a['coord'], "index:", a['index'], "count:", count)
#      count += 1
#      if (count == 10):
#           break

# print("------------------")
# count = 0
# for i in data:
#      print(i['label'], "coord:",  i['coord'], " index:", i['index'], "count:", count)
#      count += 1
#      if (count == 10):
#           break

for i in range(len(data)):
    print('data: ', data[i], 'compare_result: ', data[i]['label'] == result[i]['label'])

# plt.scatter([newList['coord'][:] for newList in result], y=[newList['coord'][:] for newList in result], c=[newList['label'] for newList in result])
# dendrogram = hierarchy.dendrogram(z)
# plt.show()
