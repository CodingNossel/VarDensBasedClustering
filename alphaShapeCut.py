import numpy as np
from scipy.spatial import Delaunay, ConvexHull

#input DBSCAN Dendogram G
def alpha_shape_cut(G, quantile):
    cutlist = []
    labels = []

    for (u, v) in G.edges():
        area_before = ConvexHull(G.nodes(u).points)
        area_after = ConvexHull(G.nodes(v).points)
        density_change = area_after - area_before

        cutlist.append((density_change, G.nodes(u).points, G.nodes(u).height))


    cutlist = get_biggest_density_change(cutlist, quantile)

#TODO wait for DBSCAN implementation
    for p in points:
        #set labels[p] to the index of the first cutlist elemnt it is part of
        for i in range(len(cutlist)):
            if p in cutlist[i][1]:
                labels[p] = i
                break
        else:
            labels[p] = -1


#TODO find out proper way to calculate density change
def get_biggest_density_change(cutlist, quantile):
    cutlist.sort(key=lambda tup: tup[0])
    cutlist = cutlist[int(len(cutlist) * quantile):]
    return cutlist








# to calculate area of a set of points -> convex hull of outer border of the Delaunay triangulation