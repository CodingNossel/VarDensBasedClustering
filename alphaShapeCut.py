import copy
import numpy as np
from scipy.spatial import Delaunay, ConvexHull 
from shapely.geometry import Polygon
import alphashape


#input: linkage matrix z after HDBScan, data, quantile
def alpha_shape_cut(z, data, quantile):
    cutlist = []
    for i in range(len(z) - 1):
        uPoints = [element for element in data if element['label'] == list[i][0] for i in range(len(list))]
        vPoints = [element for element in data if element['label'] == list[i + 1][0] for i in range(len(list))]
        alphaU = alphashape.alphashape(uPoints, quantile)
        alphaV = alphashape.alphashape(vPoints, quantile)
        areaU = concave_polygon_area(alphaU)
        areaV = concave_polygon_area(alphaV)
        areaDiff = areaU - areaV
        #TODO GET POINTS OF U
        #uPoints = [data.map(lambda x: x['label']).filter(lambda x: x == z[i][0] or x == z[i][1])]
        uPoints = []
    
        cutlist.append((areaDiff, uPoints,  areaU))
    list = get_biggest_density_change(cutlist, 0.55)
    result = copy.deepcopy(data)
    for i in range(len(list)):
        for element in data:
            if (element['label'] == list[i][0]):
                result['label'] = list[i][1]
    return result
#quantile = 0.95
#[0, 0, 1, 505, 10]
#[505, 10, 1, 0, 0]
# quantile must be up 1.0
def get_biggest_density_change(cutlist, quantile):
    cutlist.sort(key=lambda tup: tup[0])
    #cut from the point where quantile is reached
    cutlist = cutlist[:int(len(cutlist) * quantile)]
    return cutlist

def concave_polygon_area(coords):
    polygon = Polygon(coords)
    if polygon.is_convex:
        return polygon.area
    else:
        convex_parts = list(polygon.convex_hull)
        return sum(Polygon(part).area for part in convex_parts)