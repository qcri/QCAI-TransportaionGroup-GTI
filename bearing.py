#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
from matplotlib import pyplot as plt
import numpy as np

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def calculate_initial_compass_bearing(pointA, pointB):
   
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def next_point(lat, lon, dist, bearing):
    R = 6378.1 #Radius of the Earth
    brng = math.radians(bearing) #Bearing is 90 degrees converted to radians.
    d = float(dist)/1000 #Distance in km

    #lat2  52.20444 - the lat result I'm hoping for
    #lon2  0.36056 - the long result I'm hoping for.

    lat1 = math.radians(lat) #Current lat point converted to radians
    lon1 = math.radians(lon) #Current long point converted to radians

    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
         math.cos(lat1)*math.sin(d/R)*math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
                 math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return(lon2, lat2)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

def get_bbox(sequence):
    max_lon = max([s[0] for s in sequence]) + 0.001
    min_lon = min([s[0] for s in sequence]) - 0.001
    max_lat = max([s[1] for s in sequence]) + 0.001
    min_lat = min([s[1] for s in sequence]) - 0.001
    return (min_lon, min_lat, max_lon, max_lat)

def in_bbox(pt, bbox):
    if pt[0] < bbox[0] or pt[0] > bbox[2] or pt[1] < bbox[1] or pt[1] > bbox[3]:
        return False
    return True


def generate_figure(i, sequence, points, dense_path, title=None):
    # Plot result
    plt.figure()
    bbox = get_bbox(dense_path)
    relevant_points = np.array([pt for pt in points if in_bbox(pt, bbox)])
    plt.scatter(relevant_points[:, 0], relevant_points[:, 1], c='0.9')
    plt.scatter(np.array(dense_path)[:, 0], np.array(dense_path)[:, 1], c='red')
    plt.scatter(np.array(sequence)[:, 0], np.array(sequence)[:, 1], c='blue', s=50)
    plt.scatter(np.array(dense_path[-2:-1])[:, 0],
            np.array(dense_path[-2:-1])[:, 1], c='green')
    if title:
        plt.title(title)
    plt.savefig('figs/%s.png'% str(i).zfill(4), type='PNG')

def angledist(a1, a2):
    return(min(abs(a1-a2),abs((a1-a2) % 360),abs((a2-a1) % 360),abs(a2-a1)))