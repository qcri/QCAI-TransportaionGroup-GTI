import numpy as np
from scipy.spatial import cKDTree
from utils import haversine
import time
import sys
from collections import defaultdict
import time
from bearing import calculate_initial_compass_bearing
from math import sqrt, radians, sin, cos, atan2, degrees
from os import listdir,  mkdir
from datetime import datetime

start_time = time.time()

sparse_folder = "data/chicago_250/"
output_folder = "data/output/"
filenames = listdir(sparse_folder)

# Distance threshold
RADIUS_METER = 50
RADIUS_DEGREE = RADIUS_METER * 10e-6

# Angle penalty
THETA = 50


G = dict()
counter = 0



try:
    mkdir(output_folder)
except:
    pass

edges_go_path = output_folder + "edges_go.txt"
edges_path = output_folder + "edges.txt"
centroids_path = output_folder + "centroids.txt"


points = []
points_speed = {}
coord2bearing = {}
ts = []
speed = []
angle = []
pointsConsid = 0
pointsNot = 0

print("Started building trajectories!")

#uncomment this if dataset is in doha, week in doha starts on sunday
# weekdays = [6, 0, 1, 2, 3]
weekdays = [0, 1, 2, 3, 4]

for ii in range(len(filenames)):
    print("\r%s/%s" % (ii, len(filenames)), end="")
    filename = filenames[ii]
    with open(sparse_folder + filename, 'r') as f:
        data = []
        for i, line in enumerate(f):
            # format: id, lat, lng, timestamp
            id_, lat, lng, ts = line.strip().split(',')
            lat, lng, ts = float(lat), float(lng), float(ts)
            points.append((lat, lng))
            if i:
                # compute angle
                a = calculate_initial_compass_bearing((data[-1][1],
                    data[-1][0]), (lat, lng))
                data.append((lng, lat, ts, a))
            else:
                data.append((lng, lat, ts, 0))

        if len(data)>1:
            #shift every angle 1 back 
            for i in range(1, len(data)):
                data[i-1] = (data[i-1][0], data[i-1][1], data[i-1][2], data[i][3])

                point_time = abs(data[i-1][2] - data[i][2])
                point_dist = haversine((data[i-1][0],data[i-1][1]), (data[i][0],data[i][1]))

                ts = data[i-1][2]
                ts_object = datetime.fromtimestamp(ts)
                weekday = ts_object.weekday()
                hour = ts_object.hour
                # get timeslot interval
                if weekday in weekdays: 
                    interval = hour // 2
                else:
                    interval = 12 + hour // 2 

                if point_time > 0:
                    sp = point_dist/point_time
                    if sp > 30:
                        sp = 30 
                    if sp < 3:
                        sp = 3
                    points_speed[(data[i-1][0], data[i-1][1])] = (sp, interval)
                else:
                    points_speed[(data[i-1][0], data[i-1][1])] = (0, interval)
                    
            points_speed[(data[-1][0], data[-1][1])] = (points_speed[(data[-2][0], data[-2][1])])
            
            for i in range(1, len(data)-1):
                data[i] = (data[i][0], data[i][1], data[i][2], (data[i-1][3]+data[i+1][3])/2)

            data[-1] = (data[-1][0], data[-1][1], data[-1][2], data[-2][3])
        
        for lat, lng, ts, ang in data:
            try:
                coord2bearing[str(lat) + "," + str(lng)] = ang
                pointsConsid += 1
            except:
                pointsNot += 1


points = np.array(list(set(points)))
centroids = points
cidx = cKDTree(list(centroids))
print('\n\ncomputing nns for all clusters')

c_nns = cidx.query_ball_point(x=list(centroids), r=RADIUS_DEGREE, p=2)

for c in centroids:
    G[tuple(c)] = []

exceptionsoccured = 0
accepted = 0
print('\n\nConnect each cluster to the closest x clusters within D  meters')

for i, c in enumerate(centroids):
    print("\r%s/%s" % (i, len(centroids)), end="")
    # find neighbors and their respective distance
    nns_dist = np.array([haversine(c, centroids[j]) for j in c_nns[i]])

    # if there are no neighbors except for yourself
    if len(nns_dist) < 2: 
        # increase the radius to the closest neighbor
        dd, ii = cidx.query(centroids[i], k=[2])
        dd, ii = dd[0], ii[0]

        try:
            temp = centroids[ii]
            longitudecon = temp[1]
            latitudecon = temp[0]
            query = str(longitudecon) + "," + str(latitudecon)
        except Exception as e:
            exceptionsoccured += 1
            continue
        try:
            speed, interval = points_speed[longitudecon, latitudecon]
        except:
            continue
        
        dist = dd * 10e4

        if speed < 3:
            speed = 3
        time_to_go = dist/speed

        # store the distance and time it takes to go to the neighbor
        G[tuple(c)].append((tuple(centroids[ii]), (dd * 10e4, time_to_go)))
        
    # if there are available neighbors
    else:
        mins = nns_dist.argsort()
        average_speed = 0 
        longitudeiter = c[1]
        latitudeiter = c[0]
        
        try:
            speediter, intervaliter = points_speed[longitudeiter, latitudeiter]
        except:
            continue

        neighbors_with_same_interval = 0 

        # each of them is a cadidate, find the ones at the same time inerval
        for n in mins: 
            temp = centroids[c_nns[i][n]]
            longitudecon = temp[1]
            latitudecon = temp[0]
            
            try:
                speedcon, intervalcon = points_speed[longitudecon, latitudecon]
            except:
                continue
            
            if intervalcon == intervaliter:
                average_speed += speedcon
                neighbors_with_same_interval += 1
        
        if neighbors_with_same_interval > 0:
            average_speed /= neighbors_with_same_interval
        else:
            average_speed = speediter

        # proceed to filtering them
        for n in mins:
            longitudeiter = c[1]
            latitudeiter = c[0]
            query = str(longitudeiter) + "," + str(latitudeiter)
            #bearing of current point
            try:
                bearingofc = coord2bearing[query]
            except:
                exceptionsoccured += 1
                break
            #bearing of neighbor
            try:
                temp = centroids[c_nns[i][n]]
                longitudecon = temp[1]
                latitudecon = temp[0]
                query = str(longitudecon) + "," + str(latitudecon)
                bearingofcon = coord2bearing[query]

            except Exception as e:
                exceptionsoccured += 1
                continue

            dist = nns_dist[n]
            
            diff = abs(bearingofc - bearingofcon)
            absol = diff
            diff = min(diff, 360-diff)

            # apply the angle penalty
            distance_metric = sqrt(dist ** 2 + (THETA * diff /180) ** 2)


            # find the average speed
            if average_speed < 3:
                average_speed = 3
            if dist < average_speed:
                time_to_go = 1
            else:
                time_to_go = dist/average_speed
            
            if distance_metric <= RADIUS_METER and distance_metric >= 20 and (longitudeiter != longitudecon and latitudeiter != latitudecon):
                G[tuple(c)].append((tuple(centroids[c_nns[i][n]]), (dist,time_to_go)))

clusters_ids = {}
with open(centroids_path, 'w') as g:
    for i, c in enumerate(centroids):
        clusters_ids["%s,%s" % (c[0], c[1])] = str(i)
        g.write('%s %s,%s\n' % (i, c[0], c[1]))
edge_list = defaultdict(set)

with open(edges_path, 'w') as g:
    for s, edges in G.items():
        for t, (d, time_to_go) in edges:
            edge_list[clusters_ids["%s,%s" % (s[0], s[1])]].add((clusters_ids["%s,%s" % (t[0], t[1])], int(d + 1)))
            edge_list[clusters_ids["%s,%s" % (t[0], t[1])]].add((clusters_ids["%s,%s" % (s[0], s[1])], int(d + 1)))
            g.write('%s,%s %s,%s %s\n' % (s[0], s[1], t[0], t[1], time_to_go))



covered_vertices = []
for l in edge_list.values():
    covered_vertices += l

missing_vertices = list(
    set(clusters_ids.values()).difference(set(edge_list.keys())))

with open(edges_go_path, 'w') as g:
    for v, neis in edge_list.items():
        g.write('%s %s\n' % (v, ' '.join(["%s,%s" % i for i in neis])))

    g.write('\n'.join(missing_vertices))

print("\n\n--- GTI: %s seconds ---" % (time.time() - start_time))
