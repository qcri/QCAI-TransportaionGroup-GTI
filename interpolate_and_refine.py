import sys
from matplotlib import pyplot as plt
import os
import socket
import numpy as np
from time import time 
from bearing import calculate_initial_compass_bearing

def find_swapping_point(trip, i, j):
    # print(trip)
    x1 = list(map(lambda x: x[1], trip[i:j]))
    y1 = list(map(lambda x: x[0], trip[i:j]))
    A1 = np.vstack([x1, np.ones(len(x1))]).T
    function = np.linalg.lstsq(A1, y1, rcond=None)
    m1, c1 = function [0]
    residuals = function[1]
    new_bearing = calculate_initial_compass_bearing((trip[j - 1][0], trip[j - 1][1]), (trip[j][0], trip[j][1]))
    if new_bearing > 180: 
            new_bearing -= 360
    return new_bearing, residuals, x1, y1, m1, c1

def refinement(trip):
    new_points = []
    breaking_point = 0
    m1_c1s = []
    # old_bearing = calculate_initial_compass_bearing((trip[0][0], trip[0][1]), (trip[1][0], trip[1][1]))
    for i in range(2, len(trip)):
        if breaking_point == i + 1: 
            continue
        new_bearing, residuals, x1, y1, m1, c1 = find_swapping_point(trip, breaking_point, i)

        if i > breaking_point + 2 and residuals[0] > 1e-7:
        # if i > breaking_point + 2 and abs(old_bearing - new_bearing) > 10:
            m1_c1s.append((prev_m1, prev_c1, i))
            new_points += [(y1[0], x1[0])]
            new_points += [(m1 * xx + c1, xx) for xx in x1[1:-1]]
            new_points += [(y1[-1], x1[-1])]
            breaking_point = i
        prev_m1, prev_c1 = m1, c1
    new_points += [(y1[0], x1[0])]
    new_points += [(m1 * xx + c1, xx) for xx in x1[1:-1]]
    new_points += [(y1[-1], x1[-1])]
    # old_bearing = new_bearing
    return new_points

def interpolate(s, t, node_id_to_coords, node_coord_to_ids, prev_ts, ts, edges_to_time, other_time):
    """
    if interpolation not possible, break the trajectory.
    """

    s_id = node_coord_to_ids[','.join(
        ([str(float(_)) for _ in s.split(',')[1:3]]))]
    t_id = node_coord_to_ids[','.join(
        ([str(float(_)) for _ in t.split(',')[1:3]]))]

    # Socket connection to Go server
    port = int(sys.argv[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: 
        s.connect(('127.0.0.1', port))
        message = "%s,%s" % (s_id, t_id)
        s.send(str.encode(message))
        data = s.recv(16383).decode()

    # Parse response
    path_weight, path = data.split(':')

    if path_weight == '-1':
        return [(node_id_to_coords[s_id], prev_ts), (node_id_to_coords[t_id], ts)]

    
    path_node_coords = [node_id_to_coords[n] for  n in (path[1:-1].split(','))]
    timestamps = [float(prev_ts)]
    other_time = float(other_time)
    for ss, dd in zip(path_node_coords, path_node_coords[1:]):
        ts1 = (float(edges_to_time.get((ss, dd), other_time/len(path_node_coords))))
        # print(ts)
        timestamps.append(timestamps[-1] + float(ts1))
    
    timestamps.append(timestamps[-1] + edges_to_time.get((dd, t), other_time/len(path_node_coords)))

    return list(zip(path_node_coords, timestamps))


def get_bbox(sequence):
    max_lon = max([s[0] for s in sequence]) + 0.0002
    min_lon = min([s[0] for s in sequence]) - 0.0002
    max_lat = max([s[1] for s in sequence]) + 0.0002
    min_lat = min([s[1] for s in sequence]) - 0.0002
    return (min_lon, min_lat, max_lon, max_lat)


def in_bbox(pt, bbox):
    if pt[0] < bbox[0] or pt[0] > bbox[2] or pt[1] < bbox[1] or pt[1] > bbox[3]:
        return False
    return True


def generate_figure(sequence, points, dense_path):
    # Plot result
    bbox = get_bbox(sequence)
    relevant_points = np.array([pt for pt in points if in_bbox(pt, bbox)])
    plt.scatter(relevant_points[:, 0], relevant_points[:, 1], c='0.9')
    plt.scatter(np.array(dense_path)[:, 0],
                np.array(dense_path)[:, 1], c='red')
    plt.scatter(np.array(sequence)[:, 0], np.array(
        sequence)[:, 1], c='blue', s=50)

    plt.show()



    

if __name__ == '__main__':

    # centroids file
    nodes_fname = "data/output/centroids.txt"
    # raw edges file
    edges_fname = "data/output/edges.txt"

    #sparse data
    input_folder = "data/input/"

    #imputation path
    output_folder = "data/output/GTI"
    refined_folder = output_folder + "_refinement"
    
    #statistics file optional
    results_path = "data/output/stats.csv"

    try:
        os.mkdir(output_folder)
    except:
        pass

    try:
        os.mkdir(refined_folder)
    except:
        pass

    with open(results_path, 'w') as res : 
        res.write("trip_num,trip_name,sparse_points,dense_points,time_per_traj\n")
    
        
    node_coord_to_ids = {}
    node_id_to_coords = {}
    edges_to_time = {}
    dis = 0 
    files = list(os.listdir(input_folder))[:10000]
    disconnected_nodes = []

    with open(nodes_fname) as f:
        for line in f:
            cid, coords = line.strip().split(' ')
            node_coord_to_ids[coords] = cid
            node_id_to_coords[cid] = coords

    print("read nodes")
    with open(edges_fname) as f:
        for line in f:
            s, d, ts = line.strip().split(' ')
            edges_to_time[(s, d)] = ts
    
    print('there are %s trajectories' % len(files))

    

    for cnt, fil in enumerate(files):
        print("\r%s/%s" % (cnt, len(files)), end="")
        start_time = time()
        with open(os.path.join(input_folder, fil)) as f:
            # format: id, lat, lon, timestamp
            samples = f.read().strip().split('\n')
            new_samples = []
            refined_samples = []
            if len(samples) < 2:
                continue
            for i, (prev_sample, sample) in enumerate(list(zip(samples, samples[1:]))):
                if prev_sample == sample:
                    continue
                prev_lat, prev_lon = float(prev_sample.split(",")[1]), float(prev_sample.split(",")[2])

                lat, lon = float(sample.split(",")[1]), float(sample.split(",")[2])
                
                if i:
                    prev_ts = path[-1][1]
                    ts = float(sample.split(",")[3])
                else:
                    prev_ts, ts = float(prev_sample.split(",")[3]), float(sample.split(",")[3])
                other_time = (ts - prev_ts)
                try:
                    path = interpolate(prev_sample, sample,
                                node_id_to_coords,  node_coord_to_ids, prev_ts, ts, edges_to_time, other_time)
                except:
                    continue
                new_samples += path[:-1]

                path_to_be_refined = list(map(lambda x: (float(x[0].split(",")[0]), float(x[0].split(",")[1])), path))
                if len(path) > 2:
                    refined_samples += refinement(path_to_be_refined)
                else:
                    refined_samples += path_to_be_refined[:-1]
            # add the last element
            new_samples.append(path[-1])
            refined_samples.append(path_to_be_refined[-1])

            # add the last element
            try:
                new_samples.append(path[-1])
            except:
                continue

            refined_samples.append(path_to_be_refined[-1])
            if (prev_lat, prev_lon) in disconnected_nodes or (lat, lon) in disconnected_nodes:
                continue
            with open(os.path.join(output_folder, fil), 'w') as g:
                for i, sample in enumerate(new_samples):
                    g.write('%s,%s,%s,%s\n' %
                            (i, sample[0].split(',')[1], sample[0].split(',')[0], sample[1]))
            with open(os.path.join(refined_folder, fil), 'w') as g:
                for i, sample in enumerate(refined_samples):
                    g.write('%s,%s,%s,%s\n' %
                            (i, sample[1], sample[0], i))
            with open(results_path, 'a') as res : 
                res.write("%s,%s,%s,%s,%s\n" % (cnt, fil, len(samples), len(new_samples), time() - start_time))

        # with open(os.path.join(output_folder, fil), 'w') as g:
        #     for i, sample in enumerate(new_samples):
        #         g.write('%s,%s,%s,%s\n' %
        #                 (i, sample[0].split(',')[1], sample[0].split(',')[0], sample[1]))

        # with open(results_path, 'a') as res : 
        #     res.write("%s,%s,%s,%s,%s\n" % (cnt, fil, len(samples), len(new_samples), time() - start_time))
