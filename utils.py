from math import radians, cos, sin, asin, sqrt
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from collections import defaultdict
import multiprocessing
from geopy.distance import distance

def haversine(pt1, pt2):
	"""
	Calculate the great circle distance between two points
	on the earth (specified in decimal degrees)
	Sofiane: got it from:http://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
	:param pt1: point (lon, lat)
	:param pt2: point (lon, lat)
	:return: the distance in meters
	"""

	# Somewhat skrewed up this part about the order of lon, lat. Needs a check
	lon1 = pt1[1]
	lat1 = pt1[0]
	lon2 = pt2[1]
	lat2 = pt2[0]

	# convert decimal degrees to radians
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

	# haversine formula
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
	c = 2 * asin(sqrt(a))
	km = 6367 * c
	return km * 1000


def find_medoid(c):
	centroid = np.mean(c, axis=0)
	medoid = min(c, key=lambda point: haversine(point, centroid))
	return medoid

def compute_new_centroids(clusters):
	"""
	find new cluster medoids
	:param clusters: defaultdict of cluster:points
	:return: new defaultdict of the same format
	"""
	pass


def run_dbscan(points, max_dist=100):
	print ('Running DBSCAN')
	db = DBSCAN(eps=float(max_dist) / 1000 / 6371., min_samples=1, algorithm='ball_tree', metric='haversine').fit(
		np.radians(points))
	#
	# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	# core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print ('Estimated number of clusters: %d' % n_clusters_)
	clusters = [points[labels == i] for i in xrange(n_clusters_)]
	return clusters


def clustering(points, RADIUS_METER=100):
	print ('Start Clustering')
	idx = cKDTree(list(points))
	RADIUS_DEGREE = RADIUS_METER * 10e-6
	clusters = defaultdict(list)
	clustered = np.zeros(len(points))
	for i, point in enumerate(points):
		if clustered[i] == 1:
			continue
		nn_idx = idx.query_ball_point(x=point, r=RADIUS_DEGREE, p=2)
		clusters[tuple(point)] = points[nn_idx]
		clustered[nn_idx] = 1
	print ('Computing new medoids of the clusters:', len(clusters))
	#clusters = {tuple(find_medoid(points)): points for k, points in clusters.iteritems() if len(points) > 1}
	clusters = {tuple(find_medoid(points)): points for k, points in clusters.iteritems()}

	print ('Re-assigning closest points to clusters')
	new_assignment = dict()
	clustered = np.zeros(len(points))
	for i, medoid in enumerate(clusters.keys()):
		nn_idx = idx.query_ball_point(x=medoid, r=RADIUS_DEGREE, p=2)
		rel_nn_idx = [_ for _ in nn_idx if clustered[_] == 0]
		new_assignment[medoid] = points[rel_nn_idx]
		clustered[nn_idx] = 1
	#clusters = {tuple(find_medoid(points)): points for k, points in new_assignment.iteritems() if len(points) > 1}
	clusters = {}
	for k, points in new_assignment.iteritems():
		if len(points) < 2:
			clusters[k] = points
		else:
			clusters[tuple(find_medoid(points))] = points
	# centroids = np.array(clusters.keys())
	return clusters

def link_points(g, cluster, RADIUS_METER, max_links=4):
	RADIUS_DEGREE = RADIUS_METER * 10e-6
	cidx = cKDTree(cluster)
	# print 'computing nns for all clusters'
	c_nns = cidx.query_ball_point(x=list(cluster), r=RADIUS_DEGREE * 10, p=2)
	c_nns_dist = [[haversine(c, cluster[j]) for j in c_nns[i]] for i, c in enumerate(cluster)]

	g.add_nodes_from(map(tuple, cluster))
	# print 'Connect each cluster to the closest two clusters within 200 meters'
	for i, c in enumerate(cluster):
		nns_dist = np.array([haversine(c, cluster[j]) for j in c_nns[i]])
		mins = nns_dist.argsort()[1:max_links]
		for n in mins:
			if nns_dist[n] <= 200 and g.degree(tuple(c)) < max_links and d.degree(tuple(cluster[c_nns[i][n]])):
				g.add_edge(tuple(c), tuple(cluster[c_nns[i][n]]))

def create_edges(point):
	global points
	global cidx
	global RADIUS_DEGREE
	global max_links
	nns = cidx.query_ball_point(x=point, r=RADIUS_DEGREE, p=2)
	nns_dist = np.array([haversine(point, points[j]) for j in nns])
	mins = nns_dist.argsort()[1:max_links+1]
	es = []
	for n in mins:
		if nns_dist[n] <= 50:
			es.append((tuple(point), tuple(points[nns[n]])))
	return es

def link_points_edges(cluster, RADIUS_METER, max_links=4):
	node_degree = defaultdict()
	edges = []
	RADIUS_DEGREE = RADIUS_METER * 10e-6
	cidx = cKDTree(cluster)
	# print 'computing nns for all clusters'
	print ('creating index')
	c_nns = []
	for i in range(1+len(cluster)/10000):
		print (i*10000, (i+1)*10000)
		a = list(cidx.query_ball_point(x=list(cluster[i*10000: (i+1)*10000]), r=RADIUS_DEGREE, p=2))
		c_nns += a
	# c_nns = cidx.query_ball_point(x=list(cluster), r=RADIUS_DEGREE * 10, p=2)
	print( 'index created. Computing distances')
	# c_nns_dist = [[haversine(c, cluster[j]) for j in c_nns[i]] for i, c in enumerate(cluster)]
	# print 'distances computed'
	# node_degree = {x: 0 for x in map(tuple, cluster)}
	# print 'Connect each cluster to the closest two clusters within 200 meters'

	print( 'Parallel computing of edges')
	pool = multiprocessing.Pool()
	edges = pool.map(create_edges, cluster)
	# for i, c in enumerate(cluster):
	# 	print 'Point:', i, '/', len(cluster)
	# 	nns_dist = np.array([haversine(c, cluster[j]) for j in c_nns[i]])
	# 	mins = nns_dist.argsort()[1:max_links+1]
	# 	for n in mins:
	# 		# if nns_dist[n] <= 200 and node_degree[tuple(c)] < max_links and node_degree[tuple(cluster[c_nns[i][n]])] < max_links:
	# 		if nns_dist[n] <= 50:
	# 			edges.append((tuple(c), tuple(cluster[c_nns[i][n]])))
	# 			# node_degree[tuple(c)] += 1
	# 			# node_degree[tuple(cluster[c_nns[i][n]])] += 1
	return [e[i] for e in edges for i in range(len(e))]


def draw_graph(G):
	from matplotlib import collections as mc, pyplot as plt
	lines = [[s, t] for s, t in G.edges()]
	lc = mc.LineCollection(lines)
	fig, ax = plt.subplots()
	ax.add_collection(lc)
	centroids = np.array(G.nodes())
	ax.scatter(centroids[:, 0], centroids[:, 1])
	plt.show()


def densify(e, densification_rate=10):
	"""
	Densify an edge e(s,d) into a sequence of edges
	:param e: edge(s, d)
	:return: list of edges
	"""
	s = e[0]
	d = e[1]
	dist = distance(s, d).meters
	nb_points_frac = float(dist / densification_rate)
	nb_points = 0
	if (int(nb_points_frac) ==  nb_points_frac):
		# for 200/50 = 4.0 we need to generate 3 points
		nb_points = int(nb_points_frac) - 1
	elif (int(nb_points_frac) < nb_points_frac):
		# for 170/50 = ~3.4 we need to generate 3 points
		nb_points = int(nb_points_frac)
	else:
		# for 220/50 = ~4.4 we need 4 points
		nb_points = int(nb_points_frac) + 1

	x_delta = float(s[0] - d[0])/(nb_points+1)
	y_delta = float(s[1] - d[1])/(nb_points+1)
	points = []
	edges = []
	pv_pt = s
	for i in range(1, nb_points+1):
		cur_pt = (s[0]-x_delta*i, s[1]-y_delta*i)
		points.append(cur_pt)
		edges.append((pv_pt, cur_pt))
		pv_pt = cur_pt
	edges.append((pv_pt, d))
	return edges
