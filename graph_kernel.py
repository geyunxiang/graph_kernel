import numpy as np
from numpy import linalg
from scipy.sparse.csgraph import shortest_path

def get_covariance_matrix(w, d):
	"""
	w is the adjacency matrix (symmetry and binary) of a graph 
	with shape N x N (type: np.array)
	d is the dimension of the covariance matrix
	"""
	n = w.shape[0] # number of nodes in the network
	if not w.any():
		# if this is a null graph, return a zero matrix
		return np.zeros((d, d))
	e = np.ones(n) # base vector with all elements = 1
	# prepare w^i * e and |w^i * e| up to i = n
	vector_list = [None] * n # vector_list[i] = w^(i+1) * e
	norm_list = [None] * n # norm_list[i] = |w^(i+1) * e|, l1 norm
	whole_matrix = np.zeros((n, d)) # each column is a target vector
	w_temp = w.copy()
	for i in range(d):
		vector_list[i] = np.matmul(w_temp, e)
		norm_list[i] = linalg.norm(vector_list[i], 1)
		whole_matrix[:, i] = n*vector_list[i]/norm_list[i]
		w_temp = np.matmul(w_temp, w_temp)
	# calculate covariance column-wise
	# set rowvar = False to avoid transpose the matrix
	# when rowvar = False, the covariance is calculated between each column vector
	return np.cov(whole_matrix, rowvar = False)

def calculate_covariance_similarity(c1, c2):
	"""
	Input two covariance matrix, return the similarity
	"""
	# print('c1:')
	# print(c1)
	# print('c2:')
	# print(c2)
	if np.max(c1 - c2) < 1e-5:
		# if c1 == c2
		return 1 # max similarity
	if not c1.any() or not c2.any():
		# if either c1 or c2 is zero
		return 0 # min similarity
	a = 0.5 * (c1 + c2)
	# print('a:')
	# print(a)
	d1 = linalg.det(c1)
	d2 = linalg.det(c2)
	da = linalg.det(a)
	# print('det c1: ', d1)
	# print('det c2: ', d2)
	# print('det a: ', da)
	if d1 < 1e-5 or d2 < 1e-5:
		# if either c1 or c2 is nearly zero
		return 0
	return np.exp(-0.5 * np.log(linalg.det(a)/np.sqrt(linalg.det(c1) * linalg.det(c2))))

def generate_subgraph(w, i, h):
	"""
	starting from node i, select all nodes in w (adjacency matrix) whose shortest path is
	less than or equal to h. Connecting node i and all selected nodes and return the matrix
	return
	"""
	n = w.shape[0] # no. of nodes
	dist_matrix, predecessors = shortest_path(w, directed = False, return_predecessors = True, unweighted = True, indices = i)
	selected_nodes = []
	for j in range(n):
		if j == i:
			continue
		if dist_matrix[j] <= h:
			selected_nodes.append(j)
	# re-construct sub_graph
	sub_graph = np.eye(n)
	for j in selected_nodes:
		k = j
		while predecessors[k] != i:
			sub_graph[k, predecessors[k]] = 1
			sub_graph[predecessors[k], k] = 1
			k = predecessors[k]
		sub_graph[k, i] = 1
		sub_graph[i, k] = 1
	# re-connect nodes
	for j in selected_nodes:
		for k in selected_nodes:
			if j == k:
				continue
			if w[j, k] == 1:
				sub_graph[j, k] = 1
				sub_graph[k, j] = 1
	return sub_graph

def compare_two_graphs(graph_1, graph_2, h, d):
	"""
	graph 1 and 2 are two adjacency matrices (binary, symmetry) of graph G and H. 
	The dimension (n) is the same
	parameter h controls the depth of sub-graphs
	parameter d controls the dimension of Krylov sub-space when calculating covariance matrices
	"""
	n = graph_1.shape[0]
	k = 0 # kernel
	for i in range(n):
		f = 0 # function related to each node
		for j in range(1, h+1):
			# print('node %d depth %d' % (i, j))
			sub_graph_1 = generate_subgraph(graph_1, i, j)
			sub_graph_2 = generate_subgraph(graph_2, i, j)
			c1 = get_covariance_matrix(sub_graph_1, d)
			c2 = get_covariance_matrix(sub_graph_2, d)
			sim = calculate_covariance_similarity(c1, c2)
			# print('sim: ', sim)
			f += sim
		f /= float(h)
		k += f
	k /= float(n)
	return k

def test_covariance_matrix():
	# should be right
	w = np.array([
		[1, 1, 0], 
		[1, 1, 0], 
		[0, 0, 1]])
	c = get_covariance_matrix(w, 2)
	print(c)

def test_sub_graph():
	w = np.array(
		[[1, 1, 1, 0, 0, 1], 
		 [1, 1, 1, 0, 0, 0],
		 [1, 1, 1, 1, 0, 0],
		 [0, 0, 1, 1, 0, 0],
		 [0, 0, 0, 0, 1, 1],
		 [1, 0, 0, 0, 1, 1]])
	s = generate_subgraph(w, 1, 2)
	print(s)

def test_main():
	w1 = np.array(
		[[1, 1, 0, 0, 1], 
		 [1, 1, 0, 0, 0],
		 [0, 0, 1, 1, 0],
		 [0, 0, 1, 1, 1],
		 [1, 0, 0, 1, 1]])
	w2 = np.array(
		[[1, 1, 1, 1, 1],
		 [1, 1, 1, 0, 0],
		 [1, 1, 1, 0, 0],
		 [1, 0, 0, 1, 1],
		 [1, 0, 0, 1, 1]])
	print('graph similarity: ', compare_two_graphs(w1, w2, 2, 2))

def test_all_zero():
	w1 = np.zeros((82, 82))
	w2 = np.zeros((82, 82))
	print('graph similarity: ', compare_two_graphs(w1, w2, 2, 2))

def test_real_data():
	from mmdps.proc import loader, atlas
	atlasobj = atlas.get('brodmann_lrce')
	subject_1 = loader.load_single_network(atlasobj, 'caochangsheng_20161027')
	subject_1 = subject_1.threshold(0.85)
	subject_1_data = (np.abs(subject_1.data) > 0).astype(int)

	subject_2 = loader.load_single_network(atlasobj, 'caochangsheng_20161114')
	subject_2 = subject_2.threshold(0.85)
	subject_2_data = (np.abs(subject_2.data) > 0).astype(int)

	# subject_2_data = subject_1_data.copy()
	# subject_2_data[0, 3] = 0
	# subject_2_data[3, 0] = 0
	print('graph similarity: ', compare_two_graphs(subject_1_data, subject_2_data, 4, 4))

def plot_real_range_threshold():
	"""
	This function loads in two networks of the same person on different date,
	construct binary network
	"""
	from mmdps.proc import loader, atlas
	atlasobj = atlas.get('brodmann_lrce')
	subject_1 = loader.load_single_network(atlasobj, 'caochangsheng_20161027')
	subject_2 = loader.load_single_network(atlasobj, 'caochangsheng_20161114')
	subject_3 = loader.load_single_network(atlasobj, 'chenyifan_20150612')
	subject_4 = loader.load_single_network(atlasobj, 'chenyifan_20150629')
	sim_list_1 = []
	sim_list_2 = []
	sim_list_3 = []
	for threshold in range(100):
		threshold = threshold/100.0
		print(threshold)
		# sim_list_1.append(0.8)
		sim_list_1.append(compare_two_graphs(subject_1.binarize(threshold).data, subject_2.binarize(threshold).data, 4, 4))
		sim_list_2.append(compare_two_graphs(subject_3.binarize(threshold).data, subject_4.binarize(threshold).data, 4, 4))
		sim_list_3.append(compare_two_graphs(subject_1.binarize(threshold).data, subject_3.binarize(threshold).data, 4, 4))
	from matplotlib import pyplot as plt
	plt.figure()

	plt.plot(range(100), sim_list_1, label = 'cao1 vs cao2')
	plt.plot(range(100), sim_list_2, label = 'chen1 vs chen2')
	plt.plot(range(100), sim_list_3, label = 'cao1 vs chen1')
	plt.xticks(range(0, 101, 10), range(0, 101, 10))
	plt.grid(True)
	plt.legend()
	plt.title('Network similarity h = 4, d = 4')
	plt.xlabel('Threshold * 100')
	plt.ylabel('Similarity')
	plt.savefig('/Users/andy/Downloads/figure 4.png', dpi = 300)
	plt.close()

def plot_real_healthy_range_threshold():
	"""
	This function loads in the healthy networks, calculate graph similarities, and plot them in one graph.
	"""
	from mmdps.proc import group_manager, loader, atlas
	from mmdps.util import stats_utils
	atlasobj = atlas.get('brodmann_lrce')
	healthy_group = group_manager.getHealthyGroup()
	healthy_scans = [scan.filename for scan in healthy_group.scans]
	healthy_nets = [loader.load_single_network(atlasobj, filename) for filename in healthy_scans]
	thresh_list = list(range(60, 70))
	upper_bound_list = [None] * len(thresh_list)
	mean_list = [None] * len(thresh_list)
	lower_bound_list = [None] * len(thresh_list)
	for count, threshold in enumerate(thresh_list):
		print('Threshold: %d' % threshold)
		threshold = threshold/100.0
		tmp_sim_list = []
		for i in range(len(healthy_nets)):
			for j in range(i+1, len(healthy_nets)):
				# calculate similarities between each pair of networks
				tmp_sim_list.append(compare_two_graphs(healthy_nets[i].binarize(threshold).data, healthy_nets[j].binarize(threshold).data, 4, 4))
		m, l, u = stats_utils.mean_confidence_interval(tmp_sim_list)
		mean_list[count] = m
		upper_bound_list[count] = u
		lower_bound_list[count] = l
	from matplotlib import pyplot as plt
	plt.plot(thresh_list, mean_list, '.-g', label = 'mean')
	plt.plot(thresh_list, lower_bound_list, '.-c', label = 'lower bound')
	plt.plot(thresh_list, upper_bound_list, '.-c', label = 'upper bound')
	plt.fill_between(thresh_list, lower_bound_list, upper_bound_list, alpha = 0.25, color = 'g')
	
	plt.xticks(range(60, 71, 1), range(60, 71, 1))
	plt.grid(True)
	plt.legend()
	plt.title('Healthy network similarity h = 4, d = 4')
	plt.xlabel('Threshold * 100')
	plt.ylabel('Similarity')
	plt.savefig('/Users/andy/Downloads/figure 5.png', dpi = 300)
	plt.close()

if __name__ == '__main__':
	plot_real_healthy_range_threshold()
