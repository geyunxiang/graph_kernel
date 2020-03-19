from graph_kernel import compare_two_graphs
from matplotlib import pyplot as plt
from mmdps.proc import group_manager, loader, atlas, analysis_report
from mmdps.util import stats_utils
import numpy as np
import mongodb_database

thresh_list = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
WORK_PREFIX = 'C:/Users/geyx/Desktop/workshop/graph_kernel/'

def calculate_real_range_parameters():
	mdb = mongodb_database.MongoDBDatabase(None)
	atlasobj = atlas.get('brodmann_lrce')
	subject_1 = loader.load_single_network(atlasobj, 'caochangsheng_20161027')
	subject_2 = loader.load_single_network(atlasobj, 'caochangsheng_20161114')
	subject_3 = loader.load_single_network(atlasobj, 'chenyifan_20150612')
	subject_4 = loader.load_single_network(atlasobj, 'chenyifan_20150629')

	threshold_list = range(40, 60, 5)
	h_list = list(range(1, int(atlasobj.count/2)))
	d_list = list(range(2, 10))
	for thresh_count, threshold in enumerate(threshold_list, 1):
		threshold = threshold/100.0
		sim_list_1 = np.zeros(len(h_list))
		sim_list_2 = np.zeros(len(h_list))
		sim_list_3 = np.zeros(len(h_list))
		for idx, h in enumerate(h_list):
			print('threshold: %d/%d, depth: %d/%d' % (thresh_count, len(threshold_list), h, h_list[-1]))
			sim_list_1[idx] = compare_two_graphs(subject_1.binarize(threshold).data, subject_2.binarize(threshold).data, h, 4)
			sim_list_2[idx] = compare_two_graphs(subject_3.binarize(threshold).data, subject_4.binarize(threshold).data, h, 4)
			sim_list_3[idx] = compare_two_graphs(subject_1.binarize(threshold).data, subject_3.binarize(threshold).data, h, 4)
		threshold = int(threshold * 100.0)
		mdb.put_temp_data(sim_list_1, 'gk cao range h threshold %d' % threshold, 'caochangsheng_20161027 vs caochangsheng_20161114 network similarity measured by graph kernel with brodmann_lrce, threshold = %d, d = 4, h range(1, int(atlasobj.count/2))' % threshold)
		mdb.put_temp_data(sim_list_2, 'gk chen range h threshold %d' % threshold, 'chenyifan_20150612 vs chenyifan_20150629 network similarity measured by graph kernel with brodmann_lrce, threshold = %d, d = 4, h range(1, int(atlasobj.count/2))' % threshold)
		mdb.put_temp_data(sim_list_3, 'gk cao vs chen range h threshold %d' % threshold, 'caochangsheng_20161027 vs chenyifan_20150612 network similarity measured by graph kernel with brodmann_lrce, threshold = %d, d = 4, h range(1, int(atlasobj.count/2))' % threshold)

def calculate_real_healthy_range_threshold():
	"""
	This function loads in the healthy networks, calculate graph similarities, and plot them in one graph.
	"""
	atlasobj = atlas.get('brodmann_lrce')
	healthy_group = group_manager.getHealthyGroup()
	healthy_scans = [scan.filename for scan in healthy_group.scans]
	healthy_nets = [loader.load_single_network(atlasobj, filename) for filename in healthy_scans]
	thresh_list = list(range(60, 100))
	curve_mat = np.zeros((
		int(len(healthy_nets)*(len(healthy_nets)-1)/2), 
		len(thresh_list))) # each row is a curve
	for count, threshold in enumerate(thresh_list):
		print('Threshold: %d' % threshold)
		threshold = threshold/100.0
		counter = 0 # counter in curve_mat row index
		for i in range(len(healthy_nets)):
			for j in range(i+1, len(healthy_nets)):
				# calculate similarities between each pair of networks
				curve_mat[counter, count] = compare_two_graphs(healthy_nets[i].binarize(threshold).data, healthy_nets[j].binarize(threshold).data, 4, 4)
				counter += 1
	mdb = mongodb_database.MongoDBDatabase(None)
	mdb.put_temp_data(curve_mat, 'graph kernel inter-healthy', 'The inter-healthy network similarity calculated by graph kernel method with d = 4, h = 4. The binary threshold is varied range(60, 100). The data is of shape (cases, thresh_list length). Each row is a curve')

def calculate_SCI_vs_healthy():
	"""
	This function loads in healthy networks and SCI patients networks, calculate each
	patients network similarity to each healthy controls, and plot each person's similarity
	in one figure.
	"""
	atlasobj = atlas.get('brodmann_lrce')
	healthy_group = group_manager.getHealthyGroup()
	healthy_scans = [scan.filename for scan in healthy_group.scans]
	healthy_nets = [loader.load_single_network(atlasobj, filename) for filename in healthy_scans]
	
	assistant = analysis_report.GroupAnalysisAssistant('jisuizhenjiaciji', atlasobj)
	study = assistant.study
	group1 = study.getGroup('control 1')
	scans = [scan.filename for scan in group1.scans]
	group2 = study.getGroup('treatment 1')
	scans += [scan.filename for scan in group2.scans]
	SCI_nets = [loader.load_single_network(atlasobj, scan) for scan in scans]

	curve_mat = np.zeros((len(scans)*len(healthy_scans), len(thresh_list))) # each row is a curve
	column_counter = 0
	for subj_num, SCI_net in enumerate(SCI_nets):
		for healthy_net in healthy_nets:
			for count, threshold in enumerate(thresh_list):
				print('Threshold: %d, pair: %d/%d' % (threshold, column_counter, curve_mat.shape[0]))
				threshold = threshold/100.0
				curve_mat[column_counter, count] = compare_two_graphs(SCI_net.binarize(threshold).data, healthy_net.binarize(threshold).data, 4, 4)
			column_counter += 1
	mdb = mongodb_database.MongoDBDatabase(None)
	mdb.put_temp_data(curve_mat, 'graph kernel SCI 1 x HC', 'The network similarity of each pair of SCI 1 and HC calculated by graph kernel method with d = 4, h = 4. The binary threshold is varied range(60, 100). The data is of shape (cases, thresh_list length). Each row is a curve.')

def plot_healthy_patient_curve():
	mdb = mongodb_database.MongoDBDatabase(None)
	curve_mat_patients = mdb.get_temp_data('graph kernel SCI 1 x HC')['value']
	curve_mat_HC = mdb.get_temp_data('graph kernel inter-healthy')['value']
	print('data loaded')
	mean_list = [None] * len(thresh_list)
	healthy_bounds_mean = [None] * len(thresh_list)
	upper_bound_list = [None] * len(thresh_list)
	healthy_bounds_upper = [None] * len(thresh_list)
	lower_bound_list = [None] * len(thresh_list)
	healthy_bounds_lower = [None] * len(thresh_list)
	for idx in range(len(thresh_list)):
		m, l, u = stats_utils.mean_confidence_interval(curve_mat_patients[:, idx])
		mean_list[idx] = m
		upper_bound_list[idx] = u
		lower_bound_list[idx] = l
		m, l, u = stats_utils.mean_confidence_interval(curve_mat_HC[:, idx])
		healthy_bounds_mean[idx] = m
		healthy_bounds_upper[idx] = u
		healthy_bounds_lower[idx] = l
	plt.figure()
	plt.plot(thresh_list, mean_list, '.-r', label = 'SCI mean')
	plt.plot(thresh_list, lower_bound_list, '.-y', label = 'SCI lower bound')
	plt.plot(thresh_list, upper_bound_list, '.-y', label = 'SCI upper bound')

	plt.plot(thresh_list, healthy_bounds_mean, '.-g', label = 'HC mean')
	plt.plot(thresh_list, healthy_bounds_lower, '.-c', label = 'HC lower bound')
	plt.plot(thresh_list, healthy_bounds_upper, '.-c', label = 'HC upper bound')

	plt.fill_between(thresh_list, lower_bound_list, upper_bound_list, alpha = 0.25, color = 'r')
	plt.fill_between(thresh_list, healthy_bounds_lower, healthy_bounds_upper, alpha = 0.25, color = 'g')
	
	plt.xticks(range(60, 101, 5), range(60, 101, 5))
	plt.grid(True)
	plt.legend()
	plt.title('SCI 1 vs HC, h = 4, d = 4')
	plt.xlabel('Threshold * 100')
	plt.ylabel('Similarity')
	plt.savefig(WORK_PREFIX + 'SCI 1 HC.png', dpi = 300)
	plt.close()

def plot_range_threshold_curve():
	atlasobj = atlas.get('brodmann_lrce')
	mdb = mongodb_database.MongoDBDatabase(None)
	threshold_list = range(5, 100, 5)
	h_list = list(range(1, int(atlasobj.count/2)))
	d_list = list(range(2, 10))
	plt.figure()
	for threshold in threshold_list:
		curve = mdb.get_temp_data('gk chen range h threshold %d' % threshold)['value']
		plt.plot(h_list, curve, label = 'threshold %d' % threshold)
	plt.xticks(range(1, int(atlasobj.count/2), 3), range(1, int(atlasobj.count/2), 3))
	plt.grid(True)
	plt.legend(loc = 'upper right')
	plt.xlabel('Depth (h)')
	plt.ylabel('Similarity')
	plt.title('Chen Network similarity d = 4')
	plt.savefig(WORK_PREFIX + 'Chen range h.png', dpi = 300)
	plt.close()

if __name__ == '__main__':
	plot_range_threshold_curve()
