# -*- coding: UTF-8 -*-

# import tensorflow as tf

import input_data_for_patient as input_data
import numpy as np

filepath = '/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4'
train_size, test_size = input_data.get_size(filepath)
train_dirs = input_data.get_train_dir(filepath)
test_dirs = input_data.get_test_dir(filepath)

for filename in train_dirs:

	label = np.load(filepath + '/' + filename + '/label.npy')
	labels = np.array([[0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0]])
	for i in range(4):
		for j in range(4):
			labels[i][label[j]] += 1

print(labels)