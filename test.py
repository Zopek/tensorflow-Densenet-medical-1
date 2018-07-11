# -*- coding: UTF-8 -*-

# import tensorflow as tf

import input_data

filepath = '/DATA/data/qyzheng/lesion_one_normal_4'
train_size, test_size = input_data.get_num(filepath)
print(train_size)

batch_size = 100
for step in range(3):
	images, labels = input_data.train_next_batch(filepath, step, batch_size)

	print(images.shape)
	print('OK')