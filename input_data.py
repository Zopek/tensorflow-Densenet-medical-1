# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

def get_num(filepath):

	train_size = 0
	test_size = 0
	# 训练集在3个文件夹中，每个文件夹中数据数相同，总数目乘3可得
	dirs = os.listdir(filepath + '/' + str(1))
	train_size += len(dirs)

	# 测试集在1个文件夹中
	dirs = os.listdir(filepath + '/' + str(4))
	test_size += len(dirs)

	return train_size, test_size

def label_convert(label):

	labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	labels[int(label[0])] = 1
	labels[int(label[1]) + 4] = 1
	labels[int(label[2]) + 7] = 1
	labels[int(label[3]) + 11] = 1

	return labels.reshape((1, -1)) 

def train_next_batch(filepath, step, batch_size):

	train_size, test_size = get_num(filepath)
	current_num = step * batch_size

	# 由于文件夹命名从1开始，故+1
	current_file = current_num / train_size + 1
	next_file = (current_num + batch_size) / train_size + 1
	# 如果当前文件夹剩余数据不足一个batch，则与下一个文件夹数据合并
	if current_file != next_file:
		# 获取当前文件夹下文件名列表
		dirs = os.listdir(filepath + '/' + str(current_file))
		# 取出当前文件夹剩余数据
		for filename in dirs[current_num:train_size]:
			# 取出图像
			image = np.load(filepath + '/' + str(current_file) + '/' + filename + '/image.npy')
			image = image.reshape((1, -1))
			if images == []:
				images = image
			else:
				images = np.vstack((images, image))
			# 取出标签
			label = np.load(filepath + '/' + str(current_file) + '/' + filename + '/label.npy')
			label = label_convert(label)
			if filename == dirs[0]:
				labels = label
			else:
				labels = np.vstack((labels, label))

		# 进入下一文件夹
		dirs = os.listdir(filepath + '/' + str(next_file))
		# 下一文件夹需补充数据数
		rest_num = current_num + batch_size - train_size * current_file
		# 取出下一文件夹数据
		for filename in dirs[0:rest_num]:
			# 取出图像
			image = np.load(filepath + '/' + str(next_file) + '/' + filename + '/image.npy')
			image = image.reshape((1, -1))
			images = np.vstack((images, image))
			# 取出标签
			label = np.load(filepath + '/' + str(next_file) + '/' + filename + '/label.npy')
			label = label_convert(label)
			labels = np.vstack((labels, label))

		return images, labels
	else:
		dirs = os.listdir(filepath + '/' + str(current_file))
		for filename in dirs[current_num:(current_num + batch_size)]:
			# 取出图像
			image = np.load(filepath + '/' + str(current_file) + '/' + filename + '/image.npy')
			image = image.reshape((1, -1))
			if images == []:
				images = image
			else:
				images = np.vstack((images, image))
			# 取出标签
			label = np.load(filepath + '/' + str(current_file) + '/' + filename + '/label.npy')
			label = label_convert(label)
			if filename == dirs[0]:
				labels = label
			else:
				labels = np.vstack((labels, label))

		return images, labels

def test_next_batch(filepath, step, batch_size):

	dirs = os.listdir(filepath + '/' + str(4))
	for filename in dirs[0:300]:
		# 取出图像
		image = np.load(filepath + '/' + str(current_file) + '/' + filename + '/image.npy')
		image = image.reshape((1, -1))
		if images == []:
			images = image
		else:
			images = np.vstack((images, image))
		# 取出标签
		label = np.load(filepath + '/' + str(current_file) + '/' + filename + '/label.npy')
		label = label_convert(label)
		if filename == dirs[0]:
			labels = label
		else:
			labels = np.vstack((labels, label))

	return images, labels