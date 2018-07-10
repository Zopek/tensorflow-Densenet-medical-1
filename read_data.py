  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed
# path = "/DATA/data/qyzheng/lesion_one_normal_4"
# /DATA/data/qyzheng/lesion_one_normal_4/1/1020959_0_2
# index_in_epoch = 0

class DataSet(object):

  def __init__(self, images, 
  	labels):
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = images.shape[0]

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def label_convert(label):

	labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	labels[int(label[0])] = 1
	labels[int(label[1]) + 4] = 1
	labels[int(label[2]) + 7] = 1
	labels[int(label[3]) + 11] = 1

	return labels.reshape((1, -1)) 

def read_data_sets(filepath, data_num):

	dirs = os.listdir(filepath + '/' + str(data_num))
	for filename in dirs:
		# read image from dir
		image = np.load(filepath + '/' + str(data_num) + '/' + filename + '/image.npy')
		# from (224, 224) to (1, 50176)
		image = image.reshape((1, -1))
		if filename == dirs[0]:
			images = image
		else:
			images = np.vstack((images, image))

		# read label from dir
		label = np.load(filepath + '/' + str(data_num) + '/' + filename + '/label.npy')
		# from (4,) to (1, 14)
		label = label_convert(label)
		if filename == dirs[0]:
			labels = label
		else:
			labels = np.vstack((labels, label))

		if filename == dirs[1000]:
			break

	return images, labels

def load_all_sets(filepath, validation_size = 1):

#	filepath = '/DATA/data/qyzheng/lesion_one_normal_4'

	# take previous three datasets as train
	for i in range(1, 2):
		train_image, train_label = read_data_sets(filepath, i)
		if i == 1:
			train_images = train_image
			train_labels = train_label
		else:
			train_images = np.vstack((train_images, train_image))
			train_labels = np.vstack((train_labels, train_label))
		print('train is: ', train.shape)

	test_images, test_labels = read_data_sets(filepath, 4)

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]

	train = DataSet(train_images, train_labels)
	validation = DataSet(validation_images, validation_labels)
	test = DataSet(test_images, test_labels)

	return base.Datasets(train=train, validation=validation, test=test)
