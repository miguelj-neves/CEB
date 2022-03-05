"Modified from Afshine Amidi & Shervine Amidi's blog"
# Usage:  Generator for Keras
# Date:   5 Mar 2021
# Author: Jun ZHU
# Email:  Jun__Zhu@outlook.com


import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical as onehot
from config import Config, HyperParams, Dir


class DataGenerator(Sequence):
	"""Custom data generator for CEB
		list_IDs: the id of those test recordings
		data_dir: the folder storing the waveform files
		num_classes: the number of classes
		batch_size: batch size of an epoch
		SliceLength: the number of data points in each sample's input
		MaxCutOnset: the onset is randomly chosen in this range before P arrival
		shuffle: True for training mode; False for test or predict mode
		TypeConvertLabel: a python dictionary defining how to convert the earthquake type to digital label
	"""

	def __init__(self,
			list_IDs,
			data_dir='..',
			num_classes=2,
			batch_size=128,
			SliceLength=3000,
			MaxCutOnset=500,
			shuffle=True,
			TypeConvertLabel={"quake":0, "blast":1}):
		"""initialize the class
			params:
					1. list_IDs: ID of the waveform file e.g., "1.npz"
		"""

		self.data_dir = data_dir
		self.list_IDs = list_IDs
		self.TypeConvertLabel = TypeConvertLabel
		self.shape = (SliceLength, 1)
		self.MaxCutOnset = MaxCutOnset
		self.SliceLength = SliceLength
		self.batch_size = batch_size
		self.num_channels = 3
		self.num_classes = num_classes
		self.shuffle = shuffle
		# shuffle if necessary
		self.on_epoch_end()

	def __len__(self):
		"""length of the mini-batch"""

		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		"""get mini-batch data for fit_generator in Keras"""

		indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
		list_IDs_tmp = [self.list_IDs[k] for k in indexes]
		X, y = self.__data_generation(list_IDs_tmp)
		return X, y

	def on_epoch_end(self):
		"""shuffle at the end of each epoch"""

		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_tmp):
		"""generate data of mini-batch size"""

		X = np.empty((self.batch_size, *self.shape, self.num_channels))
		y = np.empty((self.batch_size), dtype=int)
		for i, ID in enumerate(list_IDs_tmp):
			data = np.load(os.path.join(self.data_dir, str(ID)+'.npz'))
			# Random sliding window avoids learning a fixed windowing scheme.
			# Please refer to Weiqiang's paper for more detail: https://doi.org/10.1093/gji/ggy423
			randomonset = np.random.randint(self.MaxCutOnset)
#			randomonset = 499
			# a sample with a random onset
			sample = np.transpose(data['wf'][:, randomonset: randomonset+self.SliceLength])
			# normalized by the maximum standard error among three components'
			X[i, ] = sample.reshape(*self.shape, self.num_channels) /	np.max(np.std(sample, axis=0))
			y[i] = self.TypeConvertLabel[str(data['label'])]
		return X, onehot(y, num_classes=self.num_classes)


if __name__ == "__main__":
	conf = Config(); hp = HyperParams(); Dir = Dir()
	# parameters for generator
	params = {
			'data_dir': Dir.waveform,
			'batch_size': hp.batch_size,
			'MaxCutOnset': conf.MaxCutOnset,
			'SliceLength': conf.newnpts,
			'num_classes': len(conf.type_to_label),
			'shuffle': False, # False for "test" or "predict" mode
			'TypeConvertLabel': conf.type_to_label}
	# Datasets
	import pandas as pd
	df = pd.read_csv(Dir.fname_csv,  delimiter=" ")
	partition = df['id']
	# Generator
	train_generator = DataGenerator(partition, **params)
	# test for a sample
	X, Y = train_generator.__getitem__(0)
	sample_in_batch = np.random.randint(params['batch_size'])
	x, y = X.__getitem__(sample_in_batch), Y.__getitem__(sample_in_batch)
	print("shape of input: ", x.shape, "target output:", y)
#	# plot this sample
#	import matplotlib.pyplot as plt
#	fig, ax = plt.subplots(3, 1, sharex=True)
#	ax[0].plot(x[:,:,0], color='k', lw=0.2)
#	ax[1].plot(x[:,:,1], color='k', lw=0.2)
#	ax[2].plot(x[:,:,2], color='k', lw=0.2)
#	plt.suptitle("Sample waveform")
#	plt.show()
#	plt.close()
