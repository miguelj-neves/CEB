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


class DataGenerator_Train(Sequence):
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
			SliceLength=5000,
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

		def channel_drop(sample):
			"""drop one channel randomly"""
			nchannels = np.random.randint(2)
			if nchannels == 0:
				channel_drop = np.random.randint(3)
				sample[:, channel_drop] = np.zeros(sample.shape[0])
			else:
				channel_drop = np.random.choice([0,1,2], size=2, replace=False)
				sample[:, channel_drop[0]] = np.zeros(sample.shape[0])
				sample[:, channel_drop[1]] = np.zeros(sample.shape[0])
			return sample
	
		def add_gaps(sample):
			"""add gaps to the waveform"""
			gap_length = np.random.randint(100, 1100)
			gap_indexes = np.arange(gap_length)+np.random.randint(sample.shape[0]-gap_length)
			sample[gap_indexes,:] = np.zeros((gap_length, sample.shape[1]))
			return sample

		def waveform_clip(sample):
			"""clip the waveform"""
			percent = np.random.randint(1, 50)
			sample = np.clip(sample, -1*np.percentile(abs(sample), percent), np.percentile(abs(sample), percent))
			return sample

		def add_noise(sample):
			"""add noise to the waveform"""
			noise = np.random.normal(-1, 1, sample.shape)
			sample = sample + noise*np.nanmax(abs(sample), axis=1).reshape(-1,1)*0.2
			return sample
	
		X = np.zeros((self.batch_size, *self.shape, self.num_channels))
		y = np.empty((self.batch_size), dtype=int)
		for i, ID in enumerate(list_IDs_tmp):
			data = np.load(os.path.join(self.data_dir, str(ID)+'.npz'))
			# Random sliding window avoids learning a fixed windowing scheme.
			# Please refer to Weiqiang's paper for more detail: https://doi.org/10.1093/gji/ggy423
			randomonset = np.random.randint(self.MaxCutOnset)
#			randomonset = 499
			# a sample with a random onset
			sample = np.transpose(data['wf'][:, randomonset:randomonset+self.SliceLength])
			sample = np.nan_to_num(sample, nan=0.0)
			#print(np.isnan(np.sum(sample)))
			psignal = np.nansum((data['wf'][:, 500:1000] - np.nanmean(data['wf'][:, 500:1000], axis=1).reshape(-1,1))**2)
			pnoise = np.nansum((data['wf'][:, :500 ] - np.nanmean(data['wf'][:, :500], axis=1).reshape(-1,1))**2)
			snr = np.mean(10*np.log10(psignal/pnoise))
			if snr > 1:
				augment_method = np.random.choice([0, 1, 2 , 3], p=[0.3, 0.2, 0.3, 0.2])
				#print("Augment", augment_method)
				#augment_method = np.random.choice([0, 1, 2 , 3], p=[0, 0, 0, 1])
				if augment_method == 0:
					sample = channel_drop(sample)
				elif augment_method == 1:
					sample = add_gaps(sample)
				elif augment_method == 2:
					sample = waveform_clip(sample)
				elif augment_method == 3:
					sample = add_noise(sample)
			# normalized by the maximum standard error among three components'
			#print(np.max(np.std(sample, axis=0)))
			normalization = np.max(np.std(sample, axis=0)) + 0.00000001
			#print(normalization)
			X[i, :sample.shape[0]] = sample.reshape(sample.shape[0], 1, self.num_channels) /	normalization
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
	train_generator = DataGenerator_Train(partition, **params)
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
