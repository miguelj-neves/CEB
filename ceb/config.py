# Usage:	Configuration of the whole run
# Date:		5 Mar 2022
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com

import os


class Config():
	"""Configuration of the whole run"""

	def __init__(self):
		# buffer size for multiprocessing 
		self.queue_size = 10
		self.workers = 10
		self.multiprocessing = True
		# logging level
		self.verbose = 1

		self.sample_rate = 100 # unit: Hz
		# data augmentation
		self.OnsetSlidingRange = 5 # unit: second
		self.CutDuration = 50 # unit: second 
		self.MaxCutOnset = self.OnsetSlidingRange * self.sample_rate
		self.newnpts = self.CutDuration * self.sample_rate
		# convert to digital labels
		self.type_to_label = {"quake":0, "blast":1}
		self.label_to_type = {self.type_to_label['quake']:"quake",
				self.type_to_label['blast']:"blast"}
		# shapes of input and output
		self.num_channels = 3
		self.num_classes = 2
		self.input_shape = (self.newnpts, 1, self.num_channels)
		self.output_shape = (self.num_classes, 1)


class HyperParams():
	"""hyperparameters of the model"""

	def __init__(self):
		self.dropout = .3
		self.lr = 1e-4
		self.l2_damping = 1e-4
		self.batch_size = 128
		self.epoch = 25


class Dir():
	"""directory and file name setting"""

	def __init__(self):
		self.dataset = os.path.join("..", "dataset")
		self.waveform = os.path.join(self.dataset, "waveform_pred")
		self.fname_csv = os.path.join(self.dataset, "waveform.csv")
		# specify a model to predict the event types
		self.model = os.path.join("..", "model", "socal_classifier.h5")
#		self.model = os.path.join("..", "model", "eastky_classifier.h5")
		# the path of the result
		self.result = os.path.join("..", "results")


if __name__ == "__main__":
	conf = Config()
	hp = HyperParams()
	Dir = Dir()
