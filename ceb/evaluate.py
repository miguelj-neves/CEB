# Usage:		Evaluate the ceb (Classify between Earthquake and Blast) model
# Date:			5 Mar 2022
# Author:		Jun ZHU
# Email:		Jun__Zhu@outlook.com


import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from config import Dir, HyperParams, Config
from datagenerator import DataGenerator


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model",
						default="../model/socal_classifier.h5",
						type=str,
						help="model/socal_classifier.h5")
	parser.add_argument("--data_list",
						default="../dataset/waveform.csv",
						type=str,
						help="../dataset/waveform.csv")
	parser.add_argument("--data_dir",
						default="../dataset/waveform_pred",
						type=str,
						help="../dataset/waveform_pred")
	args = parser.parse_args()
	return args

def evaluate(testgenerator, testlabel, model):
	"""evaluate the earthquake discriminator
		params:
			1. test waveform dataset
			2. test label dataset
			3. model
	"""

	score = model.evaluate(testgenerator,
							workers=conf.workers,
							max_queue_size=conf.queue_size,
							use_multiprocessing=conf.multiprocessing)
	predict_proba =	model.predict(testgenerator, use_multiprocessing=True)
	predict_class = np.argmax(predict_proba, axis=-1)
	true_class = np.array(testlabel[:len(predict_class)])
	# calculate confusiom matrix
	cfm = confusion_matrix(true_class, predict_class)
	print('\n----------------Confusion matrix------------------\n', cfm)
	report = classification_report(true_class, predict_class, target_names=["quake", "blast"])
	print('\n----------------Performance report----------------\n', report)
	print('--------------------------------------------------')
	return


if __name__ == "__main__":
	args = read_args()
	Dir = Dir(); hp = HyperParams(); conf = Config()
	model = load_model(args.model)
	params = {
		'data_dir': args.data_dir,
		'batch_size': 100,
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceLength': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': False, # must be false
		'TypeConvertLabel': conf.type_to_label}
	df = pd.read_csv(args.data_list, delimiter=" ")
	label = df['eventtype']; IDs = df['id']; evid = df['evid']
	test_generator = DataGenerator(IDs, **params)
	test_label = [conf.type_to_label[x] for x in label]
	evaluate(test_generator, test_label, model)
