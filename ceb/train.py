# Usage:	Predict the event types based on the 50 second waveform
# Author:	Jun ZHU
# Date:		5 Mar 2022
# Email:	Jun__Zhu@outlook.com


import os
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
from config import Dir, Config, HyperParams
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
	parser.add_argument("--result_dir",
						default="../results",
						type=str,
						help="../results")
	args = parser.parse_args()
	return args


def Predict(predict_id, predict_generator, model, output_dir):
	# predict the label of the sample
	proba = model.predict(predict_generator)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	fname = os.path.join(output_dir, "predict.txt")
	text = ["%s %.2f %.2f"%(x, y[0], y[1]) for x,y in zip(predict_id, proba)]
	with open(fname, 'w') as f:
		dic = conf.label_to_type
		f.write('%s %s %s\n'%('id', dic[0], dic[1]))
		f.write('\n'.join(text))
	return proba


if __name__ == "__main__":
	args = read_args()
	Dir = Dir(); conf = Config(); hp = HyperParams()
	model = load_model(args.model)
	params = {
		'data_dir': args.data_dir,
		'batch_size': 100,
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceLength': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': False,
		'TypeConvertLabel': conf.type_to_label}
	df = pd.read_csv(args.data_list, delimiter=" ")
	predict_id = df['id']
	miss = len(df)%params['batch_size']
	if miss!=0:
		print("\n-------------------------------\nThere are %d samples are not included in a batch.\nPlease choose another batch size which is the divisor of your data, or set the batch size as 1\n-------------------------------\n"%(miss))
		exit(0)
	predict_generator = DataGenerator(predict_id, **params)
	Predict(predict_id, predict_generator, model, output_dir=args.result_dir)
