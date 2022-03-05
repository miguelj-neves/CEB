# Usage:	Predict the event types based on the 50 second waveform
# Author:	Jun ZHU
# Date:		5 Mar 2022
# Email:	Jun__Zhu@outlook.com


import os
import numpy as np
import pandas as pd
from keras.models import load_model
from config import Dir, Config, HyperParams
from datagenerator import DataGenerator


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
	Dir = Dir(); conf = Config(); hp = HyperParams()
	model = load_model(Dir.model)
	params = {
		'data_dir': Dir.waveform,
		'batch_size': 100,
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceLength': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': False,
		'TypeConvertLabel': conf.type_to_label}
	df = pd.read_csv(Dir.fname_csv, delimiter=" ")
	predict_id = df['id']
	miss = len(df)%params['batch_size']
	if miss!=0:
		print("\n-------------------------------\nThere are %d samples are not included in a batch.\nPlease choose another batch size which is the divisor of your data, or set the batch size as 1\n-------------------------------\n"%(miss))
		exit(0)
	predict_generator = DataGenerator(predict_id, **params)
	Predict(predict_id, predict_generator, model, output_dir=Dir.result)
