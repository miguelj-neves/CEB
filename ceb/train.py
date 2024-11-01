# Usage:	Predict the event types based on the 50 second waveform
# Author:	Jun ZHU
# Date:		5 Mar 2022
# Email:	Jun__Zhu@outlook.com


import os
import argparse
import numpy as np
import pandas as pd
from keras.models import load_model
#from tensorflow.keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from config import Dir, Config, HyperParams
from datagenerator import DataGenerator
from datagenerator_train import DataGenerator_Train
from cnn_classifier import cnn

import platform
if platform.system() == "Darwin" and platform.processor() == "arm":
    from tensorflow.keras.optimizers.legacy import Adam
else:
    from tensorflow.keras.optimizers import Adam

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--starting_model",
						default=None,
						type=str,
						help="model/eastky_classifier.h5")
	parser.add_argument("--train_list",
						default="../dataset/waveform_train.csv",
						type=str,
						help="../dataset/waveform_train.csv")
	parser.add_argument("--valid_list",
						default="../dataset/waveform_valid.csv",
						type=str,
						help="../dataset/waveform_valid.csv")
	parser.add_argument("--data_dir",
						default="../dataset/waveform_pred",
						type=str,
						help="../dataset/waveform_pred")
	parser.add_argument("--result_dir",
						default="./new_model",
						type=str,
						help="./new_model")
	parser.add_argument("--output_model",
						default="new_model.h5",
						type=str,
						help="new_model.h5")
	parser.add_argument("--initial_lr",
						default=0.01,
						type=float,
						help="learning rate")
	args = parser.parse_args()
	return args

def lr_schedule(epoch, lr=1e-4):
	"""the schedule for learning rate
		lr: the learning rate of the first epoch
	"""

	if epoch > 40:
		lr *= .5e-3
	elif epoch > 20:
		lr *= 1e-2
	elif epoch >10:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr

def Train(train_generator, validate_generator, model, output_dir, output_name, initial_lr):
	# configure the model
	#scheduler = LearningRateScheduler(lr_schedule)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, min_delta = 0.01, verbose = 1)
	earlystop = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True, min_delta = 0.01, verbose = 1)
	best_save = ModelCheckpoint(output_dir+'/{epoch:02d}-{val_loss:.3f}.best_val.hdf5', save_best_only=False, monitor='val_loss', save_weights_only=False, mode="auto", save_freq="epoch", initial_value_threshold=None)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=initial_lr), metrics=['categorical_crossentropy', 'accuracy'])

	print(model.optimizer.get_config())

	# train the model
	history = model.fit(train_generator, validation_data=validate_generator, epochs=hp.epoch, callbacks=[reduce_lr, earlystop, best_save], verbose=1, workers=4, use_multiprocessing=True)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	fname = os.path.join(output_dir, output_name)
	model.save(fname)
	np.save(output_dir+"/"+output_name[:-3]+".npy",history.history)
	return history


if __name__ == "__main__":
	args = read_args()
	Dir = Dir(); conf = Config(); hp = HyperParams()
	
	params = {
		'data_dir': args.data_dir,
		'batch_size': 128,
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceLength': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': True,
		'TypeConvertLabel': conf.type_to_label}
	params2 = {
		'data_dir': args.data_dir,
		'batch_size': 128,
		'MaxCutOnset': conf.MaxCutOnset,
		'SliceLength': conf.newnpts,
		'num_classes': len(conf.type_to_label),
		'shuffle': False,
		'TypeConvertLabel': conf.type_to_label}
	dropout = hp.dropout
	num_classes_output = conf.num_classes
	input_shape = conf.input_shape
	
	if args.starting_model is None:
		lr = args.initial_lr
		l2_damping = 1e-4
		dropout = 0.3
		model = cnn(input_shape, dropout, lr=lr,num_classes_output=num_classes_output, l2_damping=l2_damping)
	else:
		lr = args.initial_lr
		l2_damping = 1e-3
		dropout = 0.3
		original_model = load_model(args.starting_model)
		original_model.trainable = False
		model = cnn(input_shape, dropout, lr=lr,num_classes_output=num_classes_output, l2_damping=l2_damping)
		# Copy the weights from the original model to the new model
		model.set_weights(original_model.get_weights())
		model.trainable = True
	
	
	
	train_df = pd.read_csv(args.train_list, delimiter=" ")
	new_train_df = pd.DataFrame(np.repeat(train_df.values, 1, axis=0), columns=train_df.columns)
	valid_df = pd.read_csv(args.valid_list, delimiter=" ")
	#miss = len(df)%params['batch_size']
	#if miss!=0:
	#	print("\n-------------------------------\nThere are %d samples are not included in a batch.\nPlease choose another batch size which is the divisor of your data, or set the batch size as 1\n-------------------------------\n"%(miss))
	#	exit(0)
	train_generator = DataGenerator_Train(new_train_df['id'], **params)
	validate_generator = DataGenerator(valid_df['id'], **params)
	Train(train_generator, validate_generator, model, output_dir=args.result_dir, output_name=args.output_model, initial_lr=args.initial_lr)
	#predict_generator = DataGenerator(predict_id, **params)
	#Predict(predict_id, predict_generator, model, output_dir=args.result_dir)
