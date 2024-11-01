# Usage:	Instantiate a CNN model as an earthquake-blast classifier
# Date:		5 Mar 2022
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K


def lr_schedule(epoch, lr=1e-3):
	"""the schedule for learning rate
		lr: the learning rate of the first epoch
	"""

	if epoch > 40:
		lr *= .5e-3
	elif epoch > 20:
		lr *= 1e-3
	elif epoch >10:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr


def cnn(input_shape, dropout=0.3, lr=0.1, num_classes_output=2, l2_damping=1e-4):
	"""set the architecture and configuration of the CNN model
		params:
			1. input shape
			2. dropout rate
			3. **kwargs: learning rate; number of classes
			4. l2_damping: the damping factor of the L2 regularization
		output:
			1. the CNN model
	"""

	K.clear_session()
	# configuration of the convolutional layer
	kernel_size, conv_strides = (3, 1), (2, 1)
	# configuration of the max pooling layer
	pool_size, pool_strides = (3, 1), 1
	# loss function
	loss_function = 'categorical_crossentropy'
	# construction of a CNN model
	classifier = Sequential([
		#-----------------------------7 conv layers----------------------
		Conv2D(8, kernel_size=kernel_size, strides=conv_strides, activation='relu',
			input_shape=input_shape, kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		Conv2D(8, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		Conv2D(16, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		Conv2D(32, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		Conv2D(64, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		Conv2D(128, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		Conv2D(256, kernel_size=kernel_size, strides=conv_strides,
			activation='relu', kernel_regularizer=l2(l2_damping)),
		BatchNormalization(),
		MaxPooling2D(pool_size=pool_size, strides=pool_strides),

		#-----------------------------a big dropout before flatten-------
		Dropout(dropout),

		#-----------------------------flatten----------------------------
		Flatten(),

		#-----------------------------dense------------------------------
		Dense(num_classes_output, activation = 'softmax')],
		name = 'Classifier')
	# compile the CNN model
	#classifier.compile(loss=loss_function, metrics=['accuracy'],
	#		optimizer=Adam(learning_rate=lr_schedule(0, lr=lr)))
	return classifier


if __name__=="__main__":
	from config import HyperParams, Dir, Config
	hp = HyperParams(); conf = Config()
	dropout = hp.dropout
	lr = hp.lr
	num_classes_output = conf.num_classes
	input_shape = conf.input_shape
	model = cnn(input_shape, dropout, lr=lr,num_classes_output=num_classes_output)
	model.summary()
