import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import tensorflow as tf
from tensorflow.keras.layers import Dense
import kerastuner as kt


def model_builder(hp):
	model = tf.keras.Sequential()

	hp_l2 = hp.Choice('l2', values=[0.1, 0.01, 0.001])
	hp_units = hp.Int('units', min_value=128, max_value=512, step=64)

	reg_fc1 = Dense(hp_units, input_shape=(5,), activation='relu',
					kernel_regularizer=tf.keras.regularizers.l2(l=hp_l2))
	reg_fc2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=hp_l2))
	reg_fc3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=hp_l2))
	reg_fc4 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=hp_l2))
	reg_fc5 = Dense(1)

	model.add(reg_fc1)
	model.add(reg_fc2)
	model.add(reg_fc3)
	model.add(reg_fc4)
	model.add(reg_fc5)

	hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])

	optimizer = tf.keras.optimizers.Adam(hp_learning_rate)
	model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

	return model


class Test(unittest.TestCase):
	def setUp(self):
		import Activity06_02
		self.exercises = Activity06_02

		self.usecols = ['AAGE','ADTIND','ADTOCC','SEOTR','WKSWORK', 'PTOTVAL']
		self.train_url = 'https://github.com/PacktWorkshops/The-TensorFlow-Workshop/blob/master/Chapter06/dataset/census-income-train.csv?raw=true'
		self.test_url = 'https://github.com/PacktWorkshops/The-TensorFlow-Workshop/blob/master/Chapter06/dataset/census-income-test.csv?raw=true'

		self.train_data = pd.read_csv(self.train_url, usecols=self.usecols)
		self.train_target = self.train_data.pop('PTOTVAL')
		self.test_data = pd.read_csv(self.test_url, usecols=self.usecols)
		self.test_target = self.test_data.pop('PTOTVAL')

		np.random.seed(8)
		tf.random.set_seed(8)

		tuner = kt.BayesianOptimization(model_builder, objective='val_mse', max_trials=10)
		tuner.search(self.train_data, self.train_target, validation_data=(self.test_data, self.test_target))
		best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
		self.best_units = best_hps.get('units')
		self.best_lr = best_hps.get('learning_rate')
		self.best_l2 = best_hps.get('l2')
		self.model = tuner.hypermodel.build(best_hps)

	def test_usecols(self):
		np_testing.assert_array_equal(self.exercises.usecols, self.usecols)

	def test_train_url(self):
		self.assertEqual(self.exercises.train_url, self.train_url)

	def test_test_url(self):
		self.assertEqual(self.exercises.test_url, self.test_url)

	def test_train_data(self):
		pd_testing.assert_frame_equal(self.exercises.train_data, self.train_data)

	def test_test_data(self):
		pd_testing.assert_frame_equal(self.exercises.test_data, self.test_data)

	def test_train_target(self):
		np_testing.assert_array_equal(self.exercises.train_target, self.train_target)

	def test_test_target(self):
		np_testing.assert_array_equal(self.exercises.test_target, self.test_target)

	def test_best_units(self):
		self.assertEqual(self.exercises.best_units, self.best_units)

	def test_best_lr(self):
		self.assertEqual(self.exercises.best_lr, self.best_lr)

	def test_best_l2(self):
		self.assertEqual(self.exercises.best_l2, self.best_l2)

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())

if __name__ == '__main__':
	unittest.main()
