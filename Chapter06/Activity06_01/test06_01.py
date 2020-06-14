import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import tensorflow as tf
from tensorflow.keras.layers import Dense


class Test(unittest.TestCase):
	def setUp(self):
		import Activity06_01
		self.exercises = Activity06_01

		self.usecols = ['AAGE','ADTIND','ADTOCC','SEOTR','WKSWORK', 'PTOTVAL']
		self.train_url = 'https://github.com/PacktWorkshops/The-TensorFlow-Workshop/blob/master/Chapter06/dataset/census-income-train.csv?raw=true'
		self.test_url = 'https://github.com/PacktWorkshops/The-TensorFlow-Workshop/blob/master/Chapter06/dataset/census-income-test.csv?raw=true'

		self.train_data = pd.read_csv(self.train_url, usecols=self.usecols)
		self.train_target = self.train_data.pop('PTOTVAL')
		self.test_data = pd.read_csv(self.test_url, usecols=self.usecols)
		self.test_target = self.test_data.pop('PTOTVAL')

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		fc1 = Dense(1048, input_shape=(5,), activation='relu')
		fc2 = Dense(512, activation='relu')
		fc3 = Dense(128, activation='relu')
		fc4 = Dense(64, activation='relu')
		fc5 = Dense(1)

		self.model.add(fc1)
		self.model.add(fc2)
		self.model.add(fc3)
		self.model.add(fc4)
		self.model.add(fc5)

		optimizer = tf.keras.optimizers.Adam(0.05)

		self.model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

		self.model2 = tf.keras.Sequential()

		reg_fc1 = Dense(1048, input_shape=(5,), activation='relu',
						kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))
		reg_fc2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))
		reg_fc3 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))
		reg_fc4 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))
		reg_fc5 = Dense(1, activation='relu')

		self.model2.add(reg_fc1)
		self.model2.add(reg_fc2)
		self.model2.add(reg_fc3)
		self.model2.add(reg_fc4)
		self.model2.add(reg_fc5)

		self.model2.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mse', metrics=['mse'])

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

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())

	def test_summary2(self):
		self.assertEqual(self.exercises.model2.summary(), self.model2.summary())


if __name__ == '__main__':
	unittest.main()
