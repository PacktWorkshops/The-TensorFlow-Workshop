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
		import Exercise05_03
		self.exercises = Exercise05_03

		self.train_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/shuttle.trn'
		self.test_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/shuttle.tst'

		self.X_train = pd.read_table(self.train_url, header=None, sep=' ')
		self.y_train = self.X_train.pop(9)
		self.X_test = pd.read_table(self.test_url, header=None, sep=' ')
		self.y_test = self.X_test.pop(9)

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		fc1 = Dense(512, input_shape=(9,), activation='relu')
		fc2 = Dense(512, activation='relu')
		fc3 = Dense(128, activation='relu')
		fc4 = Dense(128, activation='relu')
		fc5 = Dense(8, activation='softmax')

		self.model.add(fc1)
		self.model.add(fc2)
		self.model.add(fc3)
		self.model.add(fc4)
		self.model.add(fc5)

		loss = tf.keras.losses.SparseCategoricalCrossentropy()
		optimizer = tf.keras.optimizers.Adam(0.001)

		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

	def test_train_url(self):
		self.assertEqual(self.exercises.train_url, self.train_url)

	def test_test_url(self):
		self.assertEqual(self.exercises.test_url, self.test_url)

	def test_X_train(self):
		pd_testing.assert_frame_equal(self.exercises.X_train, self.X_train)

	def test_y_train(self):
		np_testing.assert_array_equal(self.exercises.y_train, self.y_train)

	def test_X_test(self):
		pd_testing.assert_frame_equal(self.exercises.X_test, self.X_test)

	def test_y_test(self):
		np_testing.assert_array_equal(self.exercises.y_test, self.y_test)

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())


if __name__ == '__main__':
	unittest.main()
