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
		import Exercise06_01
		self.exercises = Exercise06_01

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter06/dataset/connect-4.csv'

		self.data = pd.read_csv(self.file_url)
		self.target = self.data.pop('class')
		self.X_test = pd.read_csv(self.test_url, header=None)
		self.y_test = self.X_test.pop(0)

		self.y_train = self.y_train.replace(-1, 0)
		self.y_test = self.y_test.replace(-1, 0)

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		fc1 = Dense(512, input_shape=(42,), activation='relu')
		fc2 = Dense(512, activation='relu')
		fc3 = Dense(128, activation='relu')
		fc4 = Dense(128, activation='relu')
		fc5 = Dense(3, activation='softmax')

		self.model.add(fc1)
		self.model.add(fc2)
		self.model.add(fc3)
		self.model.add(fc4)
		self.model.add(fc5)

		loss = tf.keras.losses.SparseCategoricalCrossentropy()
		optimizer = tf.keras.optimizers.Adam(0.001)

		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

	def test_file_url(self):
		self.assertEqual(self.exercises.file_url, self.file_url)

	def data(self):
		pd_testing.assert_frame_equal(self.exercises.data, self.data)

	def test_target(self):
		np_testing.assert_array_equal(self.exercises.target, self.target)

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())


if __name__ == '__main__':
	unittest.main()
