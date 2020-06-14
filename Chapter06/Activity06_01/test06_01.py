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
		import Activity05_01
		self.exercises = Activity05_01

		self.file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/letter-recognition.data'
		self.data = pd.read_csv(self.file_url, header=None)
		self.target = self.data.pop(0)

		self.X_train = self.data[:15000]
		self.y_train = self.target[:15000]
		self.X_test = self.data[15000:]
		self.y_test = self.target[15000:]

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		fc1 = Dense(512, input_shape=(16,), activation='relu')
		fc2 = Dense(512, activation='relu')
		fc3 = Dense(128, activation='relu')
		fc4 = Dense(128, activation='relu')
		fc5 = Dense(26, activation='softmax')

		self.model.add(fc1)
		self.model.add(fc2)
		self.model.add(fc3)
		self.model.add(fc4)
		self.model.add(fc5)

		loss = tf.keras.losses.SparseCategoricalCrossentropy()
		optimizer = tf.keras.optimizers.Adam(0.001)

		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
		self.model.fit(self.X_train, self.y_train, epochs=5)

		self.preds_proba = self.model.predict(self.X_test)
		self.preds = self.preds_proba.argmax(axis=1)

	def test_file_url(self):
		self.assertEqual(self.exercises.file_url, self.file_url)

	def test_data(self):
		pd_testing.assert_frame_equal(self.exercises.data, self.data)

	def test_target(self):
		np_testing.assert_array_equal(self.exercises.target, self.target)

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

	def test_preds_proba(self):
		np_testing.assert_array_equal(self.exercises.preds_proba, self.preds_proba)

	def test_preds(self):
		np_testing.assert_array_equal(self.exercises.preds, self.preds)


if __name__ == '__main__':
	unittest.main()
