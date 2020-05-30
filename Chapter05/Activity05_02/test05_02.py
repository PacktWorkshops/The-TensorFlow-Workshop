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
		import Activity05_02
		self.exercises = Activity05_02

		self.feature_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/IMDB-F-features.csv'
		self.feature = pd.read_csv(self.feature_url)

		self.target_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/IMDB-F-targets.csv'
		self.target = pd.read_csv(self.target_url)

		self.X_train = self.feature[:15000]
		self.y_train = self.target[:15000]
		self.X_test = self.feature[15000:]
		self.y_test = self.target[15000:]

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.Sequential()

		fc1 = Dense(512, input_shape=(1001,), activation='relu')
		fc2 = Dense(512, activation='relu')
		fc3 = Dense(128, activation='relu')
		fc4 = Dense(128, activation='relu')
		fc5 = Dense(28, activation='sigmoid')

		self.model.add(fc1)
		self.model.add(fc2)
		self.model.add(fc3)
		self.model.add(fc4)
		self.model.add(fc5)

		loss = tf.keras.losses.BinaryCrossentropy()
		optimizer = tf.keras.optimizers.Adam(0.001)

		self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
		self.model.fit(self.X_train, self.y_train, epochs=20)

		self.preds_proba = self.model.predict(self.X_test)
		self.preds = self.preds_proba.argmax(axis=1)

	def test_feature_url(self):
		self.assertEqual(self.exercises.feature_url, self.feature_url)

	def test_feature(self):
		pd_testing.assert_frame_equal(self.exercises.feature, self.feature)

	def test_target_url(self):
		self.assertEqual(self.exercises.target_url, self.target_url)

	def test_target(self):
		pd_testing.assert_frame_equal(self.exercises.target, self.target)

	def test_X_train(self):
		pd_testing.assert_frame_equal(self.exercises.X_train, self.X_train)

	def test_y_train(self):
		pd_testing.assert_frame_equal(self.exercises.y_train, self.y_train)

	def test_X_test(self):
		pd_testing.assert_frame_equal(self.exercises.X_test, self.X_test)

	def test_y_test(self):
		pd_testing.assert_frame_equal(self.exercises.y_test, self.y_test)

	def test_summary(self):
		self.assertEqual(self.exercises.model.summary(), self.model.summary())


if __name__ == '__main__':
	unittest.main()
