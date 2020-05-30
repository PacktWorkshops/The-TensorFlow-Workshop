import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import get_file
from tensorflow.keras.metrics import Accuracy, Precision, Recall

class Test(unittest.TestCase):
	def setUp(self):
		import Exercise05_02
		self.exercises = Exercise05_02

		self.train_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/dota2PreparedTrain.csv'
		self.test_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-TensorFlow-Workshop/master/Chapter05/dataset/dota2PreparedTest.csv'

		self.X_train = pd.read_csv(self.train_url, header=None)
		self.y_train = self.X_train.pop(0)
		self.X_test = pd.read_csv(self.test_url, header=None)
		self.y_test = self.X_test.pop(0)

		self.model_url = 'https://github.com/PacktWorkshops/The-TensorFlow-Workshop/blob/master/Chapter05/model/exercise5_01_model.h5?raw=true'
		self.model_path = get_file('exercise5_01_model.h5', self.model_url)

		np.random.seed(8)
		tf.random.set_seed(8)

		self.model = tf.keras.models.load_model(self.model_path)

		self.preds_proba = self.model.predict(self.X_test)
		self.preds = self.preds_proba >= 0.5

		acc = Accuracy()
		prec = Precision()
		rec = Recall()

		acc.update_state(self.preds, self.y_test)
		self.acc_results = acc.result().numpy()

		prec.update_state(self.preds, self.y_test)
		self.prec_results = prec.result().numpy()

		rec.update_state(self.preds, self.y_test)
		self.rec_results = rec.result().numpy()

		self.f1 = 2 * (self.prec_results * self.rec_results) / (self.prec_results + self.rec_results)

	def test_train_url(self):
		self.assertEqual(self.exercises.train_url, self.train_url)

	def test_test_url(self):
		self.assertEqual(self.exercises.test_url, self.test_url)

	def test_model_url(self):
		self.assertEqual(self.exercises.model_url, self.model_url)

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

	def test_acc_results(self):
		np_testing.assert_array_equal(self.exercises.acc_results, self.acc_results)

	def test_prec_results(self):
		np_testing.assert_array_equal(self.exercises.prec_results, self.prec_results)

	def test_rec_results(self):
		np_testing.assert_array_equal(self.exercises.rec_results, self.rec_results)

	def test_f1(self):
		np_testing.assert_array_equal(self.exercises.f1, self.f1)


if __name__ == '__main__':
	unittest.main()
