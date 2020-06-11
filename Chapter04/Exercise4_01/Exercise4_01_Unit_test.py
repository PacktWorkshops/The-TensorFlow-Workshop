import import_ipynb
import tensorflow as tf
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise4_01
        super(DataTest, self).setUp()
        self.exercise = Exercise4_01
        self.data = tf.random.normal((32,8))
        
    def testDataSize(self):
        output = self.exercise.data.shape
        expected_output = self.data.shape
        self.assertAllEqual(expected_output, output)

    def testPredictionSize(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(8,), name='Input_layer'))
        model.add(tf.keras.layers.Dense(4, activation='relu', name='Hidden_layer'))
        model.add(tf.keras.layers.Dense(1, name='Output_layer'))
        prediction = model.predict(self.data)
                
        output = self.exercise.prediction.shape
        expected_output = prediction.shape
        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()