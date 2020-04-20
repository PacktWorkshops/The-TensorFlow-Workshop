import import_ipynb
import tensorflow as tf

class TensorTest(tf.test.TestCase):

    def setUp(self):
        import Exercise1_02
        super(TensorTest, self).setUp()
        self.exercise = Exercise1_02

    def testScalar(self):
        output = self.exercise.int_variable
        expected_output = tf.Variable(123, tf.int16)
        self.assertAllEqual(expected_output, output)

    def testVector(self):
        output = tf.Variable([1.1, 2.2, 3.3], tf.float32).numpy()
        expected_output = tf.Variable([1.1, 2.2, 3.3], tf.float32).numpy()

        self.assertAllEqual(expected_output, output)

    def testMatrix(self):
        output = self.exercise.matrix_variable.numpy()
        expected_output = tf.Variable([[1,2,3], [4,5,6]], tf.int16).numpy()

        self.assertAllEqual(expected_output, output)

    def testTensor(self):
        output = self.exercise.tensor_variable.numpy()
        expected_output = tf.Variable([[[1, 2], [3,4]], [[5, 6], [7, 8]]]).numpy()

        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()
    
