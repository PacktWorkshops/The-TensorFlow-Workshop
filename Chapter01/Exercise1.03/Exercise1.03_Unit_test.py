import import_ipynb
import tensorflow as tf


class AdditionTest(tf.test.TestCase):

    def setUp(self):
        import Exercise1_03
        super(AdditionTest, self).setUp()
        self.exercise = Exercise1_03

    def testIntSum(self):
        int1 = tf.Variable(20, tf.int32)
        int2 = tf.Variable(30, tf.int32)
        int_sum = int1 + int2
        output = self.exercise.int_sum.numpy()
        expected_output = int_sum.numpy()
        self.assertAllEqual(expected_output, output)

    def testVectorSum(self):
        vec1 = tf.Variable([20, 10], tf.int32)
        vec2 = tf.Variable([8, 4], tf.int32)
        vec_sum = vec1 + vec2
        output = self.exercise.vec_sum.numpy()
        expected_output = vec_sum.numpy()
        self.assertAllEqual(expected_output, output)

    def testMatrixSum(self):
        matrix1 = tf.Variable([[12, 18],[22,13]], tf.int32)
        matrix2 = tf.Variable([[20, 7],[31,48]], tf.int32)
        matrix_sum = matrix1 + matrix2
        output = self.exercise.matrix_sum.numpy()
        expected_output = matrix_sum.numpy()
        self.assertAllEqual(expected_output, output)


if __name__ == '__main__':
    tf.test.main()