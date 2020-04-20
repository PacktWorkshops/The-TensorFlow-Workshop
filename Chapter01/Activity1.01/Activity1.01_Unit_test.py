import import_ipynb
import tensorflow as tf


class AdditionTest(tf.test.TestCase):

    def setUp(self):
        import Activity1_01
        super(AdditionTest, self).setUp()
        self.activity = Activity1_01

    def testIntSum(self):
        var1 = tf.Variable(26, tf.int32)
        var2 = tf.Variable(16, tf.int32)
        var_sum = var1 + var2
        output = self.activity.var_sum.numpy()
        expected_output = var_sum.numpy()
        self.assertAllEqual(expected_output, output)

    def testVectorScalarSum(self):
        scalar1 = tf.Variable(26.2, tf.float32)
        vector1 = tf.Variable([7.2, 1.1], tf.float32)
        vector_scalar_sum = scalar1 + vector1
        output = self.activity.vector_scalar_sum.numpy()
        expected_output = vector_scalar_sum.numpy()
        self.assertAllEqual(expected_output, output)

    def testMatrixSum(self):
        matrix1 = tf.Variable([[10, -21, 18], [12, 28, -9], [-2, 20, 8]], tf.int32)
        matrix2 = tf.Variable([[-2, 17, 2], [0, 12, 7], [10, 10, -4]], tf.int32)
        matrix_sum = matrix1 + matrix2
        output = self.activity.matrix_sum.numpy()
        expected_output = matrix_sum.numpy()
        self.assertAllEqual(expected_output, output)


if __name__ == '__main__':
    tf.test.main()