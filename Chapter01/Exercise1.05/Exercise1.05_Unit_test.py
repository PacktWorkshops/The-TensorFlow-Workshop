import import_ipynb
import tensorflow as tf


class MultiplicationTest(tf.test.TestCase):

    def setUp(self):
        import Exercise1_05
        super(MultiplicationTest, self).setUp()
        self.exercise = Exercise1_05
        self.matrix1 = tf.Variable([[1,2,3,4], [5,6,7,8]])
        self.matrix2 = tf.Variable([[3,2,7,2], [6,8,2,4]])
        
    def testMultiplication1(self):
        matmul1 = tf.matmul(self.matrix1, tf.reshape(self.matrix2, [4,2]))
        output = self.exercise.matmul1.numpy()
        expected_output = matmul1.numpy()
        self.assertAllEqual(expected_output, output)

    def testMultiplication2(self):
        matmul2 = tf.matmul(tf.reshape(self.matrix1, [4, 2]), self.matrix2)
        output = self.exercise.matmul2.numpy()
        expected_output = matmul2.numpy()
        self.assertAllEqual(expected_output, output)

    def testMultiplication3(self):
        matmul3 = tf.matmul(self.matrix1, self.matrix2, transpose_a=True)
        output = self.exercise.matmul3.numpy()
        expected_output = matmul3.numpy()
        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()