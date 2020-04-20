import import_ipynb
import tensorflow as tf


class MultiplicationTest(tf.test.TestCase):

    def setUp(self):
        import Activity1_03
        super(MultiplicationTest, self).setUp()
        self.activity = Activity1_03
        
    def testMultiplication1(self):
        matrix1 = tf.Variable([[1,2,3], [4,5,6], [7,8,9]])
        matrix2 = tf.Variable([[6,2,6], [-2,4,0], [-4,2,-8]])
        matmul1 = tf.matmul(matrix1, matrix2)
        output = self.activity.matmul1.numpy()
        expected_output = matmul1.numpy()
        self.assertAllEqual(expected_output, output)

    def testMultiplication2(self):
        tensor1 = tf.Variable([[[2,1], [2, 1]], [[2,1],[2,1]], [[2,1],[2,1]]])
        tensor2 = tf.Variable([[[-3,7], [4, 5]], [[-10,2],[-3,3]], [[2,2],[0,-1]]])
        matmul2 = tf.matmul(tensor1, tensor2)
        output = self.activity.matmul2.numpy()
        expected_output = matmul2.numpy()
        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()