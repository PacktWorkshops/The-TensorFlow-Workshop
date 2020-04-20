import import_ipynb
import tensorflow as tf


class ReshapingTest(tf.test.TestCase):

    def setUp(self):
        import Activity1_02
        super(ReshapingTest, self).setUp()
        self.activity = Activity1_02
        self.matrix1 = tf.Variable([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        
    def testReshape1(self):
        reshape1 = tf.reshape(self.matrix1, shape=[6,2])
        output = self.activity.reshape1.numpy()
        expected_output = reshape1.numpy()
        self.assertAllEqual(expected_output, output)

    def testReshape2(self):
        reshape2 = tf.reshape(self.matrix1, shape=[3,2,2])
        output = self.activity.reshape2.numpy()
        expected_output = reshape2.numpy()
        self.assertAllEqual(expected_output, output)


    def testTranspose1(self):
        transpose1 = tf.transpose(self.matrix1)
        output = self.activity.transpose1.numpy()
        expected_output = transpose1.numpy()
        self.assertAllEqual(expected_output, output)

    def testTranspose2(self):
        transpose2 = tf.transpose(tf.reshape(self.matrix1, shape=[3,2,2]))
        output = self.activity.transpose2.numpy()
        expected_output = transpose2.numpy()
        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()