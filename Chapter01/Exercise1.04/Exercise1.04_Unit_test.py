import import_ipynb
import tensorflow as tf


class ReshapingTest(tf.test.TestCase):

    def setUp(self):
        import Exercise1_04
        super(ReshapingTest, self).setUp()
        self.exercise = Exercise1_04
        self.matrix1 = tf.Variable([[1,2,3,4], [5,6,7,8]])
        
    def testReshape1(self):
        reshape1 = tf.reshape(self.matrix1, shape=[4,2])
        output = self.exercise.reshape1.numpy()
        expected_output = reshape1.numpy()
        self.assertAllEqual(expected_output, output)

    def testReshape2(self):
        reshape2 = tf.reshape(self.matrix1, shape=[1,8])
        output = self.exercise.reshape2.numpy()
        expected_output = reshape2.numpy()
        self.assertAllEqual(expected_output, output)

    def testReshape3(self):
        reshape3 = tf.reshape(self.matrix1, shape=[8,1])
        output = self.exercise.reshape3.numpy()
        expected_output = reshape3.numpy()
        self.assertAllEqual(expected_output, output)

    def testReshape4(self):
        reshape4 = tf.reshape(self.matrix1, shape=[2,2,2])
        output = self.exercise.reshape4.numpy()
        expected_output = reshape4.numpy()
        self.assertAllEqual(expected_output, output)

    def testTranspose1(self):
        transpose1 = tf.transpose(self.matrix1)
        output = self.exercise.transpose1.numpy()
        expected_output = transpose1.numpy()
        self.assertAllEqual(expected_output, output)

    def testTranspose2(self):
        transpose2 = tf.transpose(tf.reshape(self.matrix1, shape=[2,2,2]))
        output = self.exercise.transpose2.numpy()
        expected_output = transpose2.numpy()
        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()