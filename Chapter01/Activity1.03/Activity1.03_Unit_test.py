import import_ipynb
import tensorflow as tf


class MultiplicationTest(tf.test.TestCase):

    def setUp(self):
        import Activity1_03
        super(MultiplicationTest, self).setUp()
        self.activity = Activity1_03
        
    def testResult(self):
        input1 = tf.Variable([[-0.013, 0.024, 0.06, 0.022], 
                       [0.001, -0.047, 0.039, 0.016],
                       [0.018, 0.030, -0.021, -0.028]], tf.float32)
        weights = tf.Variable([[19995.95], [24995.50], [36745.50], [29995.95]], tf.float32)
        bias = tf.Variable([[-2500.0],[-2500.0],[-2500.0]], tf.float32)
        output = self.activity.output.numpy()
        expected_output = (tf.matmul(input1,weights) + bias).numpy()
        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()