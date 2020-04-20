import import_ipynb
import tensorflow as tf

class VersionTest(tf.test.TestCase):

    def setUp(self):
        super(VersionTest, self).setUp()
        self.version = tf.__version__

    def testVersion(self):
        output = self.version
        expected_output = '2.1.0'

        self.assertAllEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()
