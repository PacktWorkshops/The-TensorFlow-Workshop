import tensorflow as tf

class ExerciseTest(tf.test.TestCase):

    def setUp(self):
        import exercise7_05
        super(ExerciseTest, self).setUp()
        self.exercise = exercise7_05

    def testModelInput(self):
        self.assertEqual(self.exercise.model.input_shape, (None, 32, 32, 3))
        self.assertEqual(self.exercise.model.output_shape, (None, 10))

    def testModelSize(self):
        self.assertLen(self.exercise.model.layers, 9)

    def testLayer0(self):
        layer = self.exercise.model.layers[0]
        self.assertEqual(layer.input_shape[0], (None, 32, 32, 3))
        self.assertEqual(layer.output_shape[0], (None, 32, 32, 3))

    def testLayer1(self):
        layer = self.exercise.model.layers[1]
        self.assertEqual(layer.input_shape, (None, 32, 32, 3))
        self.assertEqual(layer.output_shape, (None, 15, 15, 32))

    def testLayer2(self):
        layer = self.exercise.model.layers[2]
        self.assertEqual(layer.input_shape, (None, 15, 15, 32))
        self.assertEqual(layer.output_shape, (None, 7, 7, 64))

    def testLayer3(self):
        layer = self.exercise.model.layers[3]
        self.assertEqual(layer.input_shape, (None, 7, 7, 64))
        self.assertEqual(layer.output_shape, (None, 3, 3, 128))

    def testLayer4(self):
        layer = self.exercise.model.layers[4]
        self.assertEqual(layer.input_shape, (None, 3, 3, 128))
        self.assertEqual(layer.output_shape, (None, 1152))

    def testLayer5(self):
        layer = self.exercise.model.layers[5]
        self.assertEqual(layer.input_shape, (None, 1152))
        self.assertEqual(layer.output_shape, (None, 1152))

    def testLayer6(self):
        layer = self.exercise.model.layers[6]
        self.assertEqual(layer.input_shape, (None, 1152))
        self.assertEqual(layer.output_shape, (None, 1024))

    def testLayer7(self):
        layer = self.exercise.model.layers[7]
        self.assertEqual(layer.input_shape, (None, 1024))
        self.assertEqual(layer.output_shape, (None, 1024))

    def testLayer8(self):
        layer = self.exercise.model.layers[8]
        self.assertEqual(layer.input_shape, (None, 1024))
        self.assertEqual(layer.output_shape, (None, 10))

    def testXTrain(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.assertAllEqual(x_train / 255., self.exercise.x_train)

    def testYTrain(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.assertAllEqual(y_train.flatten(), self.exercise.y_train)

    def testXTest(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.assertAllEqual(x_test / 255., self.exercise.x_test)

    def testYTest(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.assertAllEqual(y_train.flatten(), self.exercise.y_train)

if __name__ == "__main__":
    tf.test.main()
