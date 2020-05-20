import tensorflow as tf

class ExerciseTest(tf.test.TestCase):

    def setUp(self):
        import exercise7_02
        super(ExerciseTest, self).setUp()
        self.exercise = exercise7_02

    def testVersion(self):
        self.assertEqual(tf.__version__, "2.2.0")

    def testModelInput(self):
        # test input shape for the model
        model = self.exercise.our_model
        self.assertEqual(model.input_shape, (None, 300, 300, 3))

        # test input shape for first layer
        layer0 = model.layers[0]
        self.assertEqual(layer0.input_shape, (None, 300, 300, 3))

        # test input shape for second layer
        layer1 = model.layers[1]
        self.assertEqual(layer1.input_shape, (None, 298, 298, 16))


if __name__ == "__main__":
    tf.test.main()

