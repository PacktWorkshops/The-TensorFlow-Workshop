import tensorflow as tf

class ExerciseTest(tf.test.TestCase):
    
    def setUp(self):
        import exercise7_01
        super(ExerciseTest, self).setUp()
        self.exercise = exercise7_01

    def testVersion(self):
        self.assertEqual(tf.__version__, "2.2.0")

    def testModelInput(self):
        model = self.exercise.our_first_model
        self.assertEqual(model.input_shape, (None, 300, 300, 3))
        layer0 = model.layers[0]
        self.assertEqual(layer0.input_shape, (None, 300, 300, 3))

if __name__ == "__main__":
    tf.test.main()
