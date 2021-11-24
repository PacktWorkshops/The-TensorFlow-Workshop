import os
import tensorflow as tf

class ExerciseTest(tf.test.TestCase):

    def setUp(self):
        import exercise7_04
        super(ExerciseTest, self).setUp()
        self.exercise = exercise7_04

    def testVersion(self):
        self.assertEqual(tf.__version__, "2.6.0")

    def testLayer0(self):
        model = self.exercise.our_model
        layer0 = model.layers[0]
        self.assertEqual(layer0.input_shape, (None, 300, 300, 3))
        self.assertEqual(layer0.output_shape, (None, 298, 298, 16))

    def testLayer1(self):
        model = self.exercise.our_model
        layer1 = model.layers[1]
        self.assertEqual(layer1.input_shape, (None, 298, 298, 16))
        self.assertEqual(layer1.output_shape, (None, 149, 149, 16))

    def testLayer2(self):
        model = self.exercise.our_model
        layer2 = model.layers[2]
        self.assertEqual(layer2.input_shape, (None, 149, 149, 16))
        self.assertEqual(layer2.output_shape, (None, 147, 147, 32))

    def testLayer3(self):
        model = self.exercise.our_model
        layer3 = model.layers[3]
        self.assertEqual(layer3.input_shape, (None, 147, 147, 32))
        self.assertEqual(layer3.output_shape, (None, 73, 73, 32))

    def testLayer4(self):
        model = self.exercise.our_model
        layer4 = model.layers[4]
        self.assertEqual(layer4.input_shape, (None, 73, 73, 32))
        self.assertEqual(layer4.output_shape, (None, 71, 71, 64))

    def testLayer5(self):
        model = self.exercise.our_model
        layer5 = model.layers[5]
        self.assertEqual(layer5.input_shape, (None, 71, 71, 64))
        self.assertEqual(layer5.output_shape, (None, 35, 35, 64))

    def testLayer6(self):
        model = self.exercise.our_model
        layer6 = model.layers[6]
        self.assertEqual(layer6.input_shape, (None, 35, 35, 64))
        self.assertEqual(layer6.output_shape, (None, 78400))

    def testLayer7(self):
        model = self.exercise.our_model
        layer7 = model.layers[7]
        self.assertEqual(layer7.input_shape, (None, 78400))
        self.assertEqual(layer7.output_shape, (None, 512))

    def testLayer8(self):
        model = self.exercise.our_model
        layer8 = model.layers[8]
        self.assertEqual(layer8.input_shape, (None, 512))
        self.assertEqual(layer8.output_shape, (None, 1))

    def testModelInput(self):
        model = self.exercise.our_model
        self.assertEqual(model.input_shape, (None, 300, 300, 3))
        self.assertEqual(model.output_shape, (None, 1))

    def testModelSize(self):
        model = self.exercise.our_model
        self.assertLen(model.layers, 9)

    def testNumberHumanTrainingImages(self):
        hmn_names = os.listdir(self.exercise.hmn_trn_dir)
        self.assertGreaterEqual(len(hmn_names), 500)

    def testNumberHorseTrainingImages(self):
        hrs_names = os.listdir(self.exercise.hrs_trn_dir)
        self.assertGreaterEqual(len(hrs_names), 500)

    def testNumberHumanValidationImages(self):
        hmn_val_names = os.listdir(self.exercise.hmn_val_dir)
        self.assertGreaterEqual(len(hmn_val_names), 128)

    def testNumberHorseValidationImages(self):
        hrs_val_names = os.listdir(self.exercise.hrs_val_dir)
        self.assertGreaterEqual(len(hrs_val_names), 128)

    def testHumanTrainingImageNames(self):
        hmn_names = os.listdir(self.exercise.hmn_trn_dir)
        for name in hmn_names:
            self.assertTrue(name.startswith("human"))

    def testHorseTrainingImageNames(self):
        hrs_names = os.listdir(self.exercise.hrs_trn_dir)
        for name in hrs_names:
            self.assertTrue(name.startswith("horse"))

    def testHumanValidationImageNames(self):
        hmn_val_names = os.listdir(self.exercise.hmn_val_dir)
        for name in hmn_val_names:
            self.assertTrue(name.startswith("valhuman"))

    def testHorseValidationImageNames(self):
        hrs_val_names = os.listdir(self.exercise.hrs_val_dir)
        for name in hrs_val_names:
            self.assertTrue(name.startswith("horse"))


if __name__ == "__main__":
    tf.test.main()
