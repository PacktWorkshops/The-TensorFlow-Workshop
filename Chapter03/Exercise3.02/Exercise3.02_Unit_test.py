#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import import_ipynb
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise3_02
        super(DataTest, self).setUp()
        self.exercise = Exercise3_02
        self.train_datagen = ImageDataGenerator(rescale = 1./255)
        batch_size = 25
        self.training_set = self.train_datagen.flow_from_directory(
            '../Datasets/image_data',
            target_size = (224, 224),
            batch_size = batch_size,
            class_mode = 'binary')
        
        
    def testInputDataShapes(self):
        self.test_batch = next(self.training_set)[0]
        self.batch = next(self.exercise.training_set)[0]
        self.assertEqual(self.test_batch.shape, self.batch.shape)

   
if __name__ == '__main__':
    tf.test.main()