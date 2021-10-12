#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import import_ipynb
import tensorflow as tf
import tensorflow_hub as hub


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise3_03
        super(DataTest, self).setUp()
        self.exercise = Exercise3_03
        module = hub.load('https://tfhub.dev/google/imagenet/inception_v3/classification/5')
        self.model = module.signatures['default']
        
        
    def testModel(self):
        test_model_variables = self.model.variables[0].numpy()
        model_variables = self.exercise.model.variables[0].numpy()
        self.assertAllEqual(test_model_variables, model_variables)

   
if __name__ == '__main__':
    tf.test.main()