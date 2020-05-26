#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import import_ipynb
import tensorflow as tf
import tensorflow_hub as hub


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Activity3_02
        super(DataTest, self).setUp()
        self.activity = Activity3_02
        module_handle = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.hub_layer = hub.KerasLayer(module_handle, input_shape=[], 
                           dtype=tf.string)
        
        
    def testModel(self):
        test_model_variables = self.hub_layer.variables[0].numpy()
        model_variables = self.activity.hub_layer.variables[0].numpy()
        self.assertAllEqual(test_model_variables, model_variables)

   
if __name__ == '__main__':
    tf.test.main()