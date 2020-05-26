#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import import_ipynb
import tensorflow as tf

@tf.function
def my_matmult_func(x, y):
    result = tf.matmul(x, y)
    return result

class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise3_01
        super(DataTest, self).setUp()
        self.exercise = Exercise3_01
        self.test_x = tf.random.uniform((3, 3))
        self.test_y = tf.random.uniform((3, 3))
        
        
    def testInputDataShapes(self):
        self.x = self.exercise.x
        self.y = self.exercise.y
        self.assertEqual(self.test_x.shape, self.x.shape)
        self.assertEqual(self.test_y.shape, self.y.shape)

    def testOutputDataShapes(self):
        z = self.exercise.z
        expected_z_output = my_matmult_func(self.test_x, self.test_y)
        self.assertEqual(expected_z_output.shape, z.shape)

   
if __name__ == '__main__':
    tf.test.main()