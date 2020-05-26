#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import import_ipynb
import tensorflow as tf


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Activity3_01
        super(DataTest, self).setUp()
        self.activity = Activity3_01
        self.test_x = tf.random.uniform((5, 5, 5))
        self.test_y = tf.random.uniform((5, 5, 5))
        
        
    def testInputDataShapes(self):
        self.x = self.activity.x
        self.y = self.activity.y
        self.assertEqual(self.test_x.shape, self.x.shape)
        self.assertEqual(self.test_y.shape, self.y.shape)

   
if __name__ == '__main__':
    tf.test.main()