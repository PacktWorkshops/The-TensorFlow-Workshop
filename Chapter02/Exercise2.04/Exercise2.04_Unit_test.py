import import_ipynb
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import random_seed
random_seed.set_seed(42)

def prep_ds(ds, shuffle_buffer_size=1024, batch_size=32):
    # Shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat the dataset
    ds = ds.repeat()
    # Batch the dataset
    ds = ds.batch(batch_size)
    
    return ds

class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise2_04
        super(DataTest, self).setUp()
        self.exercise = Exercise2_04
        self.data = tf.data.experimental.make_csv_dataset('../Datasets/drugsComTrain_raw.tsv', batch_size=1, field_delim='\t')
        
    def testData(self):
        output = self.exercise.df.element_spec
        expected_output = self.data.element_spec
        self.assertDictEqual(expected_output, output)

if __name__ == '__main__':
    tf.test.main()