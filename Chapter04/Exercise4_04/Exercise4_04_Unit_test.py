import import_ipynb
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import random_seed
from sklearn.preprocessing import StandardScaler
random_seed.set_seed(42)


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise4_04
        super(DataTest, self).setUp()
        self.exercise = Exercise4_04
        self.df = pd.read_csv('../Datasets/qsar_androgen_receptor.csv', sep=';')
        self.df.dropna(inplace=True)
        
    def testTargetData(self):
        target = self.df['positive'].apply(lambda x: 1 if x=='positive' else 0)
        output = self.exercise.target
        expected_output = target
        self.assertAllEqual(expected_output, output)

    def testFeatureData(self):
        features = self.df.drop('positive', axis=1)
        
        output = self.exercise.features
        expected_output = features
        self.assertAllEqual(expected_output, output)
    

if __name__ == '__main__':
    tf.test.main()