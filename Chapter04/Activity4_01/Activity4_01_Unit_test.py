import import_ipynb
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import random_seed
from sklearn.preprocessing import StandardScaler
random_seed.set_seed(42)


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Activity4_01
        super(DataTest, self).setUp()
        self.activity = Activity4_01
        self.df = pd.read_csv('../Datasets/superconductivity.csv')
        self.df.dropna(inplace=True)
        
    def testTargetData(self):
        target = self.df['critical_temp']
        output = self.activity.target
        expected_output = target
        self.assertAllEqual(expected_output, output)

    def testFeatureData(self):
        features = self.df.drop('critical_temp', axis=1)
        scaler = StandardScaler()
        feature_array = scaler.fit_transform(features)
        features = pd.DataFrame(feature_array, columns=features.columns)
        
        output = self.activity.features
        expected_output = features
        self.assertAllEqual(expected_output, output)
    

if __name__ == '__main__':
    tf.test.main()