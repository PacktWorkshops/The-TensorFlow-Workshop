import import_ipynb
import tensorflow as tf
import pandas as pd
from tensorflow.python.framework import random_seed
from sklearn.preprocessing import MinMaxScaler
random_seed.set_seed(42)


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise4_02
        super(DataTest, self).setUp()
        self.exercise = Exercise4_02
        self.df = pd.read_csv('../Datasets/Bias_correction_ucl.csv')
        self.df.drop('Date', inplace=True, axis=1)
        self.df.dropna(inplace=True)
        
    def testTargetData(self):
        target = self.df[['Next_Tmax', 'Next_Tmin']]
        output = self.exercise.target
        expected_output = target
        self.assertAllEqual(expected_output, output)

    def testFeatureData(self):
        features = self.df.drop(['Next_Tmax', 'Next_Tmin'], axis=1)
        scaler = MinMaxScaler()
        feature_array = scaler.fit_transform(features)
        features = pd.DataFrame(feature_array, columns=features.columns)
        
        output = self.exercise.features
        expected_output = features
        self.assertAllEqual(expected_output, output)
    

if __name__ == '__main__':
    tf.test.main()