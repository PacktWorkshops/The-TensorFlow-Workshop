import import_ipynb
import tensorflow as tf
import pandas as pd


class DataTest(tf.test.TestCase):

    def setUp(self):
        import Exercise2_02
        super(DataTest, self).setUp()
        self.exercise = Exercise2_02
        self.data = pd.read_csv('../Datasets/Bias_correction_ucl.csv')
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
    def testData(self):
        output = self.exercise.df
        expected_output = self.data
        pd.testing.assert_frame_equal(expected_output, output)

    def testProcessedData(self):
        year_dummies = pd.get_dummies(self.data['Date'].dt.year, prefix='year')
        month_dummies = pd.get_dummies(self.data['Date'].dt.month, prefix='month')
        df2 = pd.concat([self.data, month_dummies, year_dummies], axis=1)
        df2.drop('Date', axis=1, inplace=True)
        output = self.exercise.df2
        expected_output = df2
        pd.testing.assert_frame_equal(expected_output, output)


if __name__ == '__main__':
    tf.test.main()