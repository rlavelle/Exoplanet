from sklearn.preprocessing import Normalizer, StandardScaler, normalize
from collections import Counter
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter
from scipy import fft
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

'''
A full set of training data should have:
rows = 5087
cols = 3197

A full set of testing data should have:
rows = 570
cols = 3197

Change the values of rows and cols to determine how much of the data set you want to include
edit the number of rows to determine how many planets you want in the data set, deletes from the bottom up
edit the number of cols to determine how much light fluctuation you want in the data set, deletes from right to left
'''


class DataPreProcessing:
    def __init__(self, training_planets=5087, smote=True):
        self.ROWS = training_planets
        self.COLS = 3197
        self.TEST_ROWS = 570
        self.url_training = '../data/exoTrain.csv'
        self.url_test = '../data/exoTest.csv'
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.smote = smote

    def create_data(self):
        # read data set in as a dataFrame
        data_frame = pd.read_csv(self.url_training)
        data_frame2 = pd.read_csv(self.url_test)

        # get numpy representation of dataFrame
        array = data_frame.values
        array2 = data_frame2.values

        '''           
        array takes the form [  [LABEL_1 v_1 v_2 .... v_n]{i=0}
                                [LABEL_2 v_1 v_2 .... v_n]{i=1}
                                [LABEL_3 v_1 v_2 .... v_n]{i=2}
                                                 ....
                                [LABEL_n v_1 v_2 .... v_n]{i=ROWS}  ]
                                
        creates X as the input data for our program
        creates Y as the expected output labels for out program

        X has length COL, and ROW entries
        Y has length 1,   and ROW entries 
        '''
        self.X_train = [array[i][1:self.COLS + 1] for i in range(0, self.ROWS)]
        self.Y_train = [array[i][0] for i in range(0, self.ROWS)]

        self.X_test = [array2[i][1:self.COLS + 1] for i in range(0, self.TEST_ROWS)]
        self.Y_test = [array2[i][0] for i in range(0, self.TEST_ROWS)]

        '''
         recreates X and Y using SMOTE, to create new fake exo-planets to balance the data set
         choosing random_state=42 allows us to use SMOTE to make the data set 1:1
         for rows = 100 it adds 26 exo-planets to balance to data set to 63:63
        '''
        if self.smote: self.X_train, self.Y_train = SMOTE(random_state=42).fit_resample(self.X_train, self.Y_train)

    def get_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def get_normalized_testing_data(self):
        # fits the model with X and scales each non zero row of X to unit form
        X_rescaled = Normalizer().fit(self.X_test).transform(self.X_test)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_test

    def get_scaled_standardized_testing_data(self):
        # apply a fourier transformation to each row of data
        X_rescaled = [np.abs(fft(x, len(x))) for x in self.X_test]

        # cut attributes of planets in half, since fourier transformation is symmetric
        X_rescaled = [X_rescaled[i][0:len(X_rescaled[i])//2] for i in range(len(X_rescaled))]

        # normalize the data
        X_rescaled = normalize(X_rescaled)

        # apply a gaussian filter to smooth out the data to make it have less noise
        X_rescaled = gaussian_filter(X_rescaled, sigma=10)

        # use Standard Scaler to standardize the data with a gaussian norm
        X_rescaled = StandardScaler().fit_transform(X_rescaled)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_test

    def get_pca_testing_data(self):
        # fits the model with X and apply's the dimensionality reduction on X
        X_rescaled = PCA(n_components=570, svd_solver='full').fit_transform(self.X_test)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_test

    def get_normalized_training_data(self):
        # fits the model with X and scales each non zero row of X to unit form
        X_rescaled = Normalizer().fit(self.X_train).transform(self.X_train)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_train

    def get_scaled_standardized_training_data(self):
        # apply a fourier transformation to each row of data
        X_rescaled = [np.abs(fft(x, len(x))) for x in self.X_train]

        # cut attributes of planets in half, since fourier transformation is symmetric
        X_rescaled = [X_rescaled[i][0:len(X_rescaled[i])//2] for i in range(len(X_rescaled))]

        # normalize the data
        X_rescaled = normalize(X_rescaled)

        # apply a gaussian filter to smooth out the data to make it have less noise
        X_rescaled = gaussian_filter(X_rescaled, sigma=10)

        # use Standard Scaler to standardize the data with a gaussian norm
        X_rescaled = StandardScaler().fit_transform(X_rescaled)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_train

    def get_pca_training_data(self):
        # fits the model with X and apply's the dimensionality reduction on X
        X_rescaled = PCA(n_components=570, svd_solver='full').fit_transform(self.X_train)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_train

    def get_graphing_data(self):
        # fits the model with X and apply's the dimensionality reduction on X
        X_rescaled = PCA(n_components=3, svd_solver='full').fit_transform(self.X_train)

        # tuple of (inputs, outputs)
        return X_rescaled, self.Y_train


if __name__ == '__main__':
    ''' 
    Some examples on how to use the given class to process data
    and then get access to it, the variables data represent (X, Y)
    
    where X is defined as [[x1 x2 x3 ...], [y1 y2 y3 ...], [z1 z2 z3 ...], ..., [n1 n2 n3 ...]] and
    where Y is defined as [a, a, a, a, ..., a, b, b, b, b, b, b]. each list in X maps to a label in Y
    meaning that...
    
    data[0][planet_number], the first [0] access's X and the second [planet_number] access's a list [x1 x2 x3 ...]
    data[1][planet_number][0], the [1] access's Y and the [planet_number] gets to [a] so the second [0] access's the 'a'
    
    the len of Y_train should always be the same length as the rows variable, because Y_train holds all of the
    planet or non-planet labels, and the rows variable dictates how many planets are in the data set
    
    the len of X_train will vary, for p1 since the Normalized function doesnt change the amount of attributes, the 
    length will always be the same length as the cols variable, since the cols variable dictates the amount of 
    attributes in the data sets. for p2 the attributes count is shrunk by PCA.
    '''

    planet_number = 3
    dp = DataPreProcessing(training_planets=570)
    dp.create_data()

    data1 = dp.get_normalized_training_data()
    data2 = dp.get_pca_training_data()
    data3 = dp.get_scaled_standardized_training_data()

    planet1 = data1[0][planet_number], data1[1][planet_number]
    planet2 = data2[0][planet_number], data2[1][planet_number]
    planet3 = data3[0][planet_number], data3[1][planet_number]

    print("number of labels: " + str(Counter(data3[1])) + " number of planets: " + str(len(dp.Y_train)))
    print("dim of p1 X: " + str(len(planet1[0])) + " Y: " + str(len(dp.Y_train)))
    print("dim of p2 X: " + str(len(planet2[0])) + " Y: " + str(len(dp.Y_train)))
    print("dim of p3 X: " + str(len(planet3[0])) + " Y: " + str(len(dp.Y_train)))

    def graph(planet, title):
        plt.title(title)
        plt.xlabel("time")
        plt.ylabel("light fluctuation")
        plt.plot(planet[0])
        plt.show()
        plt.close()

    # quick visual to see the graphs of pre-processing
    graph(planet1, "normalized")
    graph(planet2, "PCA")
    graph(planet3, "gaussian")

    # to show similarity between actual planets under the data prepossessing
    plt.title("planets")
    plt.xlabel("time")
    plt.ylabel("light fluctuation")
    for i in range(37):
        plt.plot(data3[0][i])
    plt.show()
    plt.close()
