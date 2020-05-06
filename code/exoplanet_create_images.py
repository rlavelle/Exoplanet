import matplotlib.pyplot as plt
from data_preprocessing import DataPreProcessing
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
'''
File to create 2 folders of images called testimgs and trainimgs The images are created with the given raw data. 
The image names take the form 'planet#(n).png' -- where # is the planet number in the data set, and n is 
the label, where 1 means it is _NOT_ an planet and 2 means it _IS_ an planet
'''


def foo(integer):
    return 'FLUX.' + str(integer)


def create_images(file, folder):
    # reading in testing data, each planet is a row
    planets = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: planets.append(row)

    # will create a folder with 570 images
    index = 0
    for planet in planets:
        points = []
        for i in range(1, 3198): points.append(float(planet[foo(i)]))
        index += 1
        t = np.arange(0, 3197, 1)
        plt.plot(t, points)
        plt.xlabel('time')
        plt.ylabel('light fluctuation')
        plt.savefig(folder + str(index) + '(' + planet['LABEL'] + ')')
        plt.close()


def graph():
    dp = DataPreProcessing(training_planets=1000)
    dp.create_data()
    X, Y = dp.get_graphing_data()
    x = [planet[0] for planet in X]
    y = [planet[1] for planet in X]
    z = [planet[2] for planet in X]

    #fig = plt.figure()
    #ax = Axes3D(fig)

    for i in range(len(Y)):
        if Y[i] == 1:
            plt.scatter(x[i], y[i], color='red')
        else:
            plt.scatter(x[i], y[i], color='blue')
    plt.show()


if __name__ == "__main__":
    graph()
    # uncomment below two lines only to create images (takes a while)
    # create_images('../data/exoTest.csv', '../testimgs/planet')
    # create_images('../data/exoTrain.csv', '../trainimgs/planet')

