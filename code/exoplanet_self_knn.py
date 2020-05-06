import numpy as np
from data_preprocessing import DataPreProcessing
from sklearn.metrics import confusion_matrix, classification_report


class KNearestNeighbor:
    def __init__(self, k=5):
        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None
        self.k = k

    def load_data(self):
        dp = DataPreProcessing(training_planets=150)
        dp.create_data()

        self.X_train, self.Y_train = dp.get_normalized_training_data()
        self.X_test, self.Y_test = dp.get_normalized_testing_data()

    def predict(self, X):
        predictions = []
        index = 1
        for x in X:
            # predict each piece of data in the given set X
            predictions.append(self.predict_helper(x, index))
            index += 1
        return predictions

    def predict_helper(self, x, index):
        # get distance from x to each training point
        distances = [(self.distance(x, self.X_train[i]), self.Y_train[i]) for i in range(len(self.X_train))]

        # use bubble sort to sort the list of distances
        size = len(distances)
        for i in range(size):
            for j in range(size - i - 1):
                if distances[j][0] > distances[j+1][0]:
                    self.swap(distances, j, j+1)
        a = b = 0
        # from 0 to k find the label with the most occurrences
        for i in range(self.k):
            if distances[i][1] == 2.0: a+=1
            else: b+=1

        return 2.0 if a > b else 1.0

    def report(self):
        predictions = self.predict(self.X_test)
        print("\nSelf K Nearest Neighbors accuracy: " + str(round(classification_report(self.Y_test, predictions, output_dict=True)['weighted avg']['f1-score'], 2) * 100) + "%")
        print("Confusion Matrix: ")
        print(confusion_matrix(self.Y_test, predictions))


    def distance(self, x, y):
        # get euclidean distance from x to y in any dimension
        d = 0
        for i in range(len(x)): d += (x[i] - y[i]) ** 2
        return np.sqrt(d)

    def swap(self, arr, i, j):
        tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp


if __name__ == '__main__':
    knn = KNearestNeighbor(k=7)
    assert knn.distance((0, 0), (1, 1)) == np.sqrt(2)
    assert knn.distance((0, 0, 0), (1, 1, 1)) == np.sqrt(3)

    print("passed all basic tests...")

    knn.load_data()
    knn.report()


