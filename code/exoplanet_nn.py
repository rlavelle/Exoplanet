from sklearn.neural_network import MLPClassifier
from data_preprocessing import DataPreProcessing
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.externals import joblib
import os


class NeuralNetwork:
    def __init__(self):
        '''
        First entry is the type of optimizer adam refers to a gradient based optimizer
        adam works well with large data sets

        Second entry is the type of activation function, logistic refers to a sigmoid function
        where this sigmoid function will be f(x)=1/(1+e^(-x))

        Third entry is the size of the hidden layers, this means the model will have 4
        hidden layers, each of size 800 which is the same size as our data's attributes

        Fourth entry is the maximum number of iteration, meaning the classifier will either iterate
        200 times, then stop, or it will find a convergence point

        Fifth entry is the learning rate for the classifier. the learning rate of the network
        controls the step size when updating the weights

        the Sixth entry is the batch size for the network. the batch size is the amount of samples
        that will be fed through the network

        '''
        self.label = ['adam', 'relu', (800, 800), 200, 0.001, 32]
        self.file = "../models/load_nn.pkl"

        # create the multi-layer perceptron classifier
        self.clf = MLPClassifier(solver=self.label[0],
                                 activation=self.label[1],
                                 hidden_layer_sizes=self.label[2],
                                 max_iter=self.label[3],
                                 learning_rate_init=self.label[4],
                                 batch_size=self.label[5])

        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None

        self.accuracy = None
        self.matrix = None
        self.precision = None
        self.recall = None

    def load_data(self):
        dp = DataPreProcessing(training_planets=5087)
        dp.create_data()

        self.X_train, self.Y_train = dp.get_scaled_standardized_training_data()
        self.X_test, self.Y_test = dp.get_scaled_standardized_testing_data()

    def train(self):
        self.clf.fit(X=self.X_train, y=self.Y_train)

    def predict(self):
        predictions = self.clf.predict(self.X_test)
        self.accuracy = classification_report(self.Y_test, predictions, output_dict=True)['weighted avg']['f1-score']
        self.matrix = confusion_matrix(self.Y_test, predictions)
        self.precision = precision_score(self.Y_test, predictions)
        self.recall = recall_score(self.Y_test, predictions)

    def predict_planet(self, planet_number):
        return self.clf.predict([self.X_test[planet_number]]), self.Y_test[planet_number]

    def load_nn(self):
        self.clf = joblib.load(self.file)

    def save_nn(self):
        joblib.dump(self.clf, self.file)

    def delete_nn(self):
        if os.path.exists(self.file): os.remove(self.file)

    def report(self):
        predictions = self.clf.predict(self.X_test)
        print(confusion_matrix(self.Y_test, predictions))
        print(classification_report(self.Y_test, predictions))


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.load_data()
    nn.train()
    nn.report()
