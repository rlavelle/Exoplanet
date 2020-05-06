from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import DataPreProcessing
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.externals import joblib
import os


class DecisionTree:
    def __init__(self):
        self.tree = DecisionTreeClassifier()
        self.file = "../models/load_tree.pkl"

        self.X_train = self.X_test = None
        self.Y_train = self.Y_test = None

        self.accuracy = None
        self.matrix = None
        self.precision = None
        self.recall = None

    def load_data(self):
        # 2500 is the sweet spot for number of training planets gives best results
        dp = DataPreProcessing(training_planets=5087)
        dp.create_data()

        self.X_train, self.Y_train, self.X_test, self.Y_test = dp.get_data()

    def train(self):
        self.tree.fit(self.X_train, self.Y_train)

    def predict(self):
        predictions = self.tree.predict(self.X_test)
        self.accuracy = classification_report(self.Y_test, predictions, output_dict=True)['weighted avg']['f1-score']
        self.matrix = confusion_matrix(self.Y_test, predictions)
        self.precision = precision_score(self.Y_test, predictions)
        self.recall = recall_score(self.Y_test, predictions)

    def predict_planet(self, planet_number):
        return self.tree.predict([self.X_test[planet_number]]), self.Y_test[planet_number]

    def load_tree(self):
        self.tree = joblib.load(self.file)

    def save_tree(self):
        joblib.dump(self.tree, self.file)

    def delete_tree(self):
        if os.path.exists(self.file): os.remove(self.file)

    def report(self):
        predictions = self.tree.predict(self.X_test)
        print(confusion_matrix(self.Y_test, predictions))
        print(classification_report(self.Y_test, predictions))


if __name__ == '__main__':
    tree = DecisionTree()
    tree.load_data()
    tree.train()
    tree.report()
