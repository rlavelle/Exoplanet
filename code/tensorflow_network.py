from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tensorflow import keras
from data_preprocessing import DataPreProcessing


class KerasModel:
    def __init__(self):
        self.model = keras.Sequential()

        self.learning_rate = 0.001
        self.epochs = 50
        self.batch_size = 32

        self.layers = [
            {"units": 1, "input_dim": 1598, "activation": 'relu', "dropout": 0},
            {"units": 1, "input_dim": 1, "activation": 'sigmoid', "dropout": 0},
        ]

        self.X_test = self.Y_test = None
        self.X_train = self.Y_train = None

        self.precision = None
        self.recall = None
        self.matrix = None

    def load_data(self):
        dp = DataPreProcessing(training_planets=5087, smote=False)
        dp.create_data()

        self.X_train, self.Y_train = dp.get_scaled_standardized_training_data()
        self.X_test, self.Y_test = dp.get_scaled_standardized_testing_data()

        self.Y_train = [0.0 if pred == 1.0 else 1.0 for pred in self.Y_train]
        self.Y_test = [0.0 if pred == 1.0 else 1.0 for pred in self.Y_test]

    def build_model(self):
        for layer in self.layers:
            self.model.add(keras.layers.Dense(units=layer["units"], input_dim=layer["input_dim"]))
            self.model.add(keras.layers.Activation(activation=layer["activation"]))
            if layer["dropout"] > 0: self.model.add(keras.layers.Dropout(layer["dropout"]))

        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                           metrics=['accuracy'])

    def train(self):
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self):
        predictions = [round(pred[0]) for pred in self.model.predict(self.X_test, batch_size=32)]
        self.precision = precision_score(self.Y_test, predictions)
        self.recall = recall_score(self.Y_test, predictions)
        self.matrix = confusion_matrix(self.Y_test, predictions)


if __name__ == "__main__":
    km = KerasModel()

    print("loading data...")
    km.load_data()

    km.build_model()
    km.train()
    km.predict()

    print("\nKeras Model accuracys: ")
    print("\nprecision: " + str(round(km.precision, 2) * 100) + "%")
    print("recall: " + str(round(km.recall, 2) * 100) + "%")
    print("\nConfusion Matrix: ")
    print(km.matrix)
