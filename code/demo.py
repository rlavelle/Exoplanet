from exoplanet_knn import NearestNeighbor
from exoplanet_nn import NeuralNetwork
from exoplanet_tree import DecisionTree
from exoplanet_self_knn import KNearestNeighbor
from data_preprocessing import DataPreProcessing
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    dp = DataPreProcessing(training_planets=5087)

    print("creating models...")
    dp.create_data()
    data = dp.X_test

    nn = NeuralNetwork()
    knn = NearestNeighbor()
    tree = DecisionTree()
    self_knn = KNearestNeighbor(k=7)

    nn.load_data()
    nn.load_nn()

    knn.load_data()
    knn.load_knn()

    tree.load_data()
    tree.load_tree()

    self_knn.load_data()

    knn.predict()
    nn.predict()
    tree.predict()

    while True:
        ans = input("\n(E)xit | (R)eport | (P)redict: ")

        if ans == "E" or ans == "e":
            break
        elif ans == "R" or ans == "r":
            print("\nNeural Network accuracy: " + str(round(nn.accuracy, 2) * 100) + "%")
            print("Confusion Matrix: ")
            print(nn.matrix)

            print("\nK Nearest Neighbors accuracy: " + str(round(knn.accuracy, 2) * 100) + "%")
            print("Confusion Matrix: ")
            print(knn.matrix)

            print("\nDecision Tree accuracy: " + str(round(tree.accuracy, 2) * 100) + "%")
            print("Confusion Matrix: ")
            print(tree.matrix)
        elif ans == "P" or ans == "p":
            planet_number = int(input("Enter planet number to test (0-570): "))

            nn_predict = nn.predict_planet(planet_number)
            knn_predict = knn.predict_planet(planet_number)
            tree_predict = tree.predict_planet(planet_number)
            self_knn_predict = self_knn.predict([self_knn.X_test[planet_number]])

            print("Actual: " + ("exoplanet\n" if nn_predict[1] == 2 else "not an exoplanet\n"))

            print("Neural Networks prediction: " + ("exoplanet" if nn_predict[0] == 2 else "not an exoplanet"))
            print("K Nearest Neighbors prediction: " + ("exoplanet" if knn_predict[0] == 2 else "not an exoplanet"))
            print("Trees prediction: " + ("exoplanet" if tree_predict[0] == 2 else "not an exoplanet"))
            print("self K Nearest Neighbors prediction: " + ("exoplanet\n" if self_knn_predict[0] == 2 else "not an exoplanet\n"))

            plt.title("Light Graph")
            plt.xlabel("time")
            plt.ylabel("light fluctuation")
            plt.plot(data[planet_number])
            plt.show()
        else:
            print(ans + " is no a valid input")
