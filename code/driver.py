from exoplanet_knn import NearestNeighbor
from exoplanet_nn import NeuralNetwork
from exoplanet_tree import DecisionTree
from tensorflow_network import KerasModel
import os

if __name__ == '__main__':
    knn = NearestNeighbor()
    nn = NeuralNetwork()
    tree = DecisionTree()
    km = KerasModel()

    print("loading data...")
    knn.load_data()
    nn.load_data()
    tree.load_data()
    km.load_data()

    ans = input("reset models? Y/N: ")
    if ans == "Y" or ans == "y":
        nn.delete_nn()
        knn.delete_knn()
        tree.delete_tree()

    print("training km...")
    km.build_model()
    km.train()

    if os.path.exists(nn.file):
        print("loading nn...")
        nn.load_nn()
    else:
        print("training nn...")
        nn.train()

    if os.path.exists(knn.file):
        print("loading knn...")
        knn.load_knn()
    else:
        print("creating knn model...")
        knn.train()

    if os.path.exists(tree.file):
        print("loading tree...")
        tree.load_tree()
    else:
        print("creating tree...")
        tree.train()

    print("predicting testing data on all methods...")
    knn.predict()
    nn.predict()
    tree.predict()
    km.predict()

    print("\nAccuracy's: ")

    print("\nKeras Model accuracys: ")
    print("precision: " + str(round(km.precision, 2) * 100) + "%")
    print("recall: " + str(round(km.recall, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(km.matrix)

    print("\nNeural Network accuracys: ")
    print("precision: " + str(round(nn.precision, 2) * 100) + "%")
    print("recall: " + str(round(nn.recall, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(nn.matrix)

    print("\nK Nearest Neighbors accuracys: ")
    print("precision: " + str(round(knn.precision, 2) * 100) + "%")
    print("recall: " + str(round(knn.recall, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(knn.matrix)

    print("\nDecision Tree accuracys: ")
    print("precision: " + str(round(tree.precision, 2) * 100) + "%")
    print("recall: " + str(round(tree.recall, 2) * 100) + "%")
    print("Confusion Matrix: ")
    print(tree.matrix)

    ans = input("save files? Y/N: ")
    if ans == "Y" or ans == "y":
        print("saving models...")
        nn.save_nn()
        knn.save_knn()
        tree.save_tree()
