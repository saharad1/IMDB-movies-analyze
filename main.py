import sys
from algorithm_runner import AlgorithmRunner
from data import Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid


def main(argv):
    data = Data('movie_metadata.csv')
    data_knn = Data(None)
    data_knn.file = data.file
    data_rocchio = Data(None)
    data_rocchio.file = data.file

    # Question 1 *****************************************
    print("Question 1:")
    data.preprocess()
    knn = AlgorithmRunner(KNeighborsClassifier(10), data.file, data.labels)
    rocchio = AlgorithmRunner(NearestCentroid(), data.file, data.labels)
    knn_results = knn.run(data.split_to_k_folds())
    rocchio_results = rocchio.run(data.split_to_k_folds())
    print("KNN classifier: {}, {}, {}".format(knn_results[0], knn_results[1], knn_results[2]))
    print("Rocchio classifier: {}, {}, {}".format(rocchio_results[0], rocchio_results[1], rocchio_results[2]))
    # ****************************************************

    # Question 2 *********************************************
    data_knn.preprocess(knn_flag=True)
    print("")
    print("Question 2:")
    knn = AlgorithmRunner(KNeighborsClassifier(13), data_knn.file, data_knn.labels)
    knn_results = knn.run(data_knn.split_to_k_folds())
    print("KNN classifier: {}".format(knn_results[2]))

    data_rocchio.preprocess(rocchio_flag=True)
    rocchio = AlgorithmRunner(NearestCentroid(), data_rocchio.file, data_rocchio.labels)
    rocchio_results = rocchio.run(data_rocchio.split_to_k_folds())
    print("Rocchio classifier: {}".format(rocchio_results[2]))
    # ***********************************************************


if __name__ == "__main__":
    main(sys.argv)
