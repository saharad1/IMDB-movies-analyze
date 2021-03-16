
def accuracy(label_test, predicted_labels):
    """
    Evaluates the effectiveness of the classifier
    :param label_test: The actual labels in the dataset
    :param predicted_labels: The predicted labels resulted from the classifier
    :return: Accuracy score
    """
    return sum([label_test[i] == predicted_labels[i] for i in range(len(label_test))]) / len(label_test)


def recall(label_test, predicted_labels):
    """
    Evaluates the effectiveness of the classifier
    :param label_test: The actual labels in the dataset
    :param predicted_labels: The predicted labels resulted from the classifier
    :return: Recall score
    """
    tp = sum([label_test[i] == 1 and predicted_labels[i] == 1 for i in range(len(label_test))])
    fn = sum([label_test[i] == 1 and predicted_labels[i] == 0 for i in range(len(label_test))])
    return tp / (tp + fn)


def precision(label_test, predicted_labels):
    """
    Evaluates the effectiveness of the classifier
    :param label_test: The actual labels in the dataset.
    :param predicted_labels: The predicted labels resulted from the classifier.
    :return: Precision score.
    """
    tp = sum([label_test[i] == 1 and predicted_labels[i] == 1 for i in range(len(label_test))])
    fp = sum([label_test[i] == 0 and predicted_labels[i] == 1 for i in range(len(label_test))])
    return tp / (tp + fp)


class AlgorithmRunner:

    def __init__(self, algorithm, data, labels):
        self.algorithm = algorithm
        self.data = data
        self.labels = labels

    def fit(self, data, label):
        """
        The training phase of the data
        :param data: To train
        :param label: Required labels
        """
        self.algorithm.fit(data, label)

    def predict(self, test):
        """
        The prediction phase of the data
        :param test: The part of the dataset to be classified
        """
        return self.algorithm.predict(test)

    def run(self, kf):
        """
        Manages the classification process and the effectiveness of evaluation
        :param kf: The divided folds of the dataset
        :return: The metrics of the classifier
        """
        accuracy_value = 0
        recall_value = 0
        precision_value = 0
        for train, test in kf:
            data_train = self.data.iloc[train, 0:]
            data_test = self.data.iloc[test, 0:]
            label_train = self.labels.iloc[train]
            label_test = list(self.labels.iloc[test])

            self.fit(data_train, label_train)
            predicted_labels = self.predict(data_test)
            accuracy_value += accuracy(label_test, predicted_labels)
            recall_value += recall(label_test, predicted_labels)
            precision_value += precision(label_test, predicted_labels)
        return precision_value / 5, recall_value / 5, accuracy_value / 5
