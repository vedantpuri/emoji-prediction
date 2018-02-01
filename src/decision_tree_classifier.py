from sklearn import tree
from DataReader import DataReader
import numpy as np
import file_paths


def populate_features(tweet_file, labels_file):
    data_reader = DataReader(tweet_file, labels_file)
    return data_reader.get_features()


def split_train(X, Y, split_point):
    trainX = X[0:split_point]
    trainX = np.asarray(trainX)

    trainY = Y[0:split_point]
    trainY = np.asarray(trainY)

    return trainX, trainY


def split_test(X, Y, begin, end):
    testX = X[begin:end]
    testX = np.asarray(testX)

    testY = Y[begin:end]
    testY = np.asarray(testY)

    return testX, testY


def evaluate_accuracy(clf, testX, testY, testing_trange):
    counter = 0
    total = 0
    for i in range(testing_trange):
        output = clf.predict(np.asarray([testX[i]]))
        if output == testY[i]:
            counter += 1
        total += 1
    return counter / total


def run_dtc(tweet_file, labels_file):
    X, Y = populate_features(tweet_file, labels_file)
    trainX, trainY = split_train(X, Y, 800)
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainX, trainY)
    testX, testY = split_test(X, Y, 800, 1000)
    return evaluate_accuracy(clf, testX, testY, 200)

accuracy = run_dtc(file_paths.us_tweets_path, file_paths.us_labels_path)
print(accuracy)


