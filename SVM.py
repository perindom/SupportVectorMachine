"""
Author: Dominick J Peirni
Completed: December 9th, 2020
Subject: Implementation of a support vector machine binary classifier, optimized with John Platt's SMO algorithm
"""
import datetime

import numpy as np
import pandas as pd
import random

# Housekeeping
ADULT_DATA_PATH = "/Users/dominickperini/Documents/2020_Fall/Machine_Learning/Project/adult.data"
pd.set_option('mode.chained_assignment', None)

# Program Argument: Change to True to use pretrained .npy files instead of training
PRETRAINED = True

# Global variables, hyper parameters
regularization_parameter = 2 ** 5
gamma = 2 ** (-3)
tolerance_parameter = -5
max_passes_parameter = 10


def kernel(x, z):
    # x vector
    # z vector
    # Gaussian Radial Basis Function
    kern = np.exp(- (np.linalg.norm(x - z) ** 2) / (2 * gamma ** 2))
    return kern


def classifier(alpha, dpi, features, labels, b):
    # alpha vector, lagrange multipliers
    # dpi scalar, data point index of what the classifier is outputting
    # features vector, each row is a datapoint and each column is a feature
    # labels vector, binary label for each datapoint
    classification = 0
    for i in range(len(alpha)):
        classification += alpha[i] * labels[i] * kernel(features[i], features[dpi])
    classification += b
    return classification


def SMO(C, tol, max_passes, features, labels):
    # C scalar, is the regularization parameter
    # tol scalar, is the numerical tolerance
    # max_passes scalar, is the maximum number of times to iterate over alpha values without changing any
    # features matrix, each row is a datapoint and each column is a feature
    # labels vector, binary label for each datapoint

    alpha = np.zeros(features.shape[0])
    b = 0
    x = features
    y = labels
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0

        for i in range(len(y)):
            E_i = classifier(alpha=alpha, dpi=i, features=features, labels=labels, b=b) - y[i]

            if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0):
                # Pick a random j != i
                j = random.randrange(i + 1, len(y) + i) % len(y)

                alpha_old = np.zeros(len(labels))
                alpha_old[i] = alpha[i]
                alpha_old[j] = alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = 2 * kernel(x[i], x[j]) - kernel(x[i], x[i]) - kernel(x[j], x[j])

                if eta >= 0:
                    continue

                E_j = classifier(alpha=alpha, dpi=j, features=features, labels=labels, b=b) - y[j]
                alpha_j_new_clipped = alpha[j] - y[j] * (E_i - E_j) / eta

                if alpha_j_new_clipped > H:
                    alpha[j] = H
                elif L <= alpha_j_new_clipped <= H:
                    alpha[j] = alpha_j_new_clipped
                elif alpha_j_new_clipped < L:
                    alpha[j] = L

                if np.abs(alpha[j] - alpha_old[j]) < 10 ** -5:
                    continue

                alpha[i] = alpha[i] + y[i] * y[j] * (alpha_old[j] - alpha[j])

                b_1 = b - E_i - y[i] * (alpha[i] - alpha_old[i]) * kernel(x[i], x[i]) \
                      - y[j] * (alpha[j] - alpha_old[j]) * kernel(x[i], x[j])

                b_2 = b - E_j - y[i] * (alpha[i] - alpha_old[i]) * kernel(x[i], x[j]) \
                      - y[j] * (alpha[j] - alpha_old[j]) * kernel(x[j], x[j])

                if 0 < alpha[i] < C:
                    b = b_1
                elif 0 < alpha[j] < C:
                    b = b_2
                else:
                    b = (b_1 + b_2) / 2
                num_changed_alphas += 1
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    return alpha, b


def read():
    # Import data into a pandas dataframe
    adult_data = pd.read_csv(ADULT_DATA_PATH, delimiter=',', skipinitialspace=True, na_values="?")
    return adult_data


def clean(data):
    # Drop rows with null values
    data = data.dropna()
    data = data.drop('EDUCATION', axis=1)
    data = data.drop('FNLWGT', axis=1)
    return data


def normalize(data):
    cols_to_enum = ['WORK_CLASS', 'MARITAL_STATUS', 'OCCUPATION', 'RELATIONSHIP', 'RACE', 'SEX', 'COO', 'LABEL']
    for col in cols_to_enum:
        mapping = {k: v for v, k in enumerate(data[col].unique())}
        data[col + '_ENUM'] = data[col].map(mapping)
    data = data.drop(cols_to_enum, axis=1)

    for col in data.columns:
        maximum = data[col].max()
        minimum = data[col].min()
        range_of_col = maximum - minimum
        data[col] = data[col].map(lambda p: (p - minimum) / range_of_col)
    return data


def trainTestSplit(data, train_frac):
    # Split the data into a training set and a testing set
    training_set = data.sample(frac=train_frac, random_state=0)
    testing_set = data.drop(training_set.index)
    return training_set, testing_set


def shuffleInUnison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def predictAndCalcError(features, labels, alpha, b):
    predictions = np.zeros(len(labels))
    for i in range(len(labels)):
        pred = classifier(alpha=alpha, dpi=i, features=features, labels=labels, b=b / 5)
        predictions[i] = 1 if pred > 0 else 0
    error = sum(1 for x, y in zip(predictions, labels) if x != y) / len(predictions)

    return error


if __name__ == '__main__':
    begin = datetime.datetime.now()
    raw_data = read()

    cleaned_data = clean(raw_data)

    normalized_data = normalize(cleaned_data)

    percent_train = 0.02
    train_data, test_data = trainTestSplit(normalized_data, percent_train)
    train_labels = train_data['LABEL_ENUM'].to_numpy()
    train_features = train_data.drop('LABEL_ENUM', axis=1).to_numpy()

    if not PRETRAINED:
        lagrange_trained, b_trained = SMO(regularization_parameter, tolerance_parameter, max_passes_parameter,
                                          train_features, train_labels)
        np.save("alpha_trained.npy", lagrange_trained)
        np.save("b_trained.npy", b_trained)
    else:
        lagrange_trained = np.load("alpha_trained.npy")
        b_trained = np.load("b_trained.npy")

    accuracy_over_training = 1 - predictAndCalcError(train_features, train_labels, lagrange_trained, b_trained)

    test_labels = test_data['LABEL_ENUM'].to_numpy()
    test_features = test_data.drop('LABEL_ENUM', axis=1).to_numpy()

    accuracy_over_testing = 1 - predictAndCalcError(test_features, test_labels, lagrange_trained, b_trained)
    print("With the following hyperparameters: ")
    print("Training set size: ", percent_train * len(normalized_data))
    print("C:", regularization_parameter)
    print("gamma:", gamma)
    print("tolerance:", tolerance_parameter)
    print("Max Passes: ", max_passes_parameter)
    print("Training Accuracy:", accuracy_over_training)
    print("Testing Accuracy:", accuracy_over_testing)

    end = datetime.datetime.now()
    print("Time to execute:", end - begin)
