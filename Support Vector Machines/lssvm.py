import csv

import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import copy


def read_data(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # take the header out
        for row in reader:  # each row is a list
            data.append(row)
    data = np.array(data, dtype=np.float)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def kernel_function(gamma, x_i, x_j):
    dist = np.linalg.norm(np.subtract(x_i, x_j))
    return np.exp(-1 * gamma * (dist ** 2))


def get_omega_matrix(gamma, X_train):
    n = len(X_train)
    omega_matrix = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            omega_matrix[i][j] = kernel_function(gamma, X_train[i], X_train[j])
    return omega_matrix


def lssvm_train(X_train, y_train, C, gamma):
    n = len(X_train)
    alphas = np.zeros((n, 1))
    unit = np.mat(np.ones((n, 1)))
    I = np.eye(n)
    y_train_mat = np.mat(y_train).T
    omega_matrix = get_omega_matrix(gamma, X_train)
    zero = np.mat(np.zeros((1, 1)))
    left_upper_matrix = np.hstack((zero, unit.T))
    m = omega_matrix + (I / float(C))
    left_lower_matrix = np.hstack((unit, omega_matrix + (I / float(C))))
    complete_matrix = np.vstack((left_upper_matrix, left_lower_matrix))
    right_matrix = np.vstack((zero, y_train_mat))
    b_alpa_vector = np.linalg.inv(complete_matrix) * right_matrix
    b = b_alpa_vector[0][0]
    for i in range(n):
        alphas[i] = b_alpa_vector[i + 1][0]
    return b, alphas


def lssvm_predict(alphas, b, X_test, X_train, y_train, gamma):
    y_predicted = np.zeros(len(X_test))
    b_copy = copy.deepcopy(b)
    for i in range(0, len(X_test)):
        sum = b_copy
        for j in range(0, len(alphas)):
            sum += alphas[j] * kernel_function(gamma, X_train[j], X_test[i])
        y_predicted[i] = sum
        b_copy = copy.deepcopy(b)
    return y_predicted


def run_lssvm_model(X_train, y_train, X_test, y_test, C, gamma):
    b, alphas = lssvm_train(X_train, y_train, C, gamma)
    y_test_pred = lssvm_predict(alphas, b, X_test, X_train, y_train, gamma)
    y_train_pred = lssvm_predict(alphas, b, X_train, X_train, y_train, gamma)

    test_error = 0
    for i in range(len(y_test_pred)):
        if ((y_test_pred[i] * y_test[i]) < 0):
            test_error += 1
    test_error = test_error / len(y_test_pred)

    train_error = 0
    for i in range(len(y_train_pred)):
        if ((y_train_pred[i] * y_train[i]) < 0):
            train_error += 1
    train_error = train_error / len(y_train_pred)

    return round(test_error, 3), round(train_error, 3)


def run_model(kernel, X_train, y_train, X_test, y_test, C, gamma):
    if (kernel == "lssvm"):
        return run_lssvm_model(X_train, y_train, X_test, y_test, C, gamma)
    svclassifier = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=2)
    svclassifier.fit(X_train, y_train)
    y_test_pred = svclassifier.predict(X_test)
    y_train_pred = svclassifier.predict(X_train)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    return round(1 - test_accuracy, 3), round(1 - train_accuracy, 3)


X_train, y_train = read_data("data/train.csv")
X_test, y_test = read_data("data/test.csv")

kernels = ["linear", "rbf", "lssvm"]
C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

boxplot_labels = ["linear_train", "linear_test", "rbf_train", "rbf_test",
                  "lssvm_train", "lssvm_test"]
errors_dict = {label: [] for label in boxplot_labels}

for kernel in kernels:
    for C in C_list:
        for gamma in gamma_list:
            test_error, train_error = run_model(kernel, X_train, y_train,
                                                X_test, y_test, C, gamma)
            errors_dict[kernel + "_test"].append(test_error)
            errors_dict[kernel + "_train"].append(train_error)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(errors_dict.values())
ax.set_xticklabels(boxplot_labels)
fig.savefig('errors_boxplot.png', bbox_inches='tight')
print("Errors: ", errors_dict)

errors_dict = {label: [] for label in boxplot_labels}
for kernel in kernels:
    for gamma in gamma_list:
        test_error, train_error = run_model(kernel, X_train, y_train, X_test,
                                            y_test, 1, gamma)
        errors_dict[kernel + "_test"].append(test_error)
        errors_dict[kernel + "_train"].append(train_error)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(errors_dict.values())
ax.set_xticklabels(boxplot_labels)
fig.savefig('errors_boxplot_gamma.png', bbox_inches='tight')
print("Errors with varying Gamma: ", errors_dict)

errors_dict = {label: [] for label in boxplot_labels}
for kernel in kernels:
    for C in C_list:
        test_error, train_error = run_model(kernel, X_train, y_train, X_test,
                                            y_test, C, 1)
        errors_dict[kernel + "_test"].append(test_error)
        errors_dict[kernel + "_train"].append(train_error)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(errors_dict.values())
ax.set_xticklabels(boxplot_labels)
fig.savefig('errors_boxplot_C.png', bbox_inches='tight')
print("Errors with varying C: ", errors_dict)
