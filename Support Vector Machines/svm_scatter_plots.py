import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


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


def run_model(X_train, y_train, X_test, y_test, C, gamma, ax):
    svclassifier = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    svclassifier.fit(X_train, y_train)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1])
    y = np.linspace(ylim[0], ylim[1])
    Y, X = np.meshgrid(y, x)
    xy = np.c_[X.ravel(), Y.ravel()]
    P = svclassifier.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P,
               levels=[-1, 0, 1],
               linestyles=['--', '-', '--'],
               colors='k')
    ax.contourf(X, Y, P, cmap='RdBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='autumn')
    ax.scatter(svclassifier.support_vectors_[:, 0],
               svclassifier.support_vectors_[:, 1],
               facecolors='none', edgecolors='green')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


X_train, y_train = read_data("data/train.csv")
X_test, y_test = read_data("data/test.csv")

# part 1
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
run_model(X_train, y_train, X_test, y_test, 1, 1, ax)
fig.savefig('scatter.png')

# part 3 (a)
C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
fig = plt.figure(2, figsize=(9, 6))
for i in range(len(C_list)):
    ax = fig.add_subplot(3, 3, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    title = "C = " + str(C_list[i])
    ax.title.set_text(title)
    run_model(X_train, y_train, X_test, y_test, C_list[i], 1, ax)
fig.savefig('scatter_C.png')

# part 3 (b)
gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
fig = plt.figure(3, figsize=(9, 6))
for i in range(len(gamma_list)):
    ax = fig.add_subplot(3, 3, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    title = "gamma = " + str(gamma_list[i])
    ax.title.set_text(title)
    run_model(X_train, y_train, X_test, y_test, 1, gamma_list[i], ax)
fig.savefig('scatter_gamma.png')
