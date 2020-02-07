import matplotlib.pyplot as plt
import numpy as np

import sklearn
import copy
from sklearn.datasets import fetch_california_housing

plt.rcParams['font.size'] = 14


def plot_points(feature_names, data, y, plot_title):
    fig = plt.figure(figsize=(9, 6))
    for i, feature in enumerate(feature_names):
        ax = fig.add_subplot(3, 3, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        ax.plot(data[feature], y, '.')
        ax.title.set_text(feature)
        fig.savefig(plot_title)


def adaboost(feature_names, data, y, num_samples, stump_title):
    # Create stumps
    # bin the data by proportion, 10% in each bin
    bins = {}
    bin_idx = (np.arange(0, 1.1, 0.1) * num_samples).astype(np.int16)
    bin_idx[-1] = bin_idx[-1] - 1
    for feature in (feature_names):
        bins[feature] = np.sort(data[feature])[bin_idx]

    # decision stumps as weak classifiers
    # 0 if not in bin, 1 if in bin
    stumps = {}
    for feature in feature_names:
        stumps[feature] = np.zeros([num_samples, len(bins[feature]) - 1])
        for n in range(len(bins[feature]) - 1):
            stumps[feature][:, n] = data[feature] > bins[feature][n]

    # stack the weak classifiers into a matrix
    H = np.hstack([stumps[feature] for feature in feature_names])
    H = np.hstack([np.ones([num_samples, 1]), H])
    # prepare the vector for storing weights
    alphas = np.zeros(H.shape[1])

    num_iterations = 30
    MSE = np.zeros(num_iterations)  # track mean square error

    for iteration in range(num_iterations):
        f = np.matmul(H, alphas)
        r = y - f
        MSE[iteration] = np.mean(r ** 2)  # r = residual
        idx = 0
        k = np.absolute(np.matmul(H.T[idx], r))
        for i in range(1, H.shape[1]):
            if np.absolute(np.matmul(H.T[i], r)) > k:
                k = np.absolute(np.matmul(H.T[i], r))
                idx = i
        alphas[idx] = alphas[idx] + (
                    1 / np.matmul(H.T, H)[idx][idx]) * np.matmul(H.T[idx], r)

    alphasf = {}
    start = 1
    for feature in feature_names:
        alphasf[feature] = alphas[start:(start + stumps[feature].shape[1])]
        start = start + stumps[feature].shape[1]
    alphasf['mean'] = alphas[0]

    fig = plt.figure(figsize=(9, 6))
    for i, feature in enumerate(feature_names):
        ax = fig.add_subplot(3, 3, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        ax.plot(data[feature], y - np.mean(y), '.', alpha=0.5,
                color=[0.9, 0.9, 0.9])
        fq = np.matmul(stumps[feature], alphasf[feature])
        ax.plot(data[feature], fq, '.')
        ax.title.set_text(feature)
        ax.set_xlim([bins[feature][0], bins[feature][-2]])
        fig.savefig(stump_title)

    return MSE[-1]


def variable_importance(feature_names, data, y, num_samples):
    variable_importance = {}
    plot_points(feature_names, data, y, "feature_point_plot.png")
    error_original = adaboost(feature_names, data, y, num_samples,
                              "Original_Stumps.png")
    original_data = copy.deepcopy(data)
    for n, feature in enumerate(feature_names):
        np.random.shuffle(data[feature])
        plot_title = feature + "_permuted_plot.png"
        plot_points(feature_names, data, y, plot_title)
        stump_title = feature + "_Stumps.png"
        variable_importance[feature] = adaboost(feature_names, data, y,
                                                num_samples,
                                                stump_title) - error_original
        data = copy.deepcopy(original_data)
    return variable_importance


# Download data
tmp = sklearn.datasets.fetch_california_housing()
num_samples = tmp['data'].shape[0]
feature_names = tmp['feature_names']
y = tmp['target']
X = tmp['data']

data = {}
for n, feature in enumerate(feature_names):
    data[feature] = tmp['data'][:, n]

plot_points(feature_names, data, y, "feature_point_plot.png")
adaboost(feature_names, data, y, num_samples, "Original_Stumps.png")
variable_importance = variable_importance(feature_names, data, y, num_samples)
print(variable_importance)
