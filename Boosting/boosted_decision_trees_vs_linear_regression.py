from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import sklearn
from sklearn.datasets import fetch_california_housing

plt.rcParams['font.size'] = 14

# Download data

tmp = sklearn.datasets.fetch_california_housing()

num_samples = tmp['data'].shape[0]
feature_names = tmp['feature_names']
y = tmp['target']
X = tmp['data']

data = {}
for n, feature in enumerate(feature_names):
    data[feature] = tmp['data'][:, n]

clf = GradientBoostingRegressor(loss="ls")
clf.fit(X, y)

clf2 = LinearRegression()
clf2.fit(X, y)

plt.close("all")
plt.figure(figsize=[10, 10])
ax = plt.gca()
plot_partial_dependence(clf, X, feature_names, feature_names, n_cols=3, ax=ax)
plt.tight_layout()
plt.show()

print(np.mean((y - clf2.predict(X)) ** 2))
print(np.mean((y - clf.predict(X)) ** 2))
