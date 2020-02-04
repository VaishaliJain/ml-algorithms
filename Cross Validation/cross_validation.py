import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def read_data():
    data = []
    with open('data/transfusion.data') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # take the header out
        for row in reader:  # each row is a list
            data.append(row)
    data = np.array(data, dtype=np.int32)
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def get_training(x, m):
    return x[:m] + x[(m + 1):]


# Removes the extra dimension from training set
# which was introduced due to array_split
def rld(l):
    return [item for sublist in l for item in sublist]


def run_model(k, X, X_t, y, y_t):
    model = LogisticRegression(C=k)
    model.fit(X, y)
    y_predicted = model.predict(X_t)
    return f1_score(y_t, y_predicted)


X, y = read_data()

X_split = np.array_split(X, 5)
y_split = np.array_split(y, 5)
k_val = [0.1, 1, 10, 100]  # Given values of parameter C
k_acc = {}
scores = []

for i in range(0, 5):  # Outer loop for testing
    x_training = get_training(X_split, i)
    x_test = X_split[i]
    y_training = get_training(y_split, i)
    y_test = y_split[i]
    k_acc = {k: [] for k in k_val}

    for j in range(0, 4):  # Inner loop for validation
        x_training2 = get_training(x_training, j)
        x_validation = x_training[j]
        y_training2 = get_training(y_training, j)
        y_validation = y_training[j]

        for k in k_val:  # loop for evaluation over all C values
            inner_score = run_model(k, rld(x_training2), x_validation,
                                    rld(y_training2), y_validation)
            k_acc[k].append(inner_score)

    k_avg_acc = {k: np.mean(k_acc[k]) for k in k_val}
    best_k = max(k_avg_acc, key=k_avg_acc.get)
    score = run_model(best_k, rld(x_training), x_test, rld(y_training), y_test)
    scores.append(score)
    print('Fold: ', i + 1, ' Test Data f1-score:', score, 'C value: ', best_k)

average_score = np.mean(scores)
standard_deviation = np.std(scores)
print('Average f1-score: ', average_score)
print('Standard Deviation of f1-score: ', standard_deviation)
