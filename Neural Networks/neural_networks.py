import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io as sio

np.random.seed(1)


# L = 1 + (number of hidden + output layers) -- 1 for input features
# parameters: W1, b1, W2, b2, ..., W[L-1], b[L-1]
# D = L-1   -- hidden layers + output layers
# caches --  0 to (D-1)

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_activation_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    D = len(parameters) // 2
    for d in range(1, D + 1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(d)],
                                             parameters['b' + str(d)])
        caches.append(cache)
    return A, caches


def compute_cost(Yhat, Y):
    m = Y.shape[1]
    cost = (-1) * (np.dot(Y, np.transpose(np.log(Yhat))) + np.dot(1 - Y,
                                                                  np.transpose(np.log(1 - Yhat))))
    cost = np.squeeze(cost)
    return cost


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, np.transpose(A_prev))
    db = np.dot(dZ, np.ones((dZ.shape[1], 1)))
    dA_prev = np.dot(np.transpose(W), dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(Yhat, Y, caches):
    grads = {}
    D = len(caches)
    Y = Y.reshape(Yhat.shape)

    dYhat = np.divide(Yhat - Y, Yhat * (1 - Yhat))

    current_cache = caches[D - 1]
    grads["dA" + str(D - 1)], grads["dW" + str(D)], grads[
        "db" + str(D)] = linear_activation_backward(dYhat,
                                                    current_cache)

    for d in reversed(range(D - 1)):
        current_cache = caches[d]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(d + 1)], current_cache)
        grads["dA" + str(d)] = dA_prev_temp
        grads["dW" + str(d + 1)] = dW_temp
        grads["db" + str(d + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    D = len(parameters) // 2
    for d in range(D):
        parameters["W" + str(d + 1)] = parameters["W" + str(d + 1)] - (
                learning_rate * grads["dW" + str(d + 1)])
        parameters["b" + str(d + 1)] = parameters["b" + str(d + 1)] - (
                learning_rate * grads["db" + str(d + 1)])
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0001, num_iterations=3000):
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)
    grads = {}
    for i in range(0, num_iterations):
        Yhat, caches = L_model_forward(X, parameters)
        grads = L_model_backward(Yhat, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    return parameters, grads


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == y) / m)))
    return p


temp = sio.loadmat("data/mnist_all.mat")
train_data_0 = temp.get("train0")
test_data_0 = temp.get("test0")
train_data_1 = temp.get("train1")
test_data_1 = temp.get("test1")

train_x = (np.concatenate((train_data_0, train_data_1)).T) / 255
train_y = np.concatenate((np.zeros((train_data_0.shape[0], 1)),
                          np.ones((train_data_1.shape[0], 1)))).T
test_x = (np.concatenate((test_data_0, test_data_1)).T) / 255
test_y = np.concatenate(
    (np.zeros((test_data_0.shape[0], 1)), np.ones((test_data_1.shape[0], 1)))).T

# Part (c)
# layers_dims = [784, 20, 20, 1]
#
# parameters, grads = L_layer_model(train_x, train_y, layers_dims,
#                                   num_iterations=3000)
# pred_train = predict(train_x, train_y, parameters)
# pred_test = predict(test_x, test_y, parameters)

# Part (d)
paraNorm = []
for i in range(1, 10):
    layers_dims = [784]
    for d in range(0, i):  # d = no. of hidden layers
        layers_dims.append(20)
    layers_dims.append(1)
    parameters, grads = L_layer_model(train_x, train_y, layers_dims, 1, num_iterations=1)
    norm = np.linalg.norm(grads['dW1'], ord='fro')
    print(norm)
    paraNorm.append(norm)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(2, 2, 1)
line, = ax.plot(paraNorm, color='blue', lw=2)
ax.set_yscale('log')
plt.xlabel('Number of hidden layers')
plt.ylabel('Log of Frobenius norm of dW1')
pylab.show()
