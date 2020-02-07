import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def kmedians(X, K=5, maxiter=100):
    C = np.zeros([K, 2])
    clusters = np.zeros(len(X))

    # initialize cluster centers
    for i in range(0, K):
        C[i][0] = np.random.randint(0, np.max(X[:, 0]))
        C[i][1] = np.random.randint(0, np.max(X[:, 1]))

    for iter in range(maxiter):
        # cluster assignment update
        for idx, x in enumerate(X):
            cluster = 0
            min_dist = np.linalg.norm(np.subtract(x, C[0]), 1)
            for k in range(1, K):
                dist = np.linalg.norm(np.subtract(x, C[k]), 1)
                if dist < min_dist:
                    cluster = k
                    min_dist = dist
            clusters[idx] = cluster

        # cluster center update
        for k in range(K):
            points = [X[j] for j in range(len(X)) if clusters[j] == k]
            C[k] = np.median(points, axis=0)

    return C


tmp = sio.loadmat("data/mousetracks.mat")

np.random.seed(1)

tracks = {}
for trackno in range(30):
    tracks[trackno] = tmp["num%d" % (trackno)]

X = np.zeros([30 * 50, 2])

for trackno in range(30):
    X[(trackno * 50):((trackno + 1) * 50), :] = tracks[trackno]

plt.close("all")
plt.plot(X[:, 0], X[:, 1], '.')
C = kmedians(X)
print("Median: ", C)
plt.plot(C[:, 0], C[:, 1], 'ro')
plt.axis("square")
plt.xlabel("meters")
plt.ylabel("meters")
plt.show()
