import numpy as np
from collections import Counter


def minkowski(p) :
    def mink(a, b):
        assert len(a) == len(b)
        return sum(abs((a[i] - b[i])) ** p  for i in range(len(a))) ** (1 / p)
    return mink

def canberra(a, b) :
    assert len(a) == len(b)
    return sum(abs(a[i]-b[i])/(abs(a[i])+abs(b[i])) for i in range(len(a)))

def mahalanobis(x) :
    sig_inv = np.linalg.inv(np.cov(x.T))
    return lambda a, b : np.sqrt(np.dot(np.dot(a-b, sig_inv), a-b))

class KNN_Classifier :
    def __init__(self, X_train, Y_train, k, metric="L2") :
        # Store the training set with its targets and the number of neighbors
        self.X_train = X_train
        self.Y_train = Y_train
        self.k = k
        # Compute and store a norm
        if metric[0] == 'L' :
            self.distance = minkowski(int(metric[1:]))
        elif metric == 'C' :
            self.distance = canberra
        elif metric == 'M' :
            self.distance = mahalanobis(X_train)

    def classify(self, X) :
        # If X is a sequence of data points, then classify
        # all of them and return the results as an array
        if X.ndim > 1 :
            return np.array([self.classify(x) for x in X])
        # Otherwise, if X is a single data point, then classify it
        # and return that one result.
        else :
            List = [(self.distance(self.X_train[i], X), self.Y_train[i]) for i in range(len(self.X_train))]
            List.sort(key=lambda x: x[0])
            return Counter([item[1] for item in List][:self.k]).most_common(1)[0][0]

