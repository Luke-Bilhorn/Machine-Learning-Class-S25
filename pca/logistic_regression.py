import numpy as np
from random import gauss

# Extend a vector to have a one in position 0
def extend(v) :
    ve = np.ones(len(v) + 1)
    ve[1:] = v
    return ve

# The logistic function
def logistic(x) :
   value = 1./(1. + np.exp(-x))
   return value if value != 0 else float.MIN

# Make a logistic binary classifier from a given theta
def theta_to_binary_classifier(theta) :
    return lambda x : 1 if logistic(np.matmul(extend(x), theta)) >= 0.5 else 0

def logistic_regression_grad_desc(X, Y, eta=1., epsilon=.00001) :
    assert len(X) == len(Y)
    (N, D) = X.shape
    Xe = np.ones((N, D+1))
    Xe[:,1:] = X
    theta = np.array([gauss(0., 1.) for i in range(D+1)])
    theta = train(N, Xe, Y, theta, eta, epsilon, 3000, loss, gradient)  
    return theta_to_binary_classifier(theta)
    
def train(N, Xe, Y, Th0, eta, epsilon, itterations, lossFunc, gradientFunc) :
    loss0 = lossFunc(N, Xe, Y, Th0)
    #for i in range(itterations) :
    while(True) :
        grad = gradientFunc(N, Xe, Y, Th0)
        Th1 = Th0 - eta * grad
        loss1 = lossFunc(N, Xe, Y, Th1)
        if np.abs(loss1 - loss0) < epsilon : break
        Th0 = Th1
        loss0 = loss1
    return Th1

def loss(N, X, y, Th) :
    return -1 * (sum([((y[n] * np.log(logistic(np.matmul(np.matrix.transpose(Th), X[n])))) + ((1 - y[n]) * np.log(1 - logistic(np.matmul(np.matrix.transpose(Th), X[n]))))) for n in range(N)])) / N

def gradient(N, X, y, Th) :
    (N, D) = X.shape
    return np.array([sum([X[n, i] * (logistic(np.matmul(np.matrix.transpose(Th), X[n])) - y[n]) for n in range(N)]) for i in range(D)]) / N
