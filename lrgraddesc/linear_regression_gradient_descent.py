import numpy as np
import functools as ft
from random import gauss

# Extend a vector to have a one in position 0
def extend(v) :
    ve = np.ones(len(v) + 1)
    ve[1:] = v
    return ve

def pred_and_r2(reg, X_test, Y_test) :
    assert len(X_test) == len(Y_test)
    Y_pred = np.array([reg(x) for x in X_test])
    return ((np.corrcoef(Y_pred, Y_test))[0,1])**2

def theta_to_function(theta) :
    return lambda x : np.matmul(theta, extend(x))

def linear_regression_grad_desc(X, Y, eta=.1, epsilon=.00001) :
    assert len(X) == len(Y)
    (N, D) = X.shape
    Xe = np.ones((N, D+1))
    Xe[:,1:] = X
    theta = np.array([gauss(0., 1.) for i in range(D+1)])
    return train(N, Xe, Y, theta, eta, epsilon, 40000, loss, gradient)
    
def train(N, Xe, Y, Th0, eta, epsilon, itterations, lossFunc, gradientFunc, alpha=None) :
    loss0 = lossFunc(N, Xe, Y, Th0, alpha)
    for i in range(itterations) :
        grad = gradientFunc(N, Xe, Y, Th0, alpha)
        Th1 = Th0 - eta * grad
        loss1 = lossFunc(N, Xe, Y, Th1, alpha)
        if np.abs(loss1 - loss0) < epsilon : break
        Th0 = Th1
        loss0 = loss1
    return Th1

def loss(N, X, y, Th, alpha) :
    XTh = np.matmul(X, Th)
    A = (-1) * XTh
    B = np.add(y, A)
    C = np.linalg.norm(B)
    D = C ** 2
    E = D / N
    return E 
    
def RRloss(N, X, y, Th, alpha) :
    XTh = np.matmul(X, Th)
    A = (-1) * XTh
    B = np.add(y, A)
    C = np.linalg.norm(B)
    D = C ** 2
    E = D / N
    F = np.linalg.norm(Th)
    G = F ** 2
    H = G * alpha
    I = E + H
    return I
    
def LASSOloss(N, X, y, Th, alpha) :
    XTh = np.matmul(X, Th)
    A = (-1) * XTh
    B = np.add(y, A)
    C = np.linalg.norm(B)
    D = C ** 2
    E = D / N
    F = np.linalg.norm(Th)
    G = F * alpha
    H = E + G
    return H

def gradient(N, X, y, Th, alpha) :
    yT = np.matrix.transpose(y)
    yTX = np.matmul(yT, X)
    A = yTX * (-2)
    ThT = np.matrix.transpose(Th)
    XT = np.matrix.transpose(X)
    ThTXT = np.matmul(ThT, XT)
    ThTXTX = np.matmul(ThTXT, X)
    B = ThTXTX * (2)
    C = np.add(A, B)
    D = C / N
    return D
    
def RRgradient(N, X, y, Th, alpha) :
    yT = np.matrix.transpose(y)
    yTX = np.matmul(yT, X)
    A = yTX * (-2)
    ThT = np.matrix.transpose(Th)
    XT = np.matrix.transpose(X)
    ThTXT = np.matmul(ThT, XT)
    ThTXTX = np.matmul(ThTXT, X)
    B = ThTXTX * (2)
    C = np.add(A, B)
    D = C / N
    
    E = 2 * alpha * Th
    F = D + E
    return F
    
def LASSOgradient(N, X, y, Th, alpha) :
    yT = np.matrix.transpose(y)
    yTX = np.matmul(yT, X)
    A = yTX * (-2)
    ThT = np.matrix.transpose(Th)
    XT = np.matrix.transpose(X)
    ThTXT = np.matmul(ThT, XT)
    ThTXTX = np.matmul(ThTXT, X)
    B = ThTXTX * (2)
    C = np.add(A, B)
    D = C / N
    
    E = alpha * np.sign(Th)
    F = D + E
    return F

def ridge_regression_grad_desc(X, Y, eta=.1, epsilon=.00001, alpha=.01) :
    assert len(X) == len(Y)
    (N, D) = X.shape
    Xe = np.ones((N, D+1))
    Xe[:,1:] = X
    theta = np.array([gauss(0., 1.) for i in range(D+1)])
    return train(N, Xe, Y, theta, eta, epsilon, 40000, RRloss, RRgradient, alpha)

def lasso_regression_grad_desc(X, Y, eta=.1, epsilon=.00001, alpha=.01) :
    assert len(X) == len(Y)
    (N, D) = X.shape
    Xe = np.ones((N, D+1))
    Xe[:,1:] = X
    theta = np.array([gauss(0., 1.) for i in range(D+1)])
    return train(N, Xe, Y, theta, eta, epsilon, 40000, LASSOloss, LASSOgradient, alpha)
