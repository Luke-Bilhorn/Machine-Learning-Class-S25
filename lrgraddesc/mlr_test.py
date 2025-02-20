import numpy as np
from linear_regression_gradient_descent import linear_regression_grad_desc, pred_and_r2, theta_to_function
from pytest import approx
from sklearn.datasets import  load_diabetes
from random import gauss
from sklearn import preprocessing


    


def test_synthetic1_by_theta() :
    coefs = np.array([.5, .2, 1.5, -.8, .75])
    synthetic = theta_to_function(coefs)
    X = np.array([[gauss(0, 1), gauss(.5, .25), gauss(0, 3), gauss(1, .5)] for n in range(1000)])
    y = np.array([synthetic(x) for x in X])
    X_train = X[:800]
    y_train = y[:800]
    X_test = X[800:]
    y_test = y[800:]
    theta = linear_regression_grad_desc(X_train, y_train)
    assert len(theta) == len(coefs)
    for i in range(len(theta)) :
        assert abs(theta[i] - coefs[i]) < .15

def test_synthetic1_by_score() :
    coefs = np.array([.5, .2, 1.5, -.8, .75])
    synthetic = theta_to_function(coefs)
    X = np.array([[gauss(0, 1), gauss(.5, .25), gauss(0, 3), gauss(1, .5)] for n in range(1000)])
    y = np.array([synthetic(x) for x in X])
    X_train = X[:800]
    y_train = y[:800]
    X_test = X[800:]
    y_test = y[800:]
    theta = linear_regression_grad_desc(X_train, y_train)
    reg = theta_to_function(theta)
    assert pred_and_r2(reg, X_train, y_train) == approx(1., .001)
    assert pred_and_r2(reg, X_test, y_test) == approx(1., .001)
    
def test_synthetic2_by_theta() :
    coefs = np.array([.5, .2, 0, -.8, .75])
    synthetic = theta_to_function(coefs)
    X = np.array([[gauss(0, 1), gauss(.5, .25), gauss(0, 3), gauss(1, .5)] for n in range(1000)])
    y = np.array([synthetic(x) for x in X])
    X_train = X[:800]
    y_train = y[:800]
    X_test = X[800:]
    y_test = y[800:]
    theta = linear_regression_grad_desc(X_train, y_train)
    assert len(theta) == len(coefs)
    for i in range(len(theta)) :
        assert abs(theta[i] - coefs[i]) < .15

def test_synthetic2_by_score() :
    coefs = np.array([.5, .2, 0, -.8, .75])
    synthetic = theta_to_function(coefs)
    X = np.array([[gauss(0, 1), gauss(.5, .25), gauss(0, 3), gauss(1, .5)] for n in range(1000)])
    y = np.array([synthetic(x) for x in X])
    X_train = X[:800]
    y_train = y[:800]
    X_test = X[800:]
    y_test = y[800:]
    theta = linear_regression_grad_desc(X_train, y_train)
    reg = theta_to_function(theta)
    assert pred_and_r2(reg, X_train, y_train) == approx(1., .001)
    assert pred_and_r2(reg, X_test, y_test) == approx(1., .001)

def test_diabetes() :
    diabetes = load_diabetes()
    reg = theta_to_function(linear_regression_grad_desc(diabetes.data[:400], diabetes.target[:400]))
    assert pred_and_r2(reg, diabetes.data[:400], diabetes.target[:400]) == approx(.5, .05)
    assert pred_and_r2(reg, diabetes.data[400:], diabetes.target[400:]) == approx(.7, .05)
    
def test_mpg() :
    mpg_data = np.genfromtxt('auto-mpg.csv', skip_header=1, delimiter=',',usecols=(1,2,3,4,5,6,7))
    mpg_target = np.genfromtxt('auto-mpg.csv', skip_header=1, delimiter=',', usecols=(0))
    mpg_data_scaled = preprocessing.scale(mpg_data)
    mpg_target_scaled = preprocessing.scale(mpg_target)
    X_train = mpg_data_scaled[:300]
    X_test = mpg_data_scaled[300:]
    y_train = mpg_target_scaled[:300]
    y_test = mpg_target_scaled[300:]
    reg = theta_to_function(linear_regression_grad_desc(X_train, y_train))
    assert pred_and_r2(reg, X_train, y_train) == approx(.81, .015)
    assert pred_and_r2(reg, X_test, y_test) == approx(.5, .05)
  
