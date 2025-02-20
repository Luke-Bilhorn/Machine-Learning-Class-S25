
import numpy as np
from pytest import approx
from logistic_regression import theta_to_binary_classifier

def helper(theta, X, y) :
    assert len(X) == len(y)
    classy = theta_to_binary_classifier(theta)
    y_predict = [classy(x) for x in X]
    for i in range(len(y)) :
        assert y_predict[i] == approx(y[i], .00001)

def test_plain() :
    helper([1., 2., 3., 4.],
           [[ 0.6030143,   0.46440383,  2.40212227],
            [ 0.07448985, -2.53916166, -1.93251817],
            [ 0.23237028, -0.7473277,   3.90897611],
            [-0.31965506,  1.81814971, -2.45450533],
            [ 0.34550749,  3.14852736,  4.44579314],
            [ 0.3483335,   3.22130458,  1.53766886],
            [ 0.63274505,  2.89150874, -2.89695522],
            [-0.1250244,   0.91661589,  2.07004665],
            [-0.12021277, -1.0397062,   1.09712488],
            [ 0.01224893,  4.14119891, -6.78951603]],
           [1, 0, 1, 0, 1, 1, 0, 1, 1, 0])

def test_all_one() :
    helper([.001, .001],
           [[.0001], [.0002],[.0003],[.0004],[.0005]],
           [1, 1, 1, 1, 1])

def test_all_zero() :
    helper([50., 60.],
           [[-1.], [-2.],[-3.],[-4.],[-5.]],
           [0, 0, 0, 0, 0])
               
 
