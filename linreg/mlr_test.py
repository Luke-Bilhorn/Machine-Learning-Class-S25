import numpy as np
from linear_regression import multiple_linear_regression, pred_and_r2
from pytest import approx
from sklearn.datasets import fetch_california_housing, load_diabetes


def test_calif() :
    calif = fetch_california_housing()
    reg = multiple_linear_regression(calif.data[:18000], calif.target[:18000])
    assert pred_and_r2(reg, calif.data[18000:], calif.target[18000:]) == approx(.726, .001)
    
def test_diabetes() :
    diabetes = load_diabetes()
    reg = multiple_linear_regression(diabetes.data[:400], diabetes.target[:400])
    assert pred_and_r2(reg, diabetes.data[400:], diabetes.target[400:]) == approx(.714, .001)
    
