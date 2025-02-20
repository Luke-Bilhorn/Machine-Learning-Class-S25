import numpy as np
from linear_regression import mlr_ridge, pred_and_r2
from pytest import approx
from sklearn.datasets import fetch_california_housing, load_diabetes


def test_calif1() :
    calif = fetch_california_housing()
    reg = mlr_ridge(calif.data[:18000], calif.target[:18000])
    assert pred_and_r2(reg, calif.data[18000:], calif.target[18000:]) == approx(.726, .001)
    
def test_califpoint01() :
    calif = fetch_california_housing()
    reg = mlr_ridge(calif.data[:18000], calif.target[:18000], alpha=.01)
    assert pred_and_r2(reg, calif.data[18000:], calif.target[18000:]) == approx(.726, .001)
    
def test_calif50() :
    calif = fetch_california_housing()
    reg = mlr_ridge(calif.data[:18000], calif.target[:18000], alpha=50)
    assert pred_and_r2(reg, calif.data[18000:], calif.target[18000:]) == approx(.726, .001)
    
def test_diabetes1() :
    diabetes = load_diabetes()
    reg = mlr_ridge(diabetes.data[:400], diabetes.target[:400])
    assert pred_and_r2(reg, diabetes.data[400:], diabetes.target[400:]) == approx(.633, .001)
    
def test_diabetespoint1() :
    diabetes = load_diabetes()
    reg = mlr_ridge(diabetes.data[:400], diabetes.target[:400], alpha=.1)
    assert pred_and_r2(reg, diabetes.data[400:], diabetes.target[400:]) == approx(.7003, .0001)
    
def test_diabetes10() :
    diabetes = load_diabetes()
    reg = mlr_ridge(diabetes.data[:400], diabetes.target[:400], alpha=10)
    assert pred_and_r2(reg, diabetes.data[400:], diabetes.target[400:]) == approx(.4998, .0001)
    
