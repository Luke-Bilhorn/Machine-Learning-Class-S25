from knn import minkowski, canberra, mahalanobis
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pytest import approx
from sklearn.datasets import load_iris
iris_dataset = load_iris()


data = iris_dataset.data[:10]

def test_L1() :
    distances = squareform(pdist(data, metric='cityblock'))
    metric = minkowski(1)
    assert metric(data[0],data[4]) == approx(distances[0,4], .01)
    assert metric(data[1],data[1]) == approx(distances[1,1], .01)
    assert metric(data[7],data[3]) == approx(distances[7,3], .01)
    assert metric(data[5],data[9]) == approx(distances[5,9], .01)

def test_L2() :
    distances = squareform(pdist(data, metric='euclidean'))
    metric = minkowski(2)
    assert metric(data[2],data[3]) == approx(distances[2,3], .01)
    assert metric(data[6],data[6]) == approx(distances[6,6], .01)
    assert metric(data[7],data[3]) == approx(distances[7,3], .01)
    assert metric(data[1],data[8]) == approx(distances[1,8], .01)

def test_L3() :
    distances = squareform(pdist(data, metric='minkowski', p=3))
    metric = minkowski(3)
    assert metric(data[1],data[8]) == approx(distances[1,8], .01)
    assert metric(data[9],data[9]) == approx(distances[9,9], .01)
    assert metric(data[7],data[3]) == approx(distances[7,3], .01)
    assert metric(data[2],data[5]) == approx(distances[2,5], .01)

def test_canberra() :
    distances = squareform(pdist(data, metric='canberra'))
    metric = canberra
    assert metric(data[7],data[8]) == approx(distances[7,8], .01)
    assert metric(data[2],data[2]) == approx(distances[2,2], .01)
    assert metric(data[7],data[3]) == approx(distances[7,3], .01)
    assert metric(data[1],data[4]) == approx(distances[1,4], .01)

def test_mahalanobis() :
    distances = squareform(pdist(data, metric='mahalanobis'))
    metric = mahalanobis(data)
    assert metric(data[0],data[2]) == approx(distances[0,2], .01)
    assert metric(data[3],data[3]) == approx(distances[3,3], .01)
    assert metric(data[7],data[3]) == approx(distances[7,3], .01)
    assert metric(data[3],data[8]) == approx(distances[3,8], .01)

