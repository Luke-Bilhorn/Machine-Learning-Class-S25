import svm
import numpy as np
import kernel

def test_one_dim() :
    clsfyr = svm.SVM(np.array([[1,0], [-1,0]]), [1, -1], [1,1],  kernel.linear, 0)
    X = np.array([[1.0,0.0],[2.0,0.0],[3.0,0.0],[-1.0,0.0],[-2.0,0.0],[-3.0,0.0]])
    y = np.array([1,1,1,-1,-1,-1])
    pred_test = clsfyr.classify(X)
    assert np.all(y == pred_test)

def test_two_dim() :
    clsfyr = svm.SVM(np.array( [[-1.0, 0.0],[1.0, 0.0]]), [-1, 1], [1,1], kernel.linear, 0)
    X =  np.array( [[-1.0, 0.0], [-2.0, 1.0], [-2.0, -1.0], [-3.0, 0.0], [1.0, 0.0], [2.0, 1.0], [2.0, -1.0], [3.0, 0.0], [-2.0, 0.0], [-3.0, 1.0], [-3.0, -1.0], [2.0, 0.0], [3.0, 1.0], [3.0, -1.0]])
    y =  np.array([-1., -1., -1., -1.,  1.,  1.,  1.,  1., -1., -1., -1.,  1.,  1., 1.])
    pred_test = clsfyr.classify(X)
    assert np.all(y == pred_test)
