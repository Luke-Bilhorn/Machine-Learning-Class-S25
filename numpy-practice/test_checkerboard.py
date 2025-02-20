import numpy as np
from numpy_prac import checkerboard

def test_1x1() :
    assert (checkerboard(1,1) == np.asarray([[0]])).all()

def test_1x2() :
    assert (checkerboard(1,2) == np.asarray([[0,1]])).all()

def test_2x1() :
    assert (checkerboard(2,1) == np.asarray([[0],[1]])).all()

def test_2x2() :
    assert (checkerboard(2,2) == np.asarray([[0,1],[1,0]])).all()

def test_5x5() :
    assert (checkerboard(5,5) ==
            np.asarray([[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[0,1,0,1,0]])).all()

def test_3x6() :
    assert (checkerboard(3,6) ==
            np.asarray([[0,1,0,1,0,1],[1,0,1,0,1,0],[0,1,0,1,0,1],])).all()

def test_6x3() :
    assert (checkerboard(6,3) ==
            np.asarray([[0,1,0],[1,0,1],[0,1,0],[1,0,1],[0,1,0],[1,0,1]])).all()
