import numpy as np
from numpy_prac import row_col_sorted

def test_1x1() :
    assert row_col_sorted(np.asarray([[0]]))

def test_1x2() :
    assert row_col_sorted(np.asarray([[0,1]]))

def test_2x1() :
    assert row_col_sorted(np.asarray([[0],[1]]))

def test_2x2() :
    assert row_col_sorted(np.asarray([[0,1],[1,10]]))

def test_5x5() :
    assert row_col_sorted(
            np.asarray([[0,1,10,11,20],[1,10,11,20,21],[10,11,20,31,40],[41,50,61,70,81],[80,91,100,101,102]]))

def test_3x6() :
    assert row_col_sorted(
            np.asarray([[0,1,10,11,20,21],[1,10,11,20,21,30],[30,31,40,41,50,51],]))

def test_6x3() :
    assert row_col_sorted(
            np.asarray([[0,1,10],[1,10,11],[10,11,20],[21,30,31],[30,31,40],[41,50,51]]))

def test_1x2_not() :
    assert not row_col_sorted(np.asarray([[10,1]]))

def test_2x1_not() :
    assert not row_col_sorted(np.asarray([[10],[1]]))

def test_2x2_not() :
    assert not row_col_sorted(np.asarray([[0,1],[1,0]]))

def test_5x5_not() :
    assert not row_col_sorted(
            np.asarray([[0,1,10,11,20],[1,10,21,20,21],[10,11,20,31,40],[41,50,61,70,81],[80,91,100,101,102]]))

def test_3x6_not() :
    assert not row_col_sorted(
            np.asarray([[0,1,10,11,20,1],[1,10,11,20,21,30],[30,31,40,41,50,51],]))

def test_6x3_not() :
    assert not row_col_sorted(
            np.asarray([[0,1,10],[1,10,11],[10,11,20],[21,30,31],[30,31,52],[41,50,51]]))
