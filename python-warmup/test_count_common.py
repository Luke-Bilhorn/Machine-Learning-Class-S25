from warmup import count_common

def test_cc1() :
    assert count_common([], [1,4,5,3,2]) == 0

def test_cc2() :
    assert count_common([1,2,3,4], [5,6,7,8]) == 0

def test_cc3() :
    assert count_common([1,2,3,4,5], [3,4,5,6,7]) == 3

def test_cc4() :
    assert count_common([1,8,4,9,3,11,14], [4,21,7,9,2]) == 2

def test_cc5() :
    assert count_common([1,2,3,4,5,6,7], [7,3,6,1,2,5,4]) == 7
