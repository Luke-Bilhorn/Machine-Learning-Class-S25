from warmup import selection_sort
from collections import Counter


def assert_sorted(A):
    for i in range(1, len(A)) :
        assert A[i-1] <= A[i]

def assert_bag(A, B):
    assert Counter(A) == Counter(B)
    
def check_sequence(A) :
    B = list(A)
    selection_sort(B)
    assert_sorted(B)
    assert_bag(A,B)
        
def test_presorted():
    check_sequence([1, 2, 3, 4, 5, 6, 7, 8 ])
            
def test_revsorted():
    check_sequence([8, 7, 6, 5, 4, 3, 2, 1])
        
def test_shuffled():
    check_sequence([5, 8, 3, 5, 8, 7, 9, 2 ])
        
def test_hasneg():
    check_sequence([-5, 7, 14, 81, -10, 0, 8, 8])
        
def test_lotsrepeat():
    check_sequence([2, 5, 5, 5, 5, 5, 5, 2 ])
