from pca import PrincipalComponents
import numpy as np
from pytest import approx

X_art = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.], [2.,1.], [1.,2.], [3.,1.], [1.,3.], [2.,2.], [4.,2.], [2.,4.], [4.,4.], [5.,4.], [4.,5.], [6.,6.], [7.,7.]])

def test_artificial_one() :
    spca = PrincipalComponents(X_art, 1)
    assert spca.components == approx(np.array([[ 0.70710678, -0.70710678]]), .0001)
    

def test_artificial_two() :
    spca = PrincipalComponents(X_art, 2)
    assert spca.components == approx(np.array([[ 0.70710678, -0.70710678],
                                                [ 0.70710678,  0.70710678]]), .0001)
