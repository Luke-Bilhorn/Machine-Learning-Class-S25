from pca import PrincipalComponents
import numpy as np
from pytest import approx
from sklearn.datasets import load_iris
iris_dataset = load_iris()


def helper(X, M) :
    skpca = PCA(n_components=M)
    skpca.fit(X)
    skpca.components = abs(skpca.components)
    spca = PrincipalComponents(X, M)
    assert np.abs(spca.components) == approx(skpca.components, .000001)
    X_t = skpca.transform(X)
    X_transformed = spca.transform(X)
    assert X_transformed == approx(X_t, .00001)


X_art = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.], [2.,1.], [1.,2.], [3.,1.], [1.,3.], [2.,2.], [4.,2.], [2.,4.], [4.,4.], [5.,4.], [4.,5.], [6.,6.], [7.,7.]])

    

def test_artificial_1trans() :
     spca = PrincipalComponents(X_art, 1)
     X_transformed = spca.transform(X_art)
     assert X_transformed[0] == approx(-2.22044605e-16, .0001)
     assert X_transformed[2] == approx( 7.07106781e-01, .0001)
     assert X_transformed[8] == approx(-1.11022302e-16, .0001)
     

def test_artificial_2trans() :
     spca = PrincipalComponents(X_art, 2)
     X_transformed = spca.transform(X_art)
     assert X_transformed[0] == approx([-2.22044605e-16, -3.80069895e+00], .0001)
     assert X_transformed[2] == approx([ 7.07106781e-01, -3.09359217e+00], .0001)
     assert X_transformed[8] == approx([-1.11022302e-16, -9.72271824e-01], .0001)

def test_iris() :
    spca = PrincipalComponents(iris_dataset.data, 3)
    assert spca.components == approx(np.array([[ 0.36138659, -0.65658877, -0.58202985,  0.31548719],
                                                [-0.08452251, -0.73016143,  0.59791083, -0.3197231 ],
                                                [ 0.85667061,  0.17337266,  0.07623608, -0.47983899]]), .00001)
    i_transformed = spca.transform(iris_dataset.data)
    assert i_transformed[0] == approx([ 0.49786886, -1.35075351, -0.26029109], .00001)
    assert i_transformed[4] == approx([ 0.39607132, -1.4153174,  -0.32862089], .00001)
    
