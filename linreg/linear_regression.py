import numpy as np

# Extend a vector to have a one in position 0
def extend(v) :
    ve = np.ones(len(v) + 1)
    ve[1:] = v
    return ve

def extendMatrix(M) :
    #Ma = np.ones((len(M), len(M[0]) + 1))
    #Ma[1:] = M

    ha = np.array([[1]] * len(M))
    Ma = np.hstack((ha, M))
   
    return Ma

def pred_and_r2(reg, X_test, Y_test) :
    assert len(X_test) == len(Y_test)
    Y_pred = np.array([reg(x) for x in X_test])
    return ((np.corrcoef(Y_pred, Y_test))[0,1])**2


def simple_linear_regression(X, Y) :
    assert len(X) == len(Y)
    N = len(X)
    Xbar = sum(X)/N
    Ybar = sum(Y)/N
    A = sum([(X[i] - Xbar)*(Y[i] - Ybar) for i in range(N)])
    B = sum([(X[i] - Xbar)*(X[i] - Xbar) for i in range(N)])
    Th1 = A/B
    Th0 = Ybar - Th1*Xbar
    return (lambda x : Th0 + Th1*x)


def multiple_linear_regression(X, Y) :
    assert len(X) == len(Y)

    #Xe = [np.concatenate([1], X[i]) for i in range(len(X))]
    #Xe2 = np.concatenate([1]*len(X), X)
    X = extendMatrix(X)
    Xa = np.matrix.transpose(X)
    Xb = np.matmul(Xa, X)
    Xc = np.linalg.inv(Xb)
    Xd = np.matmul(Xa, Y)
    Xe = np.matmul(Xc, Xd)
    Ths = Xe
   
    #Ths = np.matmul(np.linalg.inv(np.matmul(X, np.matrix.transpose(X))), np.matmul(np.matrix.transpose(X), Y))
   
    return lambda x: np.matmul(Ths,extend(x))

def mlr_ridge(X, Y, alpha=1.) :
    assert len(X) == len(Y)

    X = extendMatrix(X)
    Xa = np.matrix.transpose(X)
    Xb = np.matmul(Xa, X)
    A = np.identity(len(X[0]))
    A[0][0] = 0
    A = A * alpha
    Xc = np.add(Xb, A)
    Xd = np.linalg.inv(Xc)
    Xe = np.matmul(Xa, Y)
    Xf = np.matmul(Xd, Xe)
    Ths = Xf
   
   
   
    return lambda x: np.matmul(Ths,extend(x))
