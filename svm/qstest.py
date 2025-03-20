'''
Modified from example found at https://pypi.org/project/qpsolvers/
'''
from numpy import array, dot
import qpsolvers
from qpsolvers import solve_qp

M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))
A = array([1., 1., 1.])
b = array([1.])

print("P " + str(type(P)) + " " + str(P.shape))
print("q " + str(type(q)) + " " + str(q.shape))
print("G " + str(type(G)) + " " + str(G.shape))
print("h " + str(type(h)) + " " + str(h.shape))
print("A " + str(type(A)) + " " + str(A.shape))
print("b " + str(type(b)) + " " + str(b.shape))


print("QP solution:", solve_qp(P, q, G, h, A, b, solver='cvxopt'))

