import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as sklrbf

# Linear kernel
def linear(x1, x2) :
    # The +1 is to ensure positive definiteness (I think)
    return np.dot(x1, x2) + 1


# Function to make a polynomial kernel, given
# a power s
def make_poly_kernel(s) :
    return lambda x1, x2 : (1 + np.dot(x1, x2))**s

# "Radial basis function" kernel
def rbf(x1, x2) :
    return sklrbf([x1], [x2])[0][0]

