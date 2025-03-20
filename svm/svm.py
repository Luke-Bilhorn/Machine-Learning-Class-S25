import numpy as np
from kernel import linear
from qpsolvers import solve_qp

DEBUG = False

# Class to represent a SVM model
class SVM:
    def __init__(self, support_vectors, sv_targets, sv_lagrangians, kernel, b):
        self.support_vectors = support_vectors
        self.sv_targets = sv_targets
        self.sv_lagrangians = sv_lagrangians
        self.kernel = kernel
        self.b = b

    # Classify a sequence of data points X
    def classify(self, x) :
        N = x.shape[0]
        return np.sign([sum([self.sv_lagrangians[n]*self.sv_targets[n]*self.kernel(self.support_vectors[n], x[i]) for n in range(len(self.support_vectors))])+self.b for i in range(N)])


# Function to train a SVM model from data.
# You may choose to have this function return an instance
# of the SVM class, but whatever your train function returns,
# it needs to be appropriate as a first parameter to classify().
def train(data, targets, kernel, C = None):
    (N, D) = data.shape
        
    # 1. Make the kernel matrix
    K = np.array([[kernel(data[j], data[i]) for j in range(N)] for i in range(N)])
   
    if DEBUG: print(K)

    # 2. Make the other  vectors and matrices
    P = np.outer(targets, np.transpose(targets)) * K
    q = np.array([[-1.]]*N)
    A = targets *1.
    b = np.array([[0.]])

    if DEBUG: 
        print(P)
        print(q)
        print(A)
        print(b)

    if (C == None):
        # 3. HARD MARGIN CLASSIFICATION
        G = -1*np.identity(N)
        h = np.array([[0.]]*N)
    else:
        # 4. SOFT MARGIN CLASSIFICATION
        G = np.concatenate((np.identity(N), -1*np.identity(N)))
        h = np.concatenate(([[C]]*N, [[0.]]*N))

    if DEBUG:
        print(str(G))
        print(str(h))

    # 5. Find the lagrange multipliers
    lagrange_multipliers = solve_qp(P, q, G, h, A, b, solver='cvxopt')
    
    if DEBUG: print(lagrange_multipliers)
    
    # 6. Identify the support vectors
    threshold=0.001
    support_vectors =                 [data[i] for i in range(lagrange_multipliers.shape[0]) if abs(lagrange_multipliers[i]) > threshold]
    sv_targets      =              [targets[i] for i in range(lagrange_multipliers.shape[0]) if abs(lagrange_multipliers[i]) > threshold]
    sv_lagrangians  = [lagrange_multipliers[i] for i in range(lagrange_multipliers.shape[0]) if abs(lagrange_multipliers[i]) > threshold]

    if DEBUG:
        print(str(support_vectors))
        print(len(support_vectors))

    # 7. Compute the intercept
    
    b = (1 / len(support_vectors))*sum([sv_targets[i] - sum([sv_lagrangians[j]*sv_targets[j]*kernel(support_vectors[i], support_vectors[j]) for j in range(len(support_vectors))]) for i in range(len(support_vectors))])

    if DEBUG: print(b)

    # Return the svm classifier
    return SVM(support_vectors, sv_targets, sv_lagrangians, kernel, b)




