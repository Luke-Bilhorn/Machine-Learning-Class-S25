import numpy as np

class PrincipalComponents :

    def __init__(self, X, M) :
        (N, D) = X.shape
        self.means = np.mean(X, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(np.cov((X - self.means).T)) #(np.cov(X - self.means).T)) #Xcov)
        self.components = [eigenvectors[i] for i in np.argsort(eigenvalues)[-M:]]
        self.components = self.components[::-1]



           


    def transform(self, X) :
        M = len(self.components)
        (N, D) = X.shape
        assert D == len(self.means)
        Xh = X - self.means
        return np.array([[np.dot(Xh[i], self.components[j]) for j in range(M)] for i in range(N)])