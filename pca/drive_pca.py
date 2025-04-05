import numpy as np
from pca import PrincipalComponents
from logistic_regression import logistic_regression_grad_desc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def log_reg_accuracy(X, Y, M) :
    X = scaler.fit_transform(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    log_reg = logistic_regression_grad_desc(X_train, Y_train)
    Y_train_classified = np.array([log_reg(x) for x in X_train])
    Y_test_classified = np.array([log_reg(x) for x in X_test])


    PCA = PrincipalComponents(X_train, M)
    X_train_PCA = PCA.transform(X_train)
    X_test_PCA = PCA.transform(X_test)
 
    log_reg_PCA = logistic_regression_grad_desc(X_train_PCA, Y_train)
    Y_train_classified_PCA = np.array([log_reg_PCA(x) for x in X_train_PCA])
    Y_test_classified_PCA = np.array([log_reg_PCA(x) for x in X_test_PCA])



    return (np.sum(Y_train_classified == Y_train)/len(Y_train),
            np.sum(Y_test_classified == Y_test)/len(Y_test),
            np.sum(Y_train_classified_PCA == Y_train)/len(Y_train),
            np.sum(Y_test_classified_PCA == Y_test)/len(Y_test))


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Breast cancer, should be (0.9906103286384976, 0.951048951048951, 0.9906103286384976, 0.951048951048951)")
print(log_reg_accuracy(cancer.data, cancer.target, 16))

from sklearn.datasets import load_iris
iris = load_iris()
print("Iris 2 vs all, should be (0.9732142857142857, 0.9736842105263158, 0.9732142857142857, 0.9736842105263158)")
print(log_reg_accuracy(iris["data"], iris["target"]==2, 3))
print("Iris 1 vs all, should be (0.7589285714285714, 0.6578947368421053, 0.7589285714285714, 0.6578947368421053)")
print(log_reg_accuracy(iris["data"], iris["target"]==1, 3))
print("Iris 0 vs all, should be (1.0, 1.0, 1.0, 1.0)")
print(log_reg_accuracy(iris["data"], iris["target"]==0, 3))

#from sklearn.datasets import load_digits
#digits_dataset = load_digits()
#bit_data = np.array([digits_dataset.data[i] for i in range(len(digits_dataset.data)) 
#                     if digits_dataset.target[i] < 2])
#bit_target = np.array([y for y in digits_dataset.target if y < 2])
#print("Digits, 0 vs 1, should be (1.0, 0.9888888888888889)")
#print(log_reg_accuracy(bit_data, bit_target, 8))














