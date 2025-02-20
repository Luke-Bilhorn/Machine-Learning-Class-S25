from knn import KNN_Classifier
import sklearn 
import numpy as np

#filename = '/homes/tvandrun/Public/datasets/breastcancer.csv'
#bc_readings = np.loadtxt(filename, delimiter=',',
#                         skiprows=1, usecols=range(2,32)) 
#bc_diagnoses = np.loadtxt(filename, delimiter=',',
#                          skiprows=1, usecols=1,dtype='str')

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
bc_readings = cancer.data
bc_diagnoses = cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bc_readings, bc_diagnoses, random_state=1)

print("L1 distance")
classy = KNN_Classifier(X_train, y_train, 2, metric="L1")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 5, metric="L1")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 10, metric="L1")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

print("L2 distance")
classy = KNN_Classifier(X_train, y_train, 2, metric="L2")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 5, metric="L2")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 10, metric="L2")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

print("L3 distance")
classy = KNN_Classifier(X_train, y_train, 2, metric="L3")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 5, metric="L3")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 10, metric="L3")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

print("Mahalanobis distance")
classy = KNN_Classifier(X_train, y_train, 2, metric="M")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 5, metric="M")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 10, metric="M")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

print("Canberra distance")
classy = KNN_Classifier(X_train, y_train, 2, metric="C")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 5, metric="C")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))

classy = KNN_Classifier(X_train, y_train, 10, metric="C")
results = classy.classify(X_test)
print(str(np.mean([results[i] == y_test[i] for i in range(len(results))])))


