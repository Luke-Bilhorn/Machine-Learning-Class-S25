from knn import KNN_Classifier
import sklearn 
import numpy as np

from sklearn.datasets import load_digits
digits_dataset = load_digits()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits_dataset.data, digits_dataset.target, random_state=1)

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



