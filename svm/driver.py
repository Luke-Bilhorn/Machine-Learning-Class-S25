import svm
import numpy as np
from kernel import linear
from kernel import rbf
from kernel import make_poly_kernel
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris


flags = sys.argv[1:]
if len(flags) == 0 :
    flags = ['-t']

if '-t' in flags or '-all' in flags :
    print("Trivial artificial data")
    X = np.array([[1.0,0.0],[2.0,0.0],[3.0,0.0],[-1.0,0.0],[-2.0,0.0],[-3.0,0.0]])
    y = np.array([1.0,1.0,1.0,-1.0,-1.0,-1.0])
    svm_trivial = svm.train(X, y, linear)
    pred_train = svm_trivial.classify(X)
    X_test = np.array([[4.0, 5.0],[1.5,0.0],[7.5,10.0],[-5.0,0.0],[-2.0,-4.0],[-2.0,4.0]])
    y_test = np.array([1.0,1.0,1.0,-1.0,-1.0,-1.0])
    pred_test = svm_trivial.classify(X_test)
    print("Training set accuracy: ", np.mean(pred_train == y))
    print("Test set accuracy: ", np.mean(pred_test == y_test))


if '-ls' in flags or '-all' in flags:
    print("Linearly separable artificial data")
    X_train = np.array( [[-1.0, 0.0], [-2.0, 1.0], [-2.0, -1.0], [-3.0, 0.0], [1.0, 0.0], [2.0, 1.0], [2.0, -1.0], [3.0, 0.0]])
    y_train = np.array([-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0])
        
    X_test = np.array( [[-2.0, 0.0], [-3.0, 1.0], [-3.0, -1.0], [2.0, 0.0], [3.0, 1.0], [3.0, -1.0]])
    y_test = np.array( [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        
        
    clsfyr = svm.train(X_train, y_train, linear)
    pred_train = clsfyr.classify(X_train)
    pred_test = clsfyr.classify(X_test)
    print("Training set accuracy: ", np.mean(pred_train == y_train))
    print("Test set accuracy: ", np.mean(pred_test == y_test))


if '-ns' in flags or '-all' in flags:
    print("Non linearly-separable artificial data")
        
    X_train = np.array( [[-1.0, 0.0], [-2.0, 2.0], [-2.0, 1.0], [-2.0, 0.0], [-2.0, -1.0], [-2.0, -2.0],  [1.0, -1.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 2.0], [2.0, 1.0], [2.0, 0.0], [2.0, -1.0], [2.0, -2.0]])
    y_train = np.array( [-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
        
    X_test = np.array( [[-3.0, 2.0], [-3.0, 1.0], [-3.0, 0.0], [-3.0, -1.0], [-3.0, -2.0], [3.0, 2.0], [3.0, 1.0], [3.0, 0.0], [3.0, -1.0], [3.0, -2.0]])
    y_test = np.array( [-1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0])
        
        
        
    clsfyr = svm.train(X_train, y_train, make_poly_kernel(2), 1)
    results = clsfyr.classify(X_test)
    pred_train = clsfyr.classify(X_train)
    pred_test = clsfyr.classify(X_test)
    print("Training set accuracy: ", np.mean(pred_train == y_train))
    print("Test set accuracy: ", np.mean(pred_test == y_test))
    
if '-bc' in flags or '-all' in flags:

    print()
    print("BREAST CANCER")
        
        
    bc_dataset = load_breast_cancer()
    bc_targets = np.empty(bc_dataset.target.shape[0])
    for j in range(bc_dataset.target.shape[0]):
        if bc_dataset.target[j] == 0:
            bc_targets[j] = -1.0
        else:
            bc_targets[j] = 1.0

    bc_x_train, bc_x_test, bc_y_train, bc_y_test = train_test_split(bc_dataset.data,bc_targets)
                        
    print("Soft Classification: Low C")
    svm_bc_soft_low = svm.train(bc_x_train, bc_y_train, linear, C = 0.1)
    bc_y_predict = svm_bc_soft_low.classify(bc_x_test)
    print("Test set accuracy: ", np.mean(bc_y_predict == bc_y_test))
    bc_y_predict = svm_bc_soft_low.classify(bc_x_train)
    print("Training set accuracy: ", np.mean(bc_y_predict == bc_y_train))
        
    print()
    print("Soft Classification: High C")
    svm_bc_soft_high = svm.train(bc_x_train, bc_y_train, linear, C = 10)
    bc_y_predict = svm_bc_soft_high.classify(bc_x_test)
    print("Test set accuracy: ", np.mean(bc_y_predict == bc_y_test))
    bc_y_predict = svm_bc_soft_high.classify(bc_x_train)
    print("Training set accuracy: ", np.mean(bc_y_predict == bc_y_train))

if '-i' in flags or '-all' in flags:

    print()
    print("IRIS")
    i_dataset = load_iris()
    for j in range(i_dataset.target.shape[0]):
        if i_dataset.target[j] == 0:
            i_dataset.target[j] = -1.0
        elif i_dataset.target[j] == 1:
            i_dataset.target[j] = 1.0
        elif i_dataset.target[j] == 2:
            i_dataset.target[j] = 1.0

    i_x_train, i_x_test, i_y_train, i_y_test = train_test_split(i_dataset.data, i_dataset.target)
        
    print("Hard Classification")
    svm_i = svm.train(i_x_train, i_y_train, linear)
    i_y_predict = svm_i.classify(i_x_test)        
    print("Test set accuracy: ", np.mean(i_y_predict == i_y_test))
    i_y_predict = svm_i.classify(i_x_train)
    print("Training set accuracy:",  np.mean(i_y_predict == i_y_train))
    
    print()
    print("Soft Classification: Low C")
    svm_i_soft_low = svm.train(i_x_train, i_y_train, linear, C = 0.1)
    i_y_predict = svm_i_soft_low.classify(i_x_test)
    print("Test set accuracy: ", np.mean(i_y_predict == i_y_test))
    i_y_predict = svm_i_soft_low.classify(i_x_train)
    print("Training set accuracy:",  np.mean(i_y_predict == i_y_train))
    
    print()
    print("Soft Classification: High C")
    svm_i_soft_high = svm.train(i_x_train, i_y_train, linear, C = 10)
    i_y_predict = svm_i_soft_high.classify(i_x_test)
    print("Test set accuracy: ", np.mean(i_y_predict == i_y_test))
    i_y_predict = svm_i_soft_high.classify(i_x_train)
    print("Training set accuracy:",  np.mean(i_y_predict == i_y_train))
    
if '-hd' in flags or '-all' in flags:

    print()
    print("HEART DISEASE")
    hd_dataset = np.genfromtxt("heart.csv", delimiter = ',')
    hd_dataset = hd_dataset[np.arange(1,302),:]
    
    hd_targets = np.empty(hd_dataset[:,13].shape[0])
    for j in range(hd_dataset[:,13].shape[0]):
        if hd_dataset[j,13] == 0:
            hd_targets[j] = -1.0
        else:
            hd_targets[j] = 1.0
    hd_x_train, hd_x_test, hd_y_train, hd_y_test = train_test_split(hd_dataset[:,np.arange(0,12)], (hd_targets))
                 
    
    print("Soft Classification: Low C")
    svm_hd_soft_low = svm.train(hd_x_train, hd_y_train, linear, C = 0.1)
    hd_y_predict = svm_hd_soft_low.classify(hd_x_test)
    print("Test set accuracy: ", np.mean(hd_y_predict == hd_y_test))
    hd_y_predict = svm_hd_soft_low.classify(hd_x_train)
    print("Training set accuracy: ", np.mean(hd_y_predict == hd_y_train))
    print()


    print("Soft Classification: High C")
    svm_hd_soft_high = svm.train(hd_x_train, hd_y_train, linear, C = 10)
    hd_y_predict = svm_hd_soft_high.classify(hd_x_test)
    print("Test set accuracy: ", np.mean(hd_y_predict == hd_y_test))
    hd_y_predict = svm_hd_soft_high.classify(hd_x_train)
    print("Training set accuracy: ", np.mean(hd_y_predict == hd_y_train))
        
 



