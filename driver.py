from gmm import train
from gmm import MixedGaussian
from gmm import ave_log_like
from datetime import datetime
import sys

import pandas as pd
import numpy as np

def run_on_data(data, k, direct=None) :
    if direct :
        print("Average log likelihood, direct: " + str(ave_log_like(direct, data)))
    trained = train(data, k)
    print("Average log likelihood, trained: " + str(ave_log_like(trained, data)))
    

flags = sys.argv[1:]
if len(flags) == 0 :
    flags = ['-2c']

print(datetime.now())

if '-2c' in flags or '-all' in flags:
    print("Two compnents  (generated)")
    twocomp = np.loadtxt("twocomponents.csv", delimiter=',')[:500]
    run_on_data(twocomp, 2, MixedGaussian([.25,.75], [2.0,3.5], [.25, .5]))
    print(datetime.now())

if '-2cc' in flags or '-all' in flags:
    print("Two compnents close (generated)")
    twocomp_close = np.loadtxt("twocomponents_close.csv", delimiter=',')[:500]
    run_on_data(twocomp_close, 2, MixedGaussian([.25,.75], [2.0,2.5], [.25, .5]))
    print(datetime.now())

if '-4c' in flags  or '-all' in flags:
    print("Four compnents (generated)")
    fourcomp = np.loadtxt("fourcomponents.csv", delimiter=',')[:500]
    run_on_data(fourcomp, 4,
                MixedGaussian([.25, .15, .05, .55], [2.5, 3.75, 6.25, 7.0], [.1, .5, 1.0, .34]))
    print(datetime.now())

if '-5c' in flags  or '-all' in flags:
    print("Five compnents (generated)")
    fivecomp = np.loadtxt("fivecomponents.csv", delimiter=',')[:500]
    run_on_data(fivecomp, 5,
                MixedGaussian([.15, .2, .1, .4, .15], [1.,2.,3.,4.,5.], [.1, .15, .05, .2, .4]))
    print(datetime.now())

if '-hw' in flags or '-all' in flags:
    heightweight = pd.read_csv("weight-height.csv")
    weight = heightweight["Weight"]
    height = heightweight["Height"]

    print("Height")
    run_on_data(height, 2,
                MixedGaussian([.5, .5], [69.026, 63.709], [2.863, 2.696]))
    print(datetime.now())

    print("Weight")
    run_on_data(weight, 2,
                MixedGaussian([.5, .5], [187.021,135.86], [19.779, 19.021]))
    print(datetime.now())

if '-of' in flags or '-all' in flags:
    oldfaithful = pd.read_csv("oldfaithful.csv")
    eruptions = oldfaithful["eruptions"]
    waiting = oldfaithful["waiting"]

    print("Eruptions")
    run_on_data(eruptions, 2)
    print(datetime.now())

    print("Waiting")
    run_on_data(waiting, 2)
    print(datetime.now())

if '-iris' in flags or '-all' in flags:
    from sklearn.datasets import load_iris
    iris_dataset = load_iris().data
    print("Petal length")
    run_on_data( iris_dataset[:,2], 3,
                 MixedGaussian([.333333,.333333,.333333],
                               [1.462,4.26,5.552], [0.172,0.465,0.546]))
    print(datetime.now())

    print("Petal width")
    run_on_data( iris_dataset[:,3], 3,
                 MixedGaussian([.333333,.333333,.333333],
                               [0.246, 1.326, 2.026], [0.104, 0.196, 0.546]))

    print(datetime.now())




