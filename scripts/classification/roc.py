#!/usr/bin/env python
"""Demonstration of logistic regression and the iris dta set."""

import numpy 
import numpy.random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.cross_validation as cv


def getIrisDataBinary(trainfrac=0.66, addNoise=False):
    """Get a binary-classification equivalent of the Iris problem - get rid of class 0.
       Also make more data and make the problem harder by adding noise."""
    iris = sklearn.datasets.load_iris()

    notone = numpy.where(iris.target != 0)
    iris.data = iris.data[notone]
    iris.target = iris.target[notone] - 1

    if addNoise:
        iris.data = numpy.repeat(iris.data,5,axis=0)
        iris.target = numpy.repeat(iris.target,5,axis=0)
        n = len(iris.target)
        for var in range(4):
            sigma = numpy.sqrt(numpy.var(iris.data[:,var]))
            iris.data[:,var] = iris.data[:,var] + numpy.random.randn(n)*sigma

    traindata, testdata, trainlabels, testlabels = cv.train_test_split(iris.data, iris.target, test_size=trainfrac)
    return traindata, trainlabels, testdata, testlabels

def calcROC(Xtrain, ytrain, Xtest, ytest, **kwargs):
    model = sklearn.linear_model.LogisticRegression(**kwargs)
    model = model.fit(Xtrain, ytrain)
    probs = model.predict_proba(Xtest)

    fpr, tpr, _  = sklearn.metrics.roc_curve(ytest, probs[:,1])
    return fpr, tpr

def irisROC(trainfrac=0.66, filename=None, **kwargs):
    # "easy" data
    Xtrain, ytrain, Xtest, ytest = getIrisDataBinary(trainfrac, addNoise=False, **kwargs)
    fpr, tpr = calcROC(Xtrain, ytrain, Xtest, ytest)
    plt.plot(fpr, tpr, 'g-', lw=2, label='Easier')

    # "hard" data
    Xtrain, ytrain, Xtest, ytest = getIrisDataBinary(trainfrac, addNoise=True, **kwargs)
    fpr, tpr = calcROC(Xtrain, ytrain, Xtest, ytest)
    plt.plot(fpr, tpr, 'r-', lw=2, label='Harder')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    outputPlot(filename)

def outputPlot(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, transparent=True)
        plt.close()

if __name__ == '__main__':
    numpy.random.seed(123)
    base = 'outputs/classification/'
    irisROC(filename=base+'roc.png')
