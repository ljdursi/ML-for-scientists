#!/usr/bin/env python
"""Demonstration of logistic regression."""

import numpy 
import numpy.random
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

def logitDemo(npts=6, midpt=0., sep=1., filename=None):
    """Use linear and logistic regression to seperate points, and plot results."""
    neach = npts/2
    categorya = midpt-numpy.arange(neach)-sep/2.
    categoryb = midpt+numpy.arange(neach)+sep/2.
    X = numpy.concatenate( (categorya, categoryb) )
    X = X.reshape(neach*2,1)
    y = numpy.array([0.]*neach + [1.]*neach)
    plt.plot(X,y,'ro')

    xpred = numpy.linspace(min(X)-1,max(X)+1,100).reshape(100,1)

    lin = sklearn.linear_model.LinearRegression()
    lin = lin.fit(X,y)

    ypred = lin.predict(xpred)
    plt.plot(xpred,ypred,'g--', lw=2, label='Linear')

    logistic = sklearn.linear_model.LogisticRegression()
    logistic = logistic.fit(X,y)
    
    ypred = logistic.predict(xpred)
    plt.plot(xpred,ypred,'r--', lw=1, label='Logistic Prediction')

    logisticf = numpy.exp(logistic.coef_[0]*(xpred[:,0]-logistic.intercept_))
    ypred = logisticf / (1. + logisticf)
    plt.plot(xpred,ypred,'g-', lw=2, label='Logit Function')

    plt.legend(loc='upper left')
    outputPlot(filename)

def outputPlot(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, transparent=True)
        plt.close()

if __name__ == '__main__':
    base='outputs/classification/'
    logitDemo(filename=base+'logistic-demo.png')
