#!/usr/bin/env python
"""Example of lasso regression and a coefficient plot."""

import numpy
import numpy.random
import matplotlib.pylab as plt
import sklearn
import sklearn.linear_model
import sklearn.cross_validation as cv
import sklearn.metrics
import sklearn.datasets

def getDiabetesData():
    diabetes = sklearn.datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    # standardize the data:
    X = X/X.std(axis=0)
    return X,y

def lassoFit(X,y,alpha=1.,trainfrac=0.66, random_state=None):
    X, y = getDiabetesData()
    Xtrain, Xtest, ytrain, ytest = cv.train_test_split(X, y, test_size=trainfrac, random_state=random_state)

    lasso = sklearn.linear_model.Lasso(alpha=alpha)
    lasso = lasso.fit(Xtrain, ytrain)
    predicted = lasso.predict(Xtest)
    mse = sklearn.metrics.mean_squared_error(ytest, predicted)
    return mse

def lassoPathPlot(X,y,eps=5.e-3,filename=None):
    alphas_lasso, coefs_lasso, _ = sklearn.linear_model.lasso_path(X, y, eps, fit_intercept=False)
    plt.plot(-numpy.log10(alphas_lasso), coefs_lasso.T)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    outputPlot(filename)

def outputPlot(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, transparent=True)
        plt.close()

if __name__ == "__main__":
    numpy.random.seed(123)
    base="./outputs/featureselect/"
    X,y = getDiabetesData()
    lassoPathPlot(X,y,filename=base+"lasso-coeffs.png")
