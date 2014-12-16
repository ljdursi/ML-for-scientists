#!/usr/bin/env python
"""Demonstration of logistic regression and the iris dta set."""

import numpy 
import numpy.random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.cross_validation as cv


def getIrisData(trainfrac=0.66):
    iris = sklearn.datasets.load_iris()
    traindata, testdata, trainlabels, testlabels = cv.train_test_split(iris.data, iris.target, test_size=trainfrac)
    return traindata, trainlabels, testdata, testlabels
    

def irisProblem(trainfrac=0.66, printConfusion=False, **kwargs):
    Xtrain, ytrain, Xtest, ytest = getIrisData(trainfrac)
    
    # these are the defaults
    #model = sklearn.linear_model.LogisticRegression.(penalty='l2',C=1,fit_intercept=True,tol=None)
    model = sklearn.linear_model.LogisticRegression(**kwargs)
    model = model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)

    nwrong = sum(predictions != ytest)
    print "Misclassifications on test set: ", nwrong, "out of ", len(ytest)
    if printConfusion:
        print sklearn.metrics.confusion_matrix(ytest,predictions)
    return nwrong


def logisticDemo(gridsize=0.03, filename=None, show=True, **kwargs):
    iris = sklearn.datasets.load_iris()
    Xtrain = iris.data
    ytrain = iris.target

    model = sklearn.linear_model.LogisticRegression(**kwargs)
    model.fit(Xtrain, ytrain)

    # make a grid over the domain and colour the points by the model
    minx = min(Xtrain[:,0]); maxx = max(Xtrain[:,0])
    miny = min(Xtrain[:,1]); maxy = max(Xtrain[:,1])

    xx, yy = numpy.meshgrid(numpy.arange(minx, maxx, gridsize),
                            numpy.arange(miny, maxy, gridsize))

    x2 = numpy.zeros_like(xx.ravel()) + numpy.mean(Xtrain[:,2])
    x3 = numpy.zeros_like(xx.ravel()) + numpy.mean(Xtrain[:,3])

    Xpredict = numpy.column_stack((xx.ravel(), yy.ravel(), x2, x3))
    ypredict = model.predict(Xpredict)

    cmap_light = colors.ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Put the result into a color plot
    ypredict = ypredict.reshape(xx.shape)
    plt.pcolormesh(xx, yy, ypredict, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, cmap=cmap_bold)
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

    if show:
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
    logisticDemo(filename=base+'logistic-iris-demo.png')
