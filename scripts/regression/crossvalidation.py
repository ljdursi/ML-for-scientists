#!/usr/bin/env python

import numpy
import numpy.random
import matplotlib.pylab as plt

import biasvariance as bv

def estimateError(x, y, d, kfolds=10):
    """Estimate the error in fitting data (x,y) with a polynomial of degree d via cross-validation.
       Returns RMS error."""
    n = len(x)

    idxes = numpy.arange(n)
    numpy.random.shuffle(idxes)
    test  = numpy.zeros((n),dtype=numpy.bool)

    err = 0.
    for i in range(kfolds):
        startv = n*i/kfolds; endv = n*(i+1)/kfolds
        test[idxes[startv:endv]] = True

        test_x  = x[test];  test_y  = y[test]
        train_x = x[-test]; train_y = y[-test]

        p = numpy.polyfit(train_x, train_y, d)
        fitf = numpy.poly1d(p)

        err = err + sum((test_y - fitf(test_x))**2)
        test[:] = False
    return numpy.sqrt(err)


def chooseDegree(npts, mindegree=0, maxdegree=20, filename=None):
    """Gets noisy data, uses cross validation to estimate error, and fits new data with best model."""
    x, y = bv.noisyData(npts)
    degrees = numpy.arange(mindegree,maxdegree+1)
    errs = numpy.zeros_like(degrees,dtype=numpy.float)
    for i,d in enumerate(degrees):
        errs[i] = estimateError(x, y, d)

    plt.subplot(1,2,1)
    plt.plot(degrees,errs,'bo-')
    plt.xlabel("Degree")
    plt.ylabel("CV Error")

    besti = numpy.argmin(errs)
    bestdegree = degrees[besti]

    plt.subplot(1,2,2)
    x2, y2 = bv.noisyData(npts)
    plt.plot(x2,y2,'ro')
    xs = numpy.linspace(min(x),max(x),150)
    fitf = numpy.poly1d(numpy.polyfit(x2,y2,bestdegree))
    plt.plot(xs,fitf(xs),'g-')
    plt.xlim((bv.MIN,bv.MAX))
    plt.ylim((-2.,2.))
    plt.suptitle('Selected Degree '+str(bestdegree))
    bv.outputPlot(filename)


if __name__ == "__main__":
    numpy.random.seed(123)
    base="./assets/img/crossvalidation/"
    chooseDegree(50,filename=base+'CV-polynomial.png')
