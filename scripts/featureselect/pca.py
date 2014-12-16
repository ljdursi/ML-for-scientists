#!/usr/bin/env python
"""Example of PCA and plot."""

import numpy
import numpy.random
import matplotlib.pylab as plt
import sklearn.decomposition

def normalComponents(mean=numpy.array([0.,0.]), sigma=numpy.array([1.,3.]), slope=2, npts=100):
    nptseach = npts/2
    cov = numpy.diag(sigma)

    x = numpy.linspace(-10.,10.,npts)
    y = slope*x
    X = numpy.column_stack((x,y))
    X[:,0] = X[:,0] + numpy.random.normal(mean[0],sigma[0],npts)
    X[:,1] = X[:,1] + numpy.random.normal(mean[1],sigma[1],npts)
    return X

def PCAPlot(filename=None, **kwargs):
    X = normalComponents(npts=500,**kwargs)
    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c='red', alpha=0.75) 

    pca = sklearn.decomposition.PCA(n_components=2)
    pca = pca.fit(X)
    Z = pca.transform(X)
    rot = pca.components_
    rot = rot.T
    x1 = numpy.dot(rot, numpy.array([10.,0.]))
    x2 = numpy.dot(rot, numpy.array([0.,10.]))
    plt.plot([0.,x1[0]], [0.,x1[1]],'g-',lw=3)
    plt.plot([0.,x2[0]], [0.,x2[1]],'g-',lw=3)

    plt.subplot(1,2,2)
    plt.scatter(Z[:,0], Z[:,1], c='red', alpha=0.75) 
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
    PCAPlot(base+"pca-demo.png")
