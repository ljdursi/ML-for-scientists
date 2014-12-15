#!/usr/bin/env python
"""Demonstration of knn learning and sklearn.neighbors."""

import numpy 
import numpy.random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn
import sklearn.neighbors

def twoNormalComponents(x1=numpy.array([-1.,-1.]), sigma1=1.5, x2=numpy.array([+1.,+1]), sigma2=1.5, npts=100):
    nptseach = npts/2
    cov1 = numpy.array([[sigma1,0.],[0.,sigma1]])
    cov2 = numpy.array([[sigma2,0.],[0.,sigma2]])

    X1 = numpy.random.multivariate_normal(x1,cov1,nptseach)
    X2 = numpy.random.multivariate_normal(x2,cov2,nptseach)

    X = numpy.concatenate((X1,X2))
    y = numpy.concatenate( (numpy.array([0]*nptseach), numpy.array([1]*nptseach)) )
    return X, y


def knnDemo(npts=100, nneigh=5, gridsize=0.03, filename=None, show=True, **kwargs):
    X,y = twoNormalComponents(npts=npts, **kwargs)

    model = sklearn.neighbors.KNeighborsClassifier(nneigh)
    model.fit(X,y)

    # make a grid over the domain and colour the points by the model
    minx = min(X[:,0]); maxx = max(X[:,0])
    miny = min(X[:,1]); maxy = max(X[:,1])

    xx, yy = numpy.meshgrid(numpy.arange(minx, maxy, gridsize),
                            numpy.arange(miny, maxy, gridsize))
    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])

    cmap_light = colors.ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = colors.ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

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
    knnDemo(filename=base+'knn-demo.png')

    plt.subplot(2,2,1)
    numpy.random.seed(123)
    knnDemo(nneigh=1,show=False)
    plt.title('k = 1')
    plt.subplot(2,2,2)
    numpy.random.seed(123)
    knnDemo(nneigh=3,show=False)
    plt.title('k = 3')
    plt.subplot(2,2,3)
    numpy.random.seed(123)
    knnDemo(nneigh=7,show=False)
    plt.title('k = 7')
    plt.subplot(2,2,4)
    numpy.random.seed(123)
    knnDemo(nneigh=13,show=False)
    plt.title('k = 13')
    outputPlot(filename=base+'knn-vary-k.png')

    plt.subplot(2,2,1)
    numpy.random.seed(123)
    knnDemo(nneigh=1,show=False)
    plt.subplot(2,2,2)
    numpy.random.seed(124)
    knnDemo(nneigh=3,show=False)
    plt.subplot(2,2,3)
    numpy.random.seed(125)
    knnDemo(nneigh=7,show=False)
    plt.subplot(2,2,4)
    numpy.random.seed(126)
    knnDemo(nneigh=13,show=False)
    outputPlot(filename=base+'knn-variance.png')
