#!/usr/bin/env python
"""Demonstration of knn learning and sklearn.neighbors."""

import numpy 
import numpy.random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn
import sklearn.datasets
import sklearn.neighbors
import sklearn.metrics
import sklearn.cross_validation as cv

def digitsProblem(trainfrac=0.66, seed=None, **kwargs):
    """Use kNN to differentiate between digits in the sklearn load_digits set.
       Randomly selects trainfrac of the data for training, the rest for testing.
       If a seed is passed, will use that - keeps test/train partition the same."""
    if seed is not None:
        numpy.random.seed(seed)
    digits = sklearn.datasets.load_digits()
    traindata, testdata, trainlabels, testlabels = cv.train_test_split(digits.data, digits.target, test_size=trainfrac)

    model = sklearn.neighbors.KNeighborsClassifier(**kwargs)
    model.fit(traindata, trainlabels)

    predicted = model.predict(testdata)
    nwrong =sum(predicted != testlabels)
    print 'Number mismatched: ', nwrong, '/', len(predicted)
    print sklearn.metrics.confusion_matrix(testlabels, predicted)

def digitsPlot(filename=None):
    digits = sklearn.datasets.load_digits()
    plt.gray() 
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(-digits.images[i]) 
    outputPlot(filename)
    
def outputPlot(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, transparent=True)
        plt.close()

if __name__ == '__main__':
    base='outputs/classification/'
    digitsPlot(base+'digits.png')
    digitsProblem()
