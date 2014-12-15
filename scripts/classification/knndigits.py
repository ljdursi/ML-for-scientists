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

def digitsProblem(trainfrac=0.66, seed=None, **kwargs):
    """Use kNN to differentiate between digits in the sklearn load_digits set.
       Randomly selects trainfrac of the data for training, the rest for testing.
       If a seed is passed, will use that - keeps test/train partition the same."""
    if seed is not None:
        numpy.random.seed(seed)
    digits = sklearn.datasets.load_digits()
    n = len(digits.target)
    idxs = numpy.arange(n)
    numpy.random.shuffle(idxs)
    trainidxs = idxs[:int(trainfrac*n)]
    testidxs  = idxs[int(trainfrac*n):]

    model = sklearn.neighbors.KNeighborsClassifier(**kwargs)
    model.fit(digits.data[trainidxs],digits.target[trainidxs])

    predicted = model.predict(digits.data[testidxs])
    expected = digits.target[testidxs]
    nwrong =sum(predicted != expected)
    print 'Number mismatched: ', nwrong, '/', len(predicted)
    print sklearn.metrics.confusion_matrix(expected, predicted)

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
