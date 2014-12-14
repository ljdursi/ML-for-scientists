#!/usr/bin/env python

import numpy
import statsmodels.api as sm
import matplotlib.pylab as plt
import biasvariance as bv

def lowessFit(x,y,**kwargs):
    """Use statsmodels to do a lowess fit to the given data.
       Returns the x and fitted y."""
    fit = sm.nonparametric.lowess(y,x,**kwargs)
    xlowess = fit[:,0]
    ylowess = fit[:,1]
    return xlowess, ylowess

if __name__ == "__main__":
    npts = 100
    numpy.random.seed(123)
    base="./outputs/nonparametric/"

    x, y = bv.noisyData(npts)
    xlowess, ylowess = lowessFit(x,y,frac=0.1,it=5)
    plt.plot(x, y, 'ro')
    plt.plot(xlowess, ylowess, 'g-')
    plt.savefig(base+"lowess-fit.png", transparent=True)
