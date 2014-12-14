#!/usr/bin/env python

import numpy
import statsmodels.nonparametric.api as smnparam
import matplotlib.pylab as plt
import biasvariance as bv

def kernelFit(x,y,**kwargs):
    """Use statsmodels to do a kernel regression to the given data.
       Returns the x and fitted y."""

    # continuous ('c') variables, locally linear 'll' as vs locally const
    model = smnparam.KernelReg(y,x,var_type='c',reg_type='ll', **kwargs)
    ykernel, mfx = model.fit()
    return x, ykernel

def tricubeKernel(xcenter, bandwidth, x):
    """Applies the tricubic kernel function to an array of x values; returns the function values."""
    h = abs(x - xcenter)/bandwidth
    y = (1. - h**3.)**3
    y[h>1.] = 0.
    return y

def gaussianKernel(xcenter, bandwidth, x):
    """Applies the normal kernel function to an array of x values; returns the function values."""
    h = abs(x - xcenter)/bandwidth
    y = numpy.exp(-h*h/2.)
    return y

def kernelDemo(x=[-1.,0.,+1.,+2.], y=[1.,2.,3.,4.], bandwidth=.4):
    """A very simple demo of kernel regression with user-supplied bandwidth."""
    plt.plot(x,y,'ro')
    plt.plot(x,y,'g--')

    kernel = gaussianKernel
    dx = x[1]-x[0]
    allxs = numpy.linspace(min(x)-dx/2.,max(x)+dx/2.,300)
    allys = numpy.zeros_like(allxs)
    for cx,cy in zip(x,y):
        locxs = numpy.linspace(cx-bandwidth,cx+bandwidth,100)
        locys = cy*kernel(cx,bandwidth,locxs)
        plt.plot(locxs, locys, 'r-')
        allys = allys + cy*kernel(cx, bandwidth, allxs)
    plt.plot(allxs, allys, 'g-', lw=2)


if __name__ == "__main__":
    npts = 100
    numpy.random.seed(123)
    base="./outputs/nonparametric/"

    x, y = bv.noisyData(npts)
    xkernel, ykernel = kernelFit(x,y)
    plt.plot(x, y, 'ro')
    plt.plot(xkernel, ykernel, 'g-', lw=2)
    plt.savefig(base+"kernel-fit.png", transparent=True)
    plt.close()

    kernelDemo()
    plt.savefig(base+"kernel-demo.png", transparent=True)
    plt.close()
