#!/usr/bin/env python

import numpy
import numpy.random
import matplotlib.pylab as plt

MIN=-1.5
MAX=+1.5

def noisyData(npts=40,slope=1.0,freq=2.*numpy.pi,noise_amp=0.125,x_noise_amp=0.1):
    x = numpy.linspace(MIN,MAX,npts)
    x = x + x_noise_amp*numpy.random.rand(npts)
    y = slope*x + numpy.sin(freq*x) + noise_amp*numpy.random.randn(npts)
    return x,y

def noisyDataPolyFit(degree, x=None, y=None, **kwargs):
    returnxy = False
    if x is None:
        x, y = noisyData(**kwargs)
        returnxy = True
    p = numpy.polyfit(x, y, degree)
    if returnxy:
        return x, y, p
    else:
        return p

def noisyDataPolyPlot(degree, filename=None, **kwargs):
    x, y, p = noisyDataPolyFit(degree, **kwargs)
    fitf = numpy.poly1d(p)
    plt.plot(x,y,'ro')
    plt.plot(x,fitf(x),'g-')
    outputPlot(filename)

def varianceDemo(degree, ntrials, filename1=None, filename2=None, **kwargs):
    pts = numpy.linspace(MIN,MAX,100)
    zeropreds = numpy.zeros((ntrials))
    for i in range(ntrials):
        x, y, p = noisyDataPolyFit(degree, **kwargs)
        fitf = numpy.poly1d(p)
        plt.plot(pts,fitf(pts),'g-')
        zeropreds[i] = fitf(0.)
    plt.plot(x,y,'ro')
    outputPlot(filename1)
    x, y = noisyData(npts=3,noise_amp=0.,x_noise_amp=0.)
    zerotrue = y[1]

    n, bins, patches = plt.hist(zeropreds, ntrials/10)
    height = max(n)*3/2
    line = plt.plot([zerotrue,zerotrue],[0,height],'r-')
    plt.setp(line,linewidth=2)
    outputPlot(filename2)

def errorVsDegree(ndegrees, npts=40, filename=None, **kwargs):
    degrees = numpy.array(range(ndegrees),dtype=float)
    errs = numpy.zeros_like(degrees)
    ntrials=10
    for i, deg in enumerate(degrees):
        for trial in range(ntrials):
            x, y =  noisyData(npts=npts,**kwargs)
            p = noisyDataPolyFit(deg, x, y)

            x2, y2 =  noisyData(npts=npts, **kwargs)
            fitf = numpy.poly1d(p)

            errs[i] = errs[i] + sum((y2-fitf(x2))**2)
        errs[i] = errs[i]/ntrials
    plt.semilogy(degrees,errs,'go-')
    plt.xlabel('Degree')
    plt.ylabel('Squared L2 error')
    outputPlot(filename)

def outputPlot(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, transparent=True)

