#!/usr/bin/env python

import numpy
import numpy.random
import matplotlib.pylab as plt

MIN=-1.0
MAX=+1.0

def noisyData(npts=40,slope=1.0,freq=2.*numpy.pi,noise_amp=0.125,x_noise_amp=0.1,returnModel=False):
    """ Generate a noisy titled sinusoid, and optionally return the noise-free model."""
    xmodel = numpy.linspace(MIN,MAX,npts)
    x = xmodel + x_noise_amp*numpy.random.rand(npts)
    ymodel = slope*xmodel + numpy.sin(freq*x)
    y = slope*x + numpy.sin(freq*x) + noise_amp*numpy.random.randn(npts)

    returns = (x,y)
    if returnModel:
        returns = returns + (xmodel, ymodel,)
    return returns

def noisyDataPolyFit(degree, x=None, y=None, returnError=False, **kwargs):
    """ Generate or take noisy data, fit it with a polynomial. 
        Returns the data if generated, the fit, and optionally the error."""
    returnxy = False
    if x is None:
        x, y, xmodel, ymodel = noisyData(returnModel=True,**kwargs)
        returnxy = True
    p = numpy.polyfit(x, y, degree)
    returns = (p,)
        
    if returnxy:
        returns = returns + (x,y)
    if returnError:
        fitp = numpy.poly1d(p)
        err = sum((ymodel - fitp(xmodel))**2)
        returns = returns + (err,)
    
    return returns

def noisyDataPolyPlot(degree, filename=None, **kwargs):
    """ Generate data, fit, and plot."""
    p, x, y = noisyDataPolyFit(degree, **kwargs)
    fitf = numpy.poly1d(p)
    plt.plot(x,y,'ro')
    plt.plot(x,fitf(x),'g-')
    outputPlot(filename)

def errorVsDegree(ndegrees, npts=40, filename=None, **kwargs):
    degrees = numpy.array(range(ndegrees),dtype=float)
    errs = numpy.zeros_like(degrees)
    ntrials=10
    for i, deg in enumerate(degrees):
        for trial in range(ntrials):
            p, x, y, err = noisyDataPolyFit(deg, returnError=True)
            errs[i] = errs[i] + err
        errs[i] = errs[i]/ntrials
    plt.semilogy(degrees,errs,'go-')
    plt.xlabel('Degree')
    plt.ylabel('Squared L2 error')
    outputPlot(filename)

def varianceDemo(degree, ntrials, filename=None, **kwargs):
    pts = numpy.linspace(MIN,MAX,100)
    zeropreds = numpy.zeros((ntrials))

    plt.subplot(2,1,1)
    for i in range(ntrials):
        p, x, y = noisyDataPolyFit(degree, **kwargs)
        fitf = numpy.poly1d(p)
        plt.plot(pts,fitf(pts),'g-')
        zeropreds[i] = fitf(0.)
    plt.plot(x,y,'ro')
    plt.xlim([MIN,MAX])
    plt.ylim([-2.,2.])
    
    plt.subplot(2,1,2)
    x, y = noisyData(npts=3,noise_amp=0.,x_noise_amp=0.)
    zerotrue = y[1]
    n, bins, patches = plt.hist(zeropreds, ntrials/20)
    height = max(n)*5/4
    line = plt.plot([zerotrue,zerotrue],[0,height],'r-')
    plt.setp(line,linewidth=2)

    mean    = numpy.mean(zeropreds)
    sd      = numpy.sqrt(numpy.var(zeropreds))
    height2 = max(n)
    plt.annotate('Bias', xy=(mean, 0.9*height), xytext=(0, 0.9*height), xycoords='data', 
            va='center', arrowprops={'facecolor':'red', 'shrink':0.05})
    line = plt.plot([mean-2*sd,mean+2*sd],[height2/3.,height2/3.],'g-')
    plt.setp(line,linewidth=7)
    if mean < 0:
        txtpos = mean-3.*sd
        align = 'left'
    else:
        txtpos = mean+3.*sd
        align = 'right'
    plt.text(txtpos, height2*9./24., 'Variance', ha=align, va='bottom')
    plt.suptitle('Polynomial, degree '+str(degree))
    plt.xlim((-0.3,0.3))
    outputPlot(filename)

def outputPlot(filename=None):
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, transparent=True)
        plt.close()

if __name__ == "__main__":
    numpy.random.seed(123)
    base="./assets/img/bias-variance/"
    varianceDemo(0, 1000, base+'const-bias-variance.png')
    varianceDemo(1, 1000, base+'lin-bias-variance.png')
    varianceDemo(6, 1000, base+'seventh-bias-variance.png')
    varianceDemo(2, 1000, base+'tenth-bias-variance.png')
    varianceDemo(20,1000, base+'twentyth-bias-variance.png')
    errorVsDegree(20, 40, base+'error-vs-degree.png')
