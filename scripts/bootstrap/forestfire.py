#!/usr/bin/env python
"""Forest fire data, from http://archive.ics.uci.edu/ml/datasets/Forest+Fires
   The burned area of forest fires in the northeast region of Portugal, including meteorological and other data."""

import numpy
import numpy.random
import urllib2
import matplotlib.pylab as plt

def forestFireData(skipZeros=True): 
    """Read the forest fire data, pull out relevant columns 
       (temperature, humidity, wind, rain, and the burned area) and return them."""
    forestfireURL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
    response = urllib2.urlopen(forestfireURL)
    data = numpy.recfromcsv(response, delimiter=',',skip_header=1)
    if skipZeros:
        data = [ record for record in data if record[12] > 0 ]

    temperature = numpy.array([ record[8] for record in data ])
    humidity    = numpy.array([ record[9] for record in data ])
    wind        = numpy.array([ record[10] for record in data ])
    rain        = numpy.array([ record[11] for record in data ])
    burnedArea  = numpy.array([ record[12] for record in data ])
    return temperature, humidity, wind, rain, burnedArea


if __name__ == "__main__":
    temp, rh, wind, rain, area = forestFireData()
    nareas = len(area)
    ntrials = 150000

    basedir = 'outputs/bootstrap/'

    n, bins, patches = plt.hist( area, 50, facecolor='red', normed=True, log=True )
    plt.xlabel('Area burned (Hectares)')
    plt.ylabel('Frequency')
    plt.savefig(basedir+'area-histogram.png', transparent=True)
    plt.close()
    numpy.random.seed(456)
    medians = numpy.array([ numpy.median(numpy.random.choice(area, nareas)) for i in xrange(ntrials) ])
    n, bins, patches = plt.hist( medians, 50, facecolor='red', normed=True )
    plt.xlabel('Median area burned (Hectares)')
    plt.ylabel('Frequency')
    plt.savefig(basedir+'median-area-histogram.png', transparent=True)
