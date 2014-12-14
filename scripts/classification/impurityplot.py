#!/usr/bin/env python
"""Makes plots of gini index, entropy"""

import matplotlib.pylab as plt
import numpy

if __name__ == "__main__":
    base = 'outputs/classification/'
    ps = numpy.linspace(0.001,0.999,100)
    gini = ps*(1.-ps)
    entropy = -ps*numpy.log(ps) - (1.-ps)*numpy.log(1.-ps)

    plt.plot(ps,gini/max(gini),'g-',lw=2,label="Gini Index")
    plt.plot(ps,entropy/max(entropy),'r-',lw=2,label="Entropy")
    plt.legend()
    plt.xlabel("Classification Probability")
    plt.ylabel("Impurity of group (normalized)")
    plt.savefig(base+'impurity-plots.png',transparency=True)
