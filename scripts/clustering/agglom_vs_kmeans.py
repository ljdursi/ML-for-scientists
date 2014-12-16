#!/usr/bin/env python
"""Agglomerative hierarchical vs kmeans clustering -- after the sklearn example"""

import matplotlib.pyplot as plt
import numpy 

import sklearn.cluster 
import sklearn.neighbors 

def sampleData(nsamples=1500, seed=0):
    n_samples = 1500
    numpy.random.seed(seed)
    t = 1.5 * numpy.pi * (1 + 3 * numpy.random.rand(1, n_samples))
    x = t * numpy.cos(t)
    y = t * numpy.sin(t)
    X = numpy.concatenate((x, y))
    X += .7 * numpy.random.randn(2, n_samples)
    X = X.T
    return X

def agglomerative(X, n_clusters=3, use_connectivity=True, **kwargs):
    if use_connectivity:
        connectivity = sklearn.neighbors.kneighbors_graph(X, 30)
    else:
        connectivity = None

    linkage = 'ward'
    model = sklearn.cluster.AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters, **kwargs)
    model.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.spectral)
    plt.title('Agglomerative')

def kMeans(X, n_clusters=3, use_connectivity=True, **kwargs):
    model = sklearn.cluster.KMeans(n_clusters=n_clusters, **kwargs)
    model.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.spectral)
    plt.title('kMeans')
    centers = model.cluster_centers_
    for i in range(n_clusters):
        plt.scatter([centers[i,0]],[centers[i,1]],c=i,marker='x',s=100)


if __name__ == "__main__":
    plt.subplot(1,2,1)
    x = sampleData()
    agglomerative(x)
    plt.subplot(1,2,2)
    kMeans(x)
    plt.savefig("outputs/clustering/agglom-vs-kmeans.png",transparent=True)

