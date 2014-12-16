#!/usr/bin/env python
"""Text processing example using kmeans.  After example by 
   Peter Prettenhofer <peter.prettenhofer@gmail.com>, Lars Buitinck <L.J.Buitinck@uva.nl>."""

import sklearn.datasets as sdata
import sklearn.feature_extraction.text as sktextfeature
import sklearn.metrics
import sklearn.cluster 

def loadNewsgroupsData(categories=None, seed=123):
    """Load the 20 newsgroups data set.  Filter on categories if present."""
    remove = ('headers', 'footers', 'quotes')

    data = sdata.fetch_20newsgroups(categories=categories, shuffle=True, random_state=seed, remove=remove)
    categories = data.target_names    

    # Break up data sets, and strip out stop words
    y = data.target

    vectorizer = sktextfeature.TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X = vectorizer.fit_transform(data.data)
    return X, y, data.data

def newsgroupsProblem(categories=None, k=4, seed=123, **kwargs):
    X, y, text = loadNewsgroupsData(categories, seed)

    km = sklearn.cluster.KMeans(n_clusters=k, init='k-means++', max_iter=100, verbose=False, **kwargs)
    km.fit(X)
    print "Homogeneity: ",  sklearn.metrics.homogeneity_score(y, km.labels_)
    print "Completeness: ",  sklearn.metrics.completeness_score(y, km.labels_)

if __name__ == "__main__":
    categories = ['comp.os.ms-windows.misc', 'misc.forsale', 'rec.motorcycles', 'talk.religion.misc']
    newsgroupsProblem(categories)

