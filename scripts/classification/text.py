#!/usr/bin/env python
"""Example of text processing using Naive Bayes Classifier.  Based on the sklearn example"""

import numpy 
import sys
from time import time
import matplotlib.pyplot as plt

import sklearn.datasets as sdata
import sklearn.feature_extraction.text as extract
import sklearn.metrics 
from sklearn.feature_extraction.text import HashingVectorizer
import sklearn.naive_bayes 
from sklearn.utils.extmath import density


def loadNewsgroupsData(categories=None, seed=123):
    """Load the 20 newsgroups data set.  Filter on categories if present."""
    remove = ('headers', 'footers', 'quotes')

    data_train = sdata.fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=seed, remove=remove)
    data_test = sdata.fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42, remove=remove)

    categories = data_train.target_names    

    # Break up data sets, and strip out stop words
    y_train, y_test = data_train.target, data_test.target

    vectorizer = extract.TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
    return X_train, y_train, X_test, y_test 

def newsgroupsProblem(categories=None, seed=123, printConfusion=False, **kwargs):
    X_train, y_train, X_test, y_test = loadNewsgroupsData(categories, seed)

    classifier = sklearn.naive_bayes.MultinomialNB(alpha=.01, **kwargs)
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    score = sklearn.metrics.f1_score(y_test, pred)
    print "f1-score: ", score

    if printConfusion:
        print sklearn.metrics.confusion_matrix(y_test, pred)

if __name__ == "__main__":
    categories = ['comp.os.ms-windows.misc', 'misc.forsale', 'rec.motorcycles', 'talk.religion.misc']
    newsgroupsProblem(categories, printConfusion=True)

