#!/usr/bin/env python
"""Some demo examples for decision trees."""

__havepydot__ = True

import pandas
import sklearn
import sklearn.tree
import sklearn.datasets
import numpy
import numpy.random
import sklearn.externals.six
import sklearn.cross_validation as cv

try:
    import pydot
except ImportError, e:
    print "Don't have pydot: will run, but won't be able to generate plots."
    __havepydot__ = False

def goodEvilData():
    """Returns the batman good/evil data set.
       Example from Rob Schapire, Princeton, as far as I know:
       http://www.cs.princeton.edu/~schapire/talks/picasso-minicourse.pdf .
       Returns dataframe train, correct labels, and a test dataframe."""
    train = { 'male'  :[True,  True,  True,  True,  False, True],
              'mask'  :[True,  True,  False, False, True,  False],
              'cape'  :[True,  True,  False, False, False, False],
              'tie'   :[False, False, True,  True,  False, False],
              'ears'  :[True,  False, False, False, True,  False],
              'smokes':[False, False, False, True,  False, False]}
    traingood = [True,  True,  True,  False, False, False]
    trainnames =['batman','robin','alfred','pengin','catwoman','joker']
    traindf = pandas.DataFrame(train, index=trainnames )
    test  = { 'male'  :[False, True],
              'mask'  :[True,  True],
              'cape'  :[True,  False],
              'tie'   :[False, False],
              'ears'  :[True,  False],
              'smokes':[False, False]}
    testnames = ['batgirl','riddler']
    testdf = pandas.DataFrame(test, index=testnames )
    return traindf, traingood, testdf

def irisProblem(trainfrac=0.66, printConfusion=False, **kwargs):
    iris = sklearn.datasets.load_iris()
    traindata, testdata, trainlabels, testlabels = cv.ntrain_test_split(iris.data, iris.target, test_size=trainfrac)
    
    # these are the defaults
    #model = sklearn.tree.DecisionTreeClassifier(criterion='gini',splitter='best',max_features=None,max_depth=None)
    model = sklearn.tree.DecisionTreeClassifier(**kwargs)
    model = model.fit(traindata, trainlabels)
    predictions = model.predict(testdata)

    nwrong = sum(predictions != testlabels)
    print "Misclassifications on test set: ", nwrong, "out of ", len(predictions)
    if printConfusion:
        print sklearn.metrics.confusion_matrix(testlabels,predictions)
    return nwrong

if __name__ == "__main__":
    base = 'outputs/classification/'
    train, labels, test = goodEvilData()
    model = sklearn.tree.DecisionTreeClassifier()
    model.fit(train, labels)
    predictions = model.predict(test)

    if __havepydot__:
        try:
            dot_data = sklearn.externals.six.StringIO()
            sklearn.tree.export_graphviz(model, out_file=dot_data)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            graph.write_png(base+"basic.png")
            dot_data2 = sklearn.externals.six.StringIO()
            sklearn.tree.export_graphviz(model, out_file=dot_data2, feature_names=train.columns)
            graph = pydot.graph_from_dot_data(dot_data2.getvalue())
            graph.write_png(base+"good-evil.png")
        except: 
            print "Could not generate decision tree visualization."
    else:
        # to satisfy make
        open(base+'basic.png', 'a').close()
        open(base+'good-eveil.png', 'a').close()

