#!/usr/bin/env python

import faam
import numpy
from sklearn import svm

class Classifier(object):
    
    def __init__(self, method, limit=None):
        self.method = method
        self.limit = limit

    def _train_faam(self, vectors, classifications):
        self.net = faam.FAAM(self.train_vectors_ct, 10000, self.train_vectors_len)
        k = 0
        for i in range( 0, self.train_vectors_ct, 100 ):
            for j in range( 100 ):
                if k < self.train_vectors_ct:
                    self.net.SetTrainer( vectors[k], classifications[k] )
                k += 1
                self.net.Train()
            if i%1000 == 0:
                print "    FAAM train, vec: %s; divisions: %s" % (i, self.net.lastr)
        self.net.Train()
        self.net.PurgeRtable()


    def _train_svm(self, vectors, classifications):
        self.clf = svm.LinearSVC()
        self.clf.fit(vectors, classifications)


    def train(self, vectors, classifications):
        if self.limit is not None:
            vectors = vectors[:self.limit]
            classifications = classifications[:self.limit]
        self.train_vectors_ct  = len(vectors)
        self.train_vectors_len = len(vectors[0])
        if self.method == "faam":
            self._train_faam(vectors, classifications)
        elif self.method == "svm":
            self._train_svm(vectors, classifications)
        else:
            print "Unknown method"


    def _evalu_faam(self, vectors):
        for i, vec in enumerate(vectors):
            if i%5000==0:
                print "    FAAM evalu, vec: %s" % (i, )
            yield self.net.Recall( vec )

    def _evalu_svm(self, vectors):
        for i, guess in enumerate(list(self.clf.predict(vectors))):
            if i%20000==0:
                print "    SVM evalu, vec: %s" % (i, )
            yield guess


    def evalu(self, vectors):
        if self.limit is not None:
            vectors = vectors[:self.limit]
        self.evalu_vectors_ct  = len(vectors)
        self.evalu_vectors_len = len(vectors[0])
        if self.method == "faam":
            return self._evalu_faam(vectors)
        elif self.method == "svm":
            return self._evalu_svm(vectors)
        else:
            print "Unknown method"

    def performance(self, vectors, classifications):
        if self.limit is not None:
            vectors = vectors[:self.limit]
            classifications = classifications[:self.limit]
        guesses = self.evalu(vectors)
        list_real_guess = zip(classifications, guesses)
        d_count = {}
        for rg in list_real_guess:
            if rg not in d_count:
                d_count[rg] = 0
            d_count[rg] += 1
        d = {}
        d["A1P1"] = d_count.get((1,1),0)
        d["A1PN"] = sum( [d_count[k] for k in d_count.keys() if k[0]==1 and k[1]!=1], 0 )
        d["A0P1"] = d_count.get((0,1),0)
        d["A0PN"] = sum( [d_count[k] for k in d_count.keys() if k[0]==0 and k[1]!=1], 0 )
        return d
