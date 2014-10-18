#!/usr/bin/env python

import cPickle
from sklearn.decomposition import PCA
from sklearn import svm

fh = open("train_hiv_027.pickle","r")

vecs = cPickle.load(fh)
assignments = cPickle.load(fh)

fh.close()

print len(vecs)
print len(vecs[0])
print len(assignments)
print sum(assignments)


pca = PCA(n_components=3)
pca.fit(vecs)
t = pca.transform(vecs)


g = []
for i in range(len(vecs)):
    if assignments[i]==1:
        g+= [t[i][0]]
print "non"
b = []
for i in range(len(vecs)):
    if assignments[i]==0:
        b+= [t[i][0]]

g.sort()
b.sort()

print ["%.3f"%g[int(len(g)*x)] for x in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]
print ["%.3f"%b[int(len(b)*x)] for x in [.1,.2,.3,.4,.5,.6,.7,.8,.9]]


clf = svm.SVC(kernel='linear')
clf.fit(t, assignments)
ps = clf.predict(t)
rps = zip(assignments, ps)

d = {}
for rp in rps:
    if rp not in d:
        d[rp] = 0
    d[rp] += 1
print d


import numpy

Y = numpy.array(assignments)
ndx = (Y==1).nonzero()[0]
posdocs = t[ndx]
fh=open("pos.txt","w")
for p in posdocs:
    print >>fh, "\t".join(map(str, p))
fh.close()
ndx = (Y==0).nonzero()[0]
posdocs = t[ndx]
fh=open("neg.txt","w")
for p in posdocs:
    print >>fh, "\t".join(map(str, p))
fh.close()
