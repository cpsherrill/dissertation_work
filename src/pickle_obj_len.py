#!/usr/bin/env python

import sys
import cPickle

fh = open(sys.argv[1],"r")
o = cPickle.load(fh)
print len(o)
fh.close()
