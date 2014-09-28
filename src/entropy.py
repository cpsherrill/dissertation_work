#!/usr/bin/env python

import os
import cPickle
import numpy
import collections
import math


def cond_entropy(dim1, dim2, levels=256):
    paired = zip(dim1, dim2)
    dim1_counter = collections.Counter()
    for level in dim1:
        dim1_counter[level] += 1
    pair_counter = collections.Counter()
    for pair in paired:
        pair_counter[pair] += 1
    column_entropies = []
    for level_col in range(levels):
        numerator = 0.0
        col_count = dim1_counter[level_col]
        for level_row in range(levels):
            density = pair_counter[(level_col, level_row)]*1.0/col_count
            if density:
                numerator += density * math.log(density)
        column_entropy = -1.0 * numerator / math.log(col_count)
        column_entropies += [(column_entropy, col_count)]
    conditional_entropy = sum([column_entropy*col_count for (column_entropy, col_count) in column_entropies]) / len(dim1)
    return conditional_entropy


def mean(list1):
    return sum(list1)*1.0/len(list1)

def nested_mean_thresholds(list1, splits=8, default=0):
    if not len(list1):
        return [default] * (2**splits)
    if splits == 1:
        return [mean(list1)]
    else:
        my_mean = mean(list1)
        list1_lower = [l for l in list1 if l<=my_mean]
        list1_upper = [l for l in list1 if l> my_mean]
        results_lower = nested_mean_thresholds(list1_lower, splits-1)
        results_upper = nested_mean_thresholds(list1_upper, splits-1, default=results_lower[-1])
        return results_lower+[my_mean]+results_upper



class lsi_dimension(object):

    def __init__(self, values):
        self.values = values

    def id_by_N_ile(self, N):
        values_sorted = sorted(list(self.values))
        len1 = len(values_sorted)
        thresholds = [values_sorted[p*len1/N] for p in range(N)] # lower bounds
        return [[i for (i,t) in enumerate(thresholds) if v>=t][-1] for v in list(self.values)]
    
    def id_by_nested_means(self, S):
        thresholds = [None] + nested_mean_thresholds(list(self.values), splits=S) # lower bounds
        return [[i for (i,t) in enumerate(thresholds) if v>=t][-1] for v in list(self.values)]




class lsi_v(object):

    def __init__(self, file_base, series_len):
        self.file_base = file_base
        self.series_len = series_len

    def _iter_v_files(self):
        for i in range(self.series_len):
            yield self.file_base + "%04d.pickle" % (i, )
    
    def _iter_vs(self):
        for fn in self._iter_v_files():
            print " ", fn
            fh = open(fn, "r")
            v = cPickle.load(fh)
            fh.close()
            yield v

    def _iter_vts(self):
        for v in self._iter_vs():
            yield v.T

    def _dim_all_docs(self, dimension_min, dimension_max):
        vts = (vt for vt in self._iter_vts())
        dims = None #vts.next() #numpy.array([]) #self._iter_vts().next()[dimension_min:dimension_max]
        for vt in self._iter_vts():
            if dims is None:
                dims = vt[dimension_min:dimension_max]
                #pass
            else:
                dims = numpy.concatenate((dims, vt[dimension_min:dimension_max]), axis=1)
                #pass
        return dims

    
    def _iter_dims_as_N_iles(self, N=8):
        for dim_min in range(0,500,50):
            print dim_min, "to", dim_min+50
            dims = self._dim_all_docs(dim_min, dim_min+50)
            for dim in dims:
                my_lsi_dim = lsi_dimension(dim)
                yield my_lsi_dim.id_by_N_ile(N)

    def _iter_dims_as_nested_mean_groups(self, S=3):
        for dim_min in range(0,500,100): ###
            print dim_min, "to", dim_min+100
            dims = self._dim_all_docs(dim_min, dim_min+100)
            for dim in dims:
                my_lsi_dim = lsi_dimension(dim)
                yield my_lsi_dim.id_by_nested_means(S)
            del dims

    def _iter_dims(self):
        for dim_min in range(0,500,50): ###
            print dim_min, "to", dim_min+50
            dims = self._dim_all_docs(dim_min, dim_min+50)
            for dim in dims:
                yield dim
            del dims


if __name__ == "__main__":
    my_lsi_v = lsi_v("../data/warehouse/gensim_complete_corpus.lsi.3630013.projection.v.", 1)
    #for v in my_lsi_v._iter_vs():
    #    print len(v)
    #print nested_mean_thresholds(range(100)+range(50)+range(50),3)
    
    
    dims_8 = [d for d in my_lsi_v._iter_dims_as_nested_mean_groups(3)]
    
    print len(dims_8)
    print len(dims_8[0])
    print dims_8[0][:100]
    fh = open("../data/warehouse/gensim_complete_corpus.lsi.3630013.projection.v.N_iles.pickle","w")
    cPickle.dump(dims_8, fh)
    fh.close()
    
