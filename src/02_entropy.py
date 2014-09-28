#!/usr/bin/env python


import os
import cPickle
import numpy
import entropy
import graph_tools

def make_dim_chunks(head=3630013, series_len=19, start=0, stop=50):
    my_lsi_v = entropy.lsi_v("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.v." % (head, ), series_len)
    fh = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.%03d-%03d.pickle" % (head, start, stop, ), "w")
    for i, d in enumerate(my_lsi_v._dim_all_docs(start, stop)):
        if i%10==0:
            print "  dim", i
        cPickle.dump(list(d), fh)
    fh.close()


def make_dims(head=3630013, series_len=19):
    my_lsi_v = entropy.lsi_v("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.v." % (head, ), series_len)
    fh = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.pickle" % (head, ), "w")
    for i, d in enumerate(my_lsi_v._iter_dims()):
        if i%10==0:
            print "  dim", i
        cPickle.dump(d, fh)
    fh.close()

#def make_dims_as_nested_means(head=3630013, series_len=19, splits=8):
#    my_lsi_v = entropy.lsi_v("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.v." % (head, ), series_len)
#    fh = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.v.nested_means.pickle" % (head, ), "w")
#    for i, d in enumerate(my_lsi_v._iter_dims_as_nested_mean_groups(splits)):
#        if i%10==0:
#            print "  dim", i
#        cPickle.dump(d, fh)
#    fh.close()

def make_dims_as_nested_means(head=3630013, splits=8):
    #fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt.pickle" % (head, ), "w")
    fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.nested_means.pickle" % (head, ), "w")
    counter = 0
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.%03d-%03d.pickle" % (head, dim_chunk, dim_chunk+50, ), "r")
        for d in range(50):
            if counter%20 == 0:
                print "dim", counter
            dim = list(cPickle.load(fh_r))
            my_lsi_dim = entropy.lsi_dimension(dim)
            cPickle.dump(my_lsi_dim.id_by_nested_means(splits), fh_w)
            counter += 1
        fh_r.close()
    fh_w.close()
    #fh_r.close()

def make_conditional_entropies(head=3630013, ndims=100, idims=range(100), jdims=range(100), levels=256):
    fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.nested_means.pickle" % (head, ), "r")
    fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.cond_entropy.txt" % (head, ), "w")
    print >> fh_w, "\t".join(["## dim1", "dim2", "cond_entropy"])
    print "reading"
    dims = []
    for i in range(ndims):
        dims += [cPickle.load(fh_r)]
    fh_r.close()
    print "generating entropies"
    for i_dim1 in idims: #range(ndims):
        print i_dim1
        for i_dim2 in jdims: #range(ndims):
            if i_dim1 != i_dim2:
                cond_entropy = entropy.cond_entropy(dims[i_dim1], dims[i_dim2], levels=levels)
                print >> fh_w, "\t".join(map(str, [i_dim1, i_dim2, cond_entropy]))
    fh_w.close()

def make_entropies(head=None):
    fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.cond_entropy.txt" % (head, ), "r")
    d_e = {}
    for line in fh_r:
        if "## " not in line:
            fields = line.strip().split("\t")
            dim1 = int(fields[0])
            dim2 = int(fields[1])
            ce   = float(fields[2])
            dims = tuple(sorted([dim1, dim2]))
            if dims not in d_e:
                d_e[dims] = ce
            else:
                d_e[dims] = max([ce, d_e[dims]])
    fh_r.close()
    return d_e

def make_cliques(head=None, threshold=0.452):
    print "making entropies from conditional entropies"
    d_e = make_entropies(head)
    print "finding cliques"
    return [i for i in graph_tools.entropies2cliques(d_e, threshold)]


if __name__ == "__main__":
    '''
    make_dim_chunks(head=3630013, series_len=19, start=   0, stop=  50)
    make_dim_chunks(head=3630013, series_len=19, start=  50, stop= 100)
    make_dim_chunks(head=3630013, series_len=19, start= 100, stop= 150)
    make_dim_chunks(head=3630013, series_len=19, start= 150, stop= 200)
    make_dim_chunks(head=3630013, series_len=19, start= 200, stop= 250)
    make_dim_chunks(head=3630013, series_len=19, start= 250, stop= 300)
    make_dim_chunks(head=3630013, series_len=19, start= 300, stop= 350)
    make_dim_chunks(head=3630013, series_len=19, start= 350, stop= 400)
    make_dim_chunks(head=3630013, series_len=19, start= 400, stop= 450)
    make_dim_chunks(head=3630013, series_len=19, start= 450, stop= 500)
    '''
    ##make_dims(head=3630013, series_len=19)
    #make_dims_as_nested_means(head=47350, splits=5)
    ##make_dims_as_nested_means(head=102400, series_len=11, splits=4)
    #make_conditional_entropies(head=3630013, ndims=100)



    # training
    #for i in range(0,500,50):
    #    make_dim_chunks(head=47350, series_len=1, start=i, stop=i+50)
    #make_dims_as_nested_means(head=47350, splits=5)
    #make_conditional_entropies(head=47350, ndims=500, idims=range(500), jdims=range(500), levels=32)
    for t in [.450, .451, .452, .453, .454, .455]:
        print t, make_cliques(head=47350, threshold=t)
