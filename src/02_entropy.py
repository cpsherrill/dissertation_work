#!/usr/bin/env python


import os
import cPickle
import numpy
import entropy
import graph_tools
from sklearn import svm
import faam


def make_dim_chunks(head=3630013, series_len=19, start=0, stop=50):
    my_lsi_v = entropy.lsi_v("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgood." % (head, ), series_len)
    fh = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (head, start, stop, ), "w")
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
    fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.nested_means.pickle" % (head, ), "w")
    counter = 0
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (head, dim_chunk, dim_chunk+50, ), "r")
        for d in range(50):
            dim = list(cPickle.load(fh_r))
            if counter%20 == 0:
                print "dim", counter, len(dim)
            my_lsi_dim = entropy.lsi_dimension(dim)
            cPickle.dump(my_lsi_dim.id_by_nested_means(splits), fh_w)
            counter += 1
        fh_r.close()
    fh_w.close()
    #fh_r.close()

def make_conditional_entropies(head=3630013, ndims=100, idims=range(100), jdims=range(100), levels=256):
    fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.nested_means.pickle" % (head, ), "r")
    fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.cond_entropy.txt" % (head, ), "w")
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
    fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.cond_entropy.txt" % (head, ), "r")
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


def make_gsid2diseases(group="train"):
    fh_r = open("../data/medline_mesh_target_diseases.%s.txt" % (group, ) ,"r")
    d_train_gsid2diseases = {}
    #diseases = []
    for line in fh_r:
        if "## " not in line:
            gsid = int(line.strip().split("\t")[1])
            disease = line.strip().split("\t")[2]
            #diseases += [disease]
            if gsid not in d_train_gsid2diseases:
                d_train_gsid2diseases[gsid] = [] 
            d_train_gsid2diseases[gsid] += [disease] 
    fh_r.close()
    return d_train_gsid2diseases

def make_tid2gsid(group="train"):
    fh_r = open("../data/medline_mesh_target_diseases.%s.ids" % (group, ),"r")
    lcounter = 0
    d_train_tid2gsid = {}
    for line in fh_r:
        if "## " not in line:
            d_train_tid2gsid[lcounter] = int(line.strip().split("\t")[1]) 
            lcounter += 1
    fh_r.close()
    return d_train_tid2gsid

def make_classifiers(head=47350, thresholds=[.452]):
    #fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.class_stats.txt" % (head, ), "w")
    all_dims = []

    print "reading vts"
    counter = 0
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (head, dim_chunk, dim_chunk+50, ), "r")
        for d in range(50):
            if counter%20 == 0:
                print "dim", counter
            dim = list(cPickle.load(fh_r))
            all_dims += [dim]
            counter += 1
        fh_r.close()
    #fh_w.close()
    print len(all_dims)
    print len(all_dims[0])
    #fh_r.close()

    print "reading vts, evalu"
    counter = 0
    all_dims_evalu = []
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (47299, dim_chunk, dim_chunk+50, ), "r")
        for d in range(50):
            if counter%20 == 0:
                print "dim", counter
            dim = list(cPickle.load(fh_r))
            all_dims_evalu += [dim]
            counter += 1
        fh_r.close()
    print len(all_dims_evalu)
    print len(all_dims_evalu[0])

    d_train_tid2gsid = make_tid2gsid(group="train")
    print "d_train_tid2gsid", len(d_train_tid2gsid), d_train_tid2gsid[47348]
    d_train_gsid2diseases = make_gsid2diseases(group="train")
    print "d_train_gsid2diseases", len(d_train_gsid2diseases), d_train_gsid2diseases[125]

    d_evalu_tid2gsid = make_tid2gsid(group="evalu")
    print "d_train_tid2gsid", len(d_evalu_tid2gsid)
    d_evalu_gsid2diseases = make_gsid2diseases(group="evalu")
    print "d_evalu_gsid2diseases", len(d_evalu_gsid2diseases)

    #diseases = sorted(list(set(diseases)))
    diseases = ["Alzheimer Disease", "Amyotrophic Lateral Sclerosis", "Arrhythmias, Cardiac", "Diabetes Mellitus", "HIV Infections", "Leukemia", "Parkinson Disease", "Raynaud Disease"]
    n_samples = len(all_dims[0])
    print n_samples
    n_esamples = len(all_dims_evalu[0])
    print n_esamples

    d_disease2cats = {}
    for disease in diseases:
        d_disease2cats[disease] = [[0,1][disease in d_train_gsid2diseases[d_train_tid2gsid[i]]] for i in range(n_samples)]
    print d_disease2cats[disease][:100]
    d_disease2catse = {}
    for disease in diseases:
        d_disease2catse[disease] = [[0,1][disease in d_evalu_gsid2diseases[d_evalu_tid2gsid[i]]] for i in range(n_esamples)]

    fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.class_stats_evalus.txt" % (head, ), "w")
    print >> fh_w, "\t".join(["## threshold", "clique", "disease", "True_Pos_eval", "True_Neg_eval", "False_Pos_eval", "False_Neg_eval", "True_Pos_train", "True_Neg_train", "False_Pos_train", "False_Neg_train"])

    d_clique_disease2pline = {}

    for threshold in thresholds:
        cliques = make_cliques(head, threshold)
        print "clique count", len(cliques)

        for clique_id, clique in enumerate(cliques):
            if clique != (3, 6, 8):
                continue
            print clique, clique_id, "of", len(cliques)
            clique_len = len(clique)
            c_dims = [all_dims[i] for i in clique]
            vecs = [[c_dims[j][i] for j in range(clique_len)] for i in range(n_samples)]
            np_vecs = numpy.array([numpy.array(d) for d in vecs])
            c_dims_e = [all_dims_evalu[i] for i in clique]
            vecs_e = [[c_dims_e[j][i] for j in range(clique_len)] for i in range(n_esamples)]

            for disease in diseases:
                if (clique, disease) in d_clique_disease2pline:
                    print >> fh_w, d_clique_disease2pline[(clique, disease)]
                    continue
                if disease == 'HIV Infections':
                    fh_p = open("train_hiv.pickle","w")
                    cPickle.dump(vecs, fh_p)
                    cPickle.dump(d_disease2cats[disease], fh_p)
                    fh_p.close()
                net = faam.FAAM(47439,10000, len(clique))
                k = 0
                y = d_disease2cats[disease] 
                for i in range( 0, 1000, 10 ):
                    for j in range( 10 ):
                        net.SetTrainer( np_vecs[k], y[k] )
                        k += 1
                    net.Train()
                    print i, net.lastr
                net.PurgeRtable()
                net_guesses = []
                for i, v in enumerate(np_vecs):
                    if i%100 == 0:
                        print i
                    net_guesses += [net.Recall( v )]
                real_guesses = zip(d_disease2catse[disease], net_guesses)
                d_count = {}
                for rg in real_guesses:
                    if rg not in d_count:
                        d_count[rg] = 0
                    d_count[rg] += 1
                print "\t".join(map(str, [threshold, clique, disease, d_count.get((1,1),0), d_count.get((1,0),0), d_count.get((0,1),0), d_count.get((0,0),0)]))


                clf = svm.LinearSVC()
                #print "  "+disease
                clf.fit(vecs, d_disease2cats[disease] ) 
                #print "  guessing"
                guesses = list(clf.predict(vecs_e))
                real_guesses = zip(d_disease2catse[disease], guesses)
                d_count = {}
                for rg in real_guesses:
                    if rg not in d_count:
                        d_count[rg] = 0
                    d_count[rg] += 1
                p_line = "\t".join(map(str, [threshold, clique, disease, d_count.get((1,1),0), d_count.get((1,0),0), d_count.get((0,1),0), d_count.get((0,0),0)]))
                guesses = list(clf.predict(vecs))
                real_guesses = zip(d_disease2cats[disease], guesses)
                d_count = {}
                for rg in real_guesses:
                    if rg not in d_count:
                        d_count[rg] = 0
                    d_count[rg] += 1
                p_line += "\t" + "\t".join(map(str, [d_count.get((1,1),0), d_count.get((1,0),0), d_count.get((0,1),0), d_count.get((0,0),0)]))
                print >> fh_w, p_line
                d_clique_disease2pline[(clique, disease)] = p_line
    fh_w.close()
        

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
    ##for t in [.450, .451, .452, .453, .454, .455]:
    ##    print t, make_cliques(head=47350, threshold=t)

    #evalu
    #for i in range(0,500,50):
    #    make_dim_chunks(head=47299, series_len=1, start=i, stop=i+50)


    make_classifiers(head=47350, thresholds=[.449, .450, .451, .452, .453])
