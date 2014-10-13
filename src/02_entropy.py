#!/usr/bin/env python


import os
import cPickle
import numpy
import entropy
import graph_tools
from sklearn import svm
import faam
import classifiers


def make_dim_chunks(head=3630013, series_len=19, start=0, stop=50, restrict_disease=True):
    my_lsi_v = entropy.lsi_v("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgood." % (["complete", "disease"][restrict_disease], head, ), series_len)
    fh = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (["complete", "disease"][restrict_disease], head, start, stop, ), "w")
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

def make_dims_as_nested_means(head=3630013, splits=8, restrict_disease=False):
    #fh_r = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt.pickle" % (head, ), "w")
    fh_w = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.nested_means-%s.pickle" % (["complete", "disease"][restrict_disease], head, splits), "w")
    counter = 0
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (["complete", "disease"][restrict_disease], head, dim_chunk, dim_chunk+50, ), "r")
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

def make_conditional_entropies(head=3630013, ndims=100, idims=range(100), jdims=range(100), levels=256, restrict_disease=False, splits=5):
    fh_r = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.nested_means-%s.pickle" % (["complete", "disease"][restrict_disease], head, splits, ), "r")
    fh_w = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.cond_entropy.txt" % (["complete", "disease"][restrict_disease], head, ), "w")
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

def make_conditional_entropies(head=3630013, ndims=100, idims=range(100), jdims=range(100), levels=256, restrict_disease=False, splits=5):
    fh_r = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.nested_means-%s.pickle" % (["complete", "disease"][restrict_disease], head, splits, ), "r")
    fh_w = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.cond_entropy.txt" % (["complete", "disease"][restrict_disease], head, ), "w")
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

def make_conditional_entropies_3(head=3630013, ndims=100, idims=range(100), jdims=99, kdims=100, levels=256, restrict_disease=False, splits=3):
    fh_r = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.nested_means-%s.pickle" % (["complete", "disease"][restrict_disease], head, splits, ), "r")
    fh_w = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.cond_entropy_3.txt" % (["complete", "disease"][restrict_disease], head, ), "w")
    print >> fh_w, "\t".join(["## dim1", "dim2", "dim3", "cond_entropy"])
    print "reading"
    dims = []
    for i in range(ndims):
        dims += [cPickle.load(fh_r)]
    fh_r.close()
    print "generating entropies"
    for i_dim1 in idims: #range(ndims):
        print i_dim1
        for i_dim2 in range(jdims): #range(ndims):
            if i_dim1 != i_dim2:
                for i_dim3 in range(i_dim2+1,kdims):
                    
                    cond_entropy = entropy.cond_entropy_3(dims[i_dim1], dims[i_dim2], dims[i_dim3], levels=levels)
                    print >> fh_w, "\t".join(map(str, [i_dim1, i_dim2, i_dim3, cond_entropy]))
    fh_w.close()


def make_cliques(head=None, threshold=0.452, restrict_disease=False):
    print "making entropies from conditional entropies"
    d_e = make_entropies(head, restrict_disease)
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

def make_classifiers(head=47350, thresholds=[.452], method="faam", restrict_disease=False):
    #fh_w = open("../data/warehouse/gensim_complete_corpus.lsi.%s.projection.vt2.class_stats.txt" % (["complete", "disease"][restrict_disease],head, ), "w")
    all_dims_train = []

    print "reading vts, train"
    counter = 0
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (["complete", "disease"][restrict_disease], head, dim_chunk, dim_chunk+50, ), "r")
        for d in range(50):
            if counter%20 == 0:
                print "dim", counter
            dim = list(cPickle.load(fh_r))
            all_dims_train += [dim]
            counter += 1
        fh_r.close()
    #fh_w.close()
    print len(all_dims_train)
    print len(all_dims_train[0])
    #fh_r.close()

    print "reading vts, evalu"
    counter = 0
    all_dims_evalu = []
    for dim_chunk in range(0,500,50):
        fh_r = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vgoodt2.%03d-%03d.pickle" % (["complete", "disease"][restrict_disease], 47299, dim_chunk, dim_chunk+50, ), "r")
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
    print "d_evalu_tid2gsid", len(d_evalu_tid2gsid)
    d_evalu_gsid2diseases = make_gsid2diseases(group="evalu")
    print "d_evalu_gsid2diseases", len(d_evalu_gsid2diseases)

    #diseases = sorted(list(set(diseases)))
    diseases = ["Alzheimer Disease", "Amyotrophic Lateral Sclerosis", "Arrhythmias, Cardiac", "Diabetes Mellitus", "HIV Infections", "Leukemia", "Parkinson Disease", "Raynaud Disease"]
    n_samples_train = len(all_dims_train[0])
    print n_samples_train
    n_samples_evalu = len(all_dims_evalu[0])
    print n_samples_evalu

    d_disease2cats_train = {}
    for disease in diseases:
        d_disease2cats_train[disease] = [[0,1][disease in d_train_gsid2diseases[d_train_tid2gsid[i]]] for i in range(n_samples_train)]
    #print d_disease2cats_train[disease][:100]
    d_disease2cats_evalu = {}
    for disease in diseases:
        d_disease2cats_evalu[disease] = [[0,1][disease in d_evalu_gsid2diseases[d_evalu_tid2gsid[i]]] for i in range(n_samples_evalu)]

    fh_w = open("../data/warehouse/gensim_%s_corpus.lsi.%s.projection.vt2.class_stats_evalus.%s.txt" % (["complete", "disease"][restrict_disease], head, method, ), "w")
    print >> fh_w, "\t".join(["## threshold", "method", "clique", "disease", "True_Pos_eval", "True_Neg_eval", "False_Pos_eval", "False_Neg_eval", "True_Pos_train", "True_Neg_train", "False_Pos_train", "False_Neg_train"])

    d_clique_disease2pline = {}

    for threshold in thresholds:
        cliques = make_cliques(head, threshold, restrict_disease)
        print "threshold", threshold
        print "clique count", len(cliques)

        for clique_id, clique in enumerate(cliques):
            #if clique != (3, 6, 8):
            #    continue
            print "  ", clique, clique_id, "of", len(cliques)
            clique_len = len(clique)
            c_dims_train = [all_dims_train[i] for i in clique]
            vecs_train = [[c_dims_train[j][i] for j in range(clique_len)] for i in range(n_samples_train)]
            #np_vecs = numpy.array([numpy.array(d) for d in vecs])
            c_dims_evalu = [all_dims_evalu[i] for i in clique]
            vecs_evalu = [[c_dims_evalu[j][i] for j in range(clique_len)] for i in range(n_samples_evalu)]

            for disease in diseases:
                if (clique, disease) in d_clique_disease2pline:
                    print >> fh_w, str(threshold)+"\t"+d_clique_disease2pline[(clique, disease)]
                    continue
                #if disease == 'HIV Infections':
                #    fh_p = open("train_hiv.pickle","w")
                #    cPickle.dump(vecs, fh_p)
                #    cPickle.dump(d_disease2cats[disease], fh_p)
                #    fh_p.close()


                p_line = "\t".join(map(str, [method, clique, disease]))
                my_classifier = classifiers.Classifier(method, limit=None)
                my_classifier.train(vecs_train, d_disease2cats_train[disease])
                d_performance = my_classifier.performance(vecs_evalu, d_disease2cats_evalu[disease])
                p_line += "\t" + "\t".join(map(str, [d_performance[k] for k in ["A1P1", "A1PN", "A0P1", "A0PN"]]))
                d_performance = my_classifier.performance(vecs_train, d_disease2cats_train[disease])
                p_line += "\t" + "\t".join(map(str, [d_performance[k] for k in ["A1P1", "A1PN", "A0P1", "A0PN"]]))
                print >> fh_w, str(threshold)+"\t"+p_line
                
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


    #make_classifiers(head=47350, thresholds=[.449, .450, .451, .452, .453], method="faam")

    # training disease
    #for i in range(0,500,50): # ~5m
    #    make_dim_chunks(head=None, series_len=1, start=i, stop=i+50, restrict_disease=True)
    #make_dims_as_nested_means(head=None, splits=5, restrict_disease=True) #~5m
    #make_dims_as_nested_means(head=None, splits=3, restrict_disease=True) #~5m
    make_conditional_entropies(head=None, ndims=500, idims=range(500), jdims=range(500), levels=32, restrict_disease=True, splits=5) #~2.5h
    make_conditional_entropies_3(head=None, ndims=100, idims=range(100), jdims=99, kdims=100, levels=256, restrict_disease=False, splits=3)
    ##for t in [.450, .451, .452, .453, .454, .455]:
    ##    print t, make_cliques(head=47350, threshold=t)

    #evalu
    #for i in range(0,500,50):
    #    make_dim_chunks(head=47299, series_len=1, start=i, stop=i+50, restrict_disease=True)


    #make_classifiers(head=47350, thresholds=[.448, .449, .450, .451, .452, .453, .454, .455, .456, .457, .458], method="svm", restrict_disease=True)
