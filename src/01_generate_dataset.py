#!/usr/bin/env python

import cPickle
import medline
import nlp
import random
import logging

logging.basicConfig(filename='example.log',level=logging.DEBUG)

def determine_stats(medline_dir):
    # determine stats for medline files
    my_medline = medline.medline(medline_dir)
    fh_m = open("../data/medline_mesh.txt","w")
    print >> fh_m, "\t".join(["## pmid", "major", "key", "minors"])
    for i, (pmid, d_info) in  enumerate(my_medline._read_pmids(log_fn = "../data/medline_stats2.txt")):
        meshs = d_info["source"].get("MH", [])
        for mesh in meshs:
            mymesh = medline.mesh(mesh)
            print >> fh_m, "\t".join([pmid, mymesh.major, str(mymesh.key)]+mymesh.minors)
    fh_m.close()

def select_files_of_interest():
    print "determining pmids of interest"
    # select only files of interest
    fh = open("../data/medline_stats.txt","r")
    counter_total = 0
    counter_eng = 0
    counter_long = 0
    counter_recent = 0
    counter_mesh = 0
    pmids = set([])
    for i, line in enumerate(fh):
        #if i %1000000 == 0: 
        #    print i
        if "## " not in line:
            [pmid, lang, ab_word_ct, date, mesh_ct] = line.strip().split("\t")
            counter_total += 1
            if lang != "eng":
                continue
            counter_eng += 1
            if int(ab_word_ct) < 150:
                continue
            counter_long += 1
            if int(date) < 20010000:
                continue
            counter_recent += 1
            if int(mesh_ct) == 0:
                continue
            counter_mesh += 1
            pmids.add(pmid)
    fh.close()
    print "counter_total:", counter_total
    print "counter_eng:", counter_eng
    print "counter_long:", counter_long
    print "counter_recent:", counter_recent
    print "counter_mesh:", counter_mesh
    
    fh_w = open("../data/pmids_source.pickle","w")
    cPickle.dump(pmids, fh_w)
    fh_w.close()
    return pmids

def generate_corpus_file(pmids, medline_dir, fn):
    print "generating corpus file"
    my_medline = medline.medline(medline_dir)
    fh = open(fn, "w")
    print >> fh, "\t".join(["## pmid", "stop_stem"])
    for i, (pmid, abstract) in enumerate(my_medline.get_processed_abstracts(pmids)):
        if i%1000000==0:
            print i
        print >> fh, "\t".join([pmid, abstract])
    fh.close()


def get_pmids(train=True, evalu=True):
    pmids = []
    if train:
        fh_r = open("../data/medline_mesh_target_diseases.train.ids","r")
        pmids += [int(l.split("\t")[0]) for l in fh_r.readlines() if "## " not in l]
        fh_r.close()
    if evalu:
        fh_r = open("../data/medline_mesh_target_diseases.evalu.ids","r")
        pmids += [int(l.split("\t")[0]) for l in fh_r.readlines() if "## " not in l]
        fh_r.close() 
    return set(pmids)

def generate_dictionary(restrict_disease=False):
    pmids = None
    if restrict_disease:
        pmids = get_pmids(train=True, evalu=True)
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt", pmids=pmids)
    mycorpus.make_dictionary("../data/gensim_%s_corpus.dict" % (["complete", "disease"][restrict_disease], ))

def generate_gensim_corpus(head=None, restrict_disease=False, train=True, evalu=True):
    pmids = None
    if restrict_disease:
        pmids = get_pmids(train=train, evalu=evalu)    
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt", head=head, pmids=pmids, ignore_head=True)
    fn_base = "../data/gensim_%s_corpus" % (["complete", "disease"][restrict_disease], )
    mycorpus.get_dictionary(fn_base + ".dict")
    mycorpus.make_gensim_corpus(fn_base + ".mm."+str(head), fn_base + ".tfidf."+str(head))
    
def generate_gensim_models(head=None, restrict_disease=False):
    pmids = None
    if restrict_disease:
        pmids = get_pmids(train=True, evalu=True)    
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt", head=head, pmids=pmids)
    fn_base = "../data/gensim_%s_corpus" % (["complete", "disease"][restrict_disease], )
    mycorpus.get_dictionary(fn_base + ".dict")
    mycorpus.make_gensim_models(fn_base+".mm."+str(head), fn_base+".tfidf."+str(head), fn_base+".lsi."+str(head), fn_base+".lda."+str(head))

def generate_gensim_v(head=None):
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt", head)
    mycorpus.make_gensim_v("../data/gensim_complete_corpus.mm."+str(head), "../data/gensim_complete_corpus.tfidf."+str(head), "../data/gensim_complete_corpus.lsi."+str(head), "../data/gensim_complete_corpus.lsi.%s.projection.v.pickle" % (str(head), ) )
    fh_w = open("../data/gensim_complete_corpus.lsi.%s.projection.v.pickle" % (str(head), ),"w")
    cPickle.dump(mycorpus.v, fh_w)
    fh_w.close()

    #fh_r = open("../data/gensim_complete_corpus.lsi.800.projection.v.pickle","r")
    #v = cPickle.load(fh_r)
    #fh_r.close()
    
def generate_gensim_vs(head=None, restrict_ids=True, restrict_disease=False):
    pmids = None
    if restrict_disease:
        if restrict_ids:
            if head == 47299:
                pmids = get_pmids(train=False, evalu=True )
            else:
                pmids = get_pmids(train=True,  evalu=False)
        else:
            pmids = get_pmids(train=True, evalu=True)    
        print len(pmids)
    else:
        if restrict_ids:
            if head == 47299:
                pmids = get_pmids(train=False, evalu=True )
            else:
                pmids = get_pmids(train=True,  evalu=False)
    ids = None
    


    if pmids:
        ids = []
        print len(pmids)
        te_pmids = set(get_pmids(train=True, evalu=True ))
        fh_r = open("../data/corpus_all_pmids.txt", "r")
        counter = 0
        printed = False
        for line_num, line in enumerate(fh_r):
            if "## " not in line:
                line_pmid = int(line.strip())
                if line_pmid in te_pmids:
                    if line_pmid in pmids:
                        ids += [counter] # header
                        if not printed:
                            printed = True
                            print "  adding:", line_pmid, counter
                    counter += 1
        print "  counter: ", counter
        fh_r.close()
        print len(ids)
    

    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt", head=None, pmids=ids)


    fn_base    = "../data/gensim_%s_corpus" % (["complete", "disease"][restrict_disease], )
    fn_base_wh = "../data/warehouse/gensim_%s_corpus" % (["complete", "disease"][restrict_disease], )
    corpus_size = head
    if head is None:
        corpus_size = 47299 + 47350 - 2
    mycorpus.make_gensim_vs(fn_base+".mm."+str(None), fn_base+".tfidf."+str(None), fn_base_wh+".lsi."+str(head) , 200000, corpus_size, ids)


def generate_test_train():
    fh = open("../data/medline_mesh_target_diseases.txt", "r")
    lines = [line.replace("\n","") for line in fh.readlines()]
    fh.close()
    pmids = [line.split("\t")[0] for line in lines]
    pmids = sorted(list(set(pmids)))
    random.seed(1)
    set_evaluation = set(random.sample(pmids, len(pmids)/2))
    set_train = set([pmid for pmid in pmids if pmid not in set_evaluation])


    fh_w_evalu = open("../data/medline_mesh_target_diseases.evalu.ids","w")
    fh_w_train = open("../data/medline_mesh_target_diseases.train.ids","w")
    print >> fh_w_evalu, "\t".join(["## pmid", "id"])
    print >> fh_w_train, "\t".join(["## pmid", "id"])
    fh = open("../data/corpus_all_stop-stem.txt", "r")
    d_pmid2id = {}
    for line_num, line in enumerate(fh):
        if line_num % 500000 == 0:
            print line_num
        fields = line.split("\t")
        pmid = fields[0]
        if pmid in set_train:
            print >> fh_w_train, "\t".join([pmid, str(line_num)])
            d_pmid2id[pmid] = line_num
        if pmid in set_evaluation:
            print >> fh_w_evalu, "\t".join([pmid, str(line_num)])
            d_pmid2id[pmid] = line_num
    fh.close()
    fh_w_evalu.close()
    fh_w_train.close()

    print len(set_train)
    print len(set_evaluation)
    print len(d_pmid2id)

    fh_w_evalu = open("../data/medline_mesh_target_diseases.evalu.txt","w")
    fh_w_train = open("../data/medline_mesh_target_diseases.train.txt","w")
    print >> fh_w_evalu, "\t".join(["## pmid", "id", "major", "primary", "minors"])
    print >> fh_w_train, "\t".join(["## pmid", "id", "major", "primary", "minors"])
    for line in lines:
        fields = line.split("\t")
        if fields[0] in set_evaluation and fields[0] in d_pmid2id:
            print >> fh_w_evalu, "\t".join([fields[0], str(d_pmid2id[fields[0]])]+fields[1:])
        if fields[0] in set_train      and fields[0] in d_pmid2id:
            print >> fh_w_train, "\t".join([fields[0], str(d_pmid2id[fields[0]])]+fields[1:])
    fh_w_evalu.close()
    fh_w_train.close()
    

if __name__ == "__main__":
    medline_dir = "/home/csherrill/src/school/proj/all_medline/data/20120702/all_medline/all_medline_files/"
    #determine_stats(medline_dir)
    #pmids = select_files_of_interest()
    #generate_corpus_file(pmids, medline_dir, "../data/corpus_all_stop-stem.txt")
    #generate_dictionary()
    #generate_gensim_corpus(409600)
    #generate_gensim_models(409600)
    #generate_gensim_vs(3630013)
    #generate_test_train()
    #generate_gensim_vs(47350, restrict_ids=True) #project training only
    #generate_gensim_vs(47299, restrict_ids=True) #project evaluation only
    
    #generate_dictionary(restrict_disease=True)               # ~2 minutes
    #generate_gensim_corpus(head=None,  restrict_disease=True, train=True,  evalu=True ) # ~2 minutes
    #generate_gensim_corpus(head=47350, restrict_disease=True, train=True,  evalu=False) # ~2 minutes
    #generate_gensim_corpus(head=47299, restrict_disease=True, train=False, evalu=True ) # ~2 minutes

    #generate_gensim_models(head=None, restrict_disease=True) # ~31 m
    generate_gensim_vs(head=None, restrict_ids=False, restrict_disease=True)     # ~ 6m
    #generate_gensim_vs(47350, restrict_ids=True, restrict_disease=True) #project training only
    #generate_gensim_vs(47299, restrict_ids=True, restrict_disease=True) #project evaluation only
