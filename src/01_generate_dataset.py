#!/usr/bin/env python

import cPickle
import medline
import nlp


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

def generate_dictionary():
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt")
    mycorpus.make_dictionary("../data/gensim_complete_corpus.dict")

def generate_gensim_corpus():
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt")
    mycorpus.get_dictionary("../data/gensim_complete_corpus.dict")
    mycorpus.make_gensim_corpus("../data/gensim_complete_corpus.mm", "../data/gensim_complete_corpus.tfidf")
    
def generate_gensim_models():
    mycorpus = nlp.MyCorpus("../data/corpus_all_stop-stem.txt")
    mycorpus.get_dictionary("../data/gensim_complete_corpus.dict")
    mycorpus.make_gensim_models("../data/gensim_complete_corpus.mm", "../data/gensim_complete_corpus.tfidf", "../data/gensim_complete_corpus.lsi", "../data/gensim_complete_corpus.lda")

if __name__ == "__main__":
    medline_dir = "/home/csherrill/src/school/proj/all_medline/data/20120702/all_medline/all_medline_files/"
    #determine_stats(medline_dir)
    #pmids = select_files_of_interest()
    #generate_corpus_file(pmids, medline_dir, "../data/corpus_all_stop-stem.txt")
    #generate_dictionary()
    #generate_gensim_corpus()
    generate_gensim_models()
    
