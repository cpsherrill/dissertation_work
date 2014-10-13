#!/usr/bin/env python

import urllib
import os
from gensim import corpora, models, similarities
import gensim
import cPickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()

#stop_list = ["a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"]

stop_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
s_stop_list = set(stop_list)

bad_chars = ['.', '!', '#', '%', '$', "'", '&', '+', '*', '=', '?', '@', '[', ']', '\\', '_', '{', '}', '|', '~', '(', ')', ',', '-', ';', '<', '>', '/']
s_bad_chars = set(bad_chars)



def text2stop_stem(text):
    text = text.lower()
    for char1 in bad_chars:
        text = text.replace(char1, " ")
    words = filter(lambda x: x!= "", text.split(" "))
    words_stop = filter(lambda x: x not in s_stop_list, words)
    words_stop_stem = map(lambda x: ps.stem(x), words_stop)
    return words_stop_stem

class MyCorpus(object):
    def __init__(self, corpus_fn="", head=None, pmids=None, ignore_head=True):
        self.corpus_fn = corpus_fn
        self.head = head
        self.pmids = pmids
        self.ignore_head = ignore_head

    def __iter__(self):
        first_pmid = True
        fh = open(self.corpus_fn, "r")
        for line_num, line in enumerate(fh):
            if line_num % 200000 == 0:
                print line_num
            if not self.ignore_head and self.head and line_num+1 == self.head:
                break
            if line_num != 0:
                fields = line.strip().split("\t", 1)
                if self.pmids is None or int(fields[0]) in self.pmids:
                    if first_pmid:
                        print "reading pmid:", fields[0]
                        first_pmid = False
                    yield fields[1].split(" ")
        fh.close()

    def _iter_bow(self):
        for token_list in self:
            yield self.dictionary.doc2bow(token_list)

    def make_dictionary(self, fn):
        dictionary = corpora.Dictionary(self)
        print dictionary
        #dictionary.save(fn)
        token_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < 5]
        stop_ids = [dictionary.token2id[stopword] for stopword in stop_list if stopword in dictionary.token2id]
        dictionary.filter_tokens(token_ids + stop_ids) # remove words that appear in few documents or in stop_list
        dictionary.compactify()
        print dictionary
        dictionary.save(fn)
        self.dictionary = dictionary

    def get_dictionary(self, fn):
        self.dictionary = corpora.Dictionary.load(fn)

    def make_gensim_corpus(self, fn_mmcorpus, fn_tfidf):
        print "making corpus"
        corpora.MmCorpus.serialize(fn_mmcorpus, self._iter_bow())
        print "making tfidf"
        tfidf = models.TfidfModel(self._iter_bow())
        tfidf.save(fn_tfidf)

    def make_gensim_models(self, fn_mmcorpus, fn_tfidf, fn_lsi, fn_lda):
        self.mycorpus = corpora.MmCorpus(fn_mmcorpus)
        print "  corpus:", self.mycorpus
        self.tfidf = models.TfidfModel.load(fn_tfidf)
        print "  tfidf"
        self.corpus_tfidf = self.tfidf[self.mycorpus]
        print "  lsi"
        self.lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=500)
        self.lsi.save(fn_lsi)
        print "  lda"
        self.lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=500)
        self.lda.save(fn_lda)
    
    def make_gensim_v(self, fn_mmcorpus, fn_tfidf, fn_lsi, fn_v):
        self.mycorpus = corpora.MmCorpus(fn_mmcorpus)
        print "  corpus:", self.mycorpus
        self.tfidf = models.TfidfModel.load(fn_tfidf)
        print "  tfidf"
        self.corpus_tfidf = self.tfidf[self.mycorpus]
        print "  lsi"
        self.lsi = models.LsiModel.load(fn_lsi)
        self.v = gensim.matutils.corpus2dense(self.lsi[self.corpus_tfidf], len(self.lsi.projection.s)).T / self.lsi.projection.s

    def make_gensim_vs(self, fn_mmcorpus, fn_tfidf, fn_lsi, chunk_size=300, corpus_size=800, restrict_ids=None):
        num_chunks = corpus_size/chunk_size +1
        for chunk_id in range(num_chunks):
            print chunk_id+1, "of", num_chunks
            print corpus_size/num_chunks*(chunk_id+0), corpus_size/num_chunks*(chunk_id+1)
            if restrict_ids is None:
                my_iterator = (d for i,d in enumerate(corpora.MmCorpus(fn_mmcorpus)) if i>=(corpus_size/num_chunks*(chunk_id+0)) and i<(corpus_size/num_chunks*(chunk_id+1)))
            else:
                my_iterator = (d for i,d in enumerate(corpora.MmCorpus(fn_mmcorpus)) if i in restrict_ids )
            self.mycorpus = my_iterator
            print "  corpus:", self.mycorpus
            self.tfidf = models.TfidfModel.load(fn_tfidf)
            print "  tfidf"
            self.corpus_tfidf = self.tfidf[self.mycorpus]
            print "  lsi"
            self.lsi = models.LsiModel.load(fn_lsi)
            self.v = gensim.matutils.corpus2dense(self.lsi[self.corpus_tfidf], len(self.lsi.projection.s)).T / self.lsi.projection.s
            fh_w = open(fn_lsi+".projection.vgood.%04d.pickle" % (chunk_id, ),"w")
            cPickle.dump(self.v, fh_w)
            fh_w.close()

            
