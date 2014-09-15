#!/usr/bin/env python


import os

class medline(object):
    def __init__(self, medline_dir="./", pmids=None):
        self.medline_dir = medline_dir
        self.pmids = pmids

    def _iter_fns(self):
        for prefix_dir in os.listdir(self.medline_dir):
            for fn in os.listdir(self.medline_dir + prefix_dir):
                yield self.medline_dir + prefix_dir +"/"+ fn

    def _read_medline(self, fl):
        pass

    





if __name__ == "__main__":
    medline_dir = "/home/csherrill/src/school/proj/all_medline/data/20120702/all_medline/all_medline_files/"
    my_medline = medline(medline_dir)
    for fn in my_medline._iter_fns():
        print fn

