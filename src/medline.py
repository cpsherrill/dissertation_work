#!/usr/bin/env python


import os
import nlp

class mesh(object):
    def __init__(self, raw=""):
        self.raw = raw
        self._parse()
    
    def _parse(self):
        self.key = "*" in self.raw
        parts = self.raw.replace("*","").split("/")
        self.major = parts[0]
        self.minors = []
        if len(parts) > 1:
            self.minors = parts[1:]
        


class medline(object):
    def __init__(self, medline_dir="./", pmids=None):
        self.medline_dir = medline_dir
        self.pmids = pmids

    def _iter_fps(self):
        prefix_dirs = sorted([p for p in os.listdir(self.medline_dir) if "." not in p])
        for i, prefix_dir in enumerate(prefix_dirs):
            if i%10 == 0:
                print i, "of", len(prefix_dirs), prefix_dir
            for fn in sorted(os.listdir(self.medline_dir + prefix_dir)):
                yield self.medline_dir + prefix_dir +"/"+ fn

    def _read_medline(self, fp):
        fh = open(fp,"r")
        lines = fh.readlines()
        fh.close()
        if len(lines) < 5:
            return None
        d_a = {}          # dictionary of found fields
        a = ""
        d_info = {}       # organized info
        for line in lines:
            line = line.replace("\r","").replace("\n","")
            if line != "":
                pa = line[:4].strip() # potential new field
                if pa != "":
                    a = pa
                if a in d_a:
                    d_a[a] += [line[6:]]
                else:
                    d_a[a] =  [line[6:]]
                d_info["lang"]       = d_a.get("LA", [None] )[0]
                d_info["abstract"]   = " ".join(d_a.get("AB", [""])).replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ")
                d_info["ab_word_ct"] = len(d_info["abstract"].split(" "))
                d_info["date"]       = d_a.get("DA", [None])[0]
                d_info["mesh_ct"]    = len(d_a.get("MH", []))
                d_info["source"]     = d_a
        return d_info
        
    def _read_pmids(self, pmids=None, log_fn=None):
        if log_fn:
            log_fh = open(log_fn, "w")
            print >> log_fh, "\t".join(["## pmid", "lang", "ab_word_ct", "date", "mesh_ct"])
        
        for fp in self._iter_fps():
            try:
                pmid = fp.split("/")[-1].split(".")[0]
            except:
                print fp
            if pmids is None or pmid in pmids:
                d_info = self._read_medline(fp)
                if d_info is None: #pmid not available
                    pass
                    #if log_fn:
                    #    print >> log_fh, "\t".join(map(str, [pmid, None, None, None, None ]))
                    #    print pmid
                else:
                    d_info["pmid"] = pmid
                    if log_fn:
                        print >> log_fh, "\t".join(map(str, [pmid, d_info["lang"], d_info["ab_word_ct"], d_info["date"], d_info["mesh_ct"] ]))
                    yield (pmid, d_info)
        if log_fn:
            log_fh.close()

    def get_processed_abstracts(self, pmids=None):
        for i, (pmid, d_info) in  enumerate(self._read_pmids(pmids=pmids, log_fn=None)):
            abstract = d_info.get("abstract","")
            ab_words_stop_stem = nlp.text2stop_stem(abstract)
            yield pmid, " ".join(ab_words_stop_stem)


if __name__ == "__main__":
    medline_dir = "/home/csherrill/src/school/proj/all_medline/data/20120702/all_medline/all_medline_files/"
    my_medline = medline(medline_dir)
    my_medline._read_pmids(log_fn = "../data/medline_stats.txt")

