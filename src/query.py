#!/usr/bin/env python

from optparse import OptionParser
import fileinput
import sys

def parse_header(line):
    fields = line[3:].lower().replace("\n","").replace("\r","").split("\t")
    return ["<%s>" % (f.strip().replace(" ","_"), ) for f in fields]

def get_header(fh_r):
    lines_read = 0
    while True:
        line = fh_r.readline()
        lines_read += 1
        if not line.startswith("##"):
            sys.stderr.write('The header line does not start with "## "\n')
            return None, lines_read
        if line.startswith("###"):
            # this is an acceptable notes line 
            continue
        if line.startswith("## "):
            return parse_header(line), lines_read
        sys.stderr.write('The header line does not start with "## "\n')
        return None, lines_read

def print_list(fh_r, header, lines_read):
    d = {}
    for field_name in header:
        d[field_name]=[]
    for line_num, line in enumerate(fh_r):
        lines_read += 1
        if line_num == 5:
            break
        fields = line.replace("\n","").replace("\r","").split("\t")
        for field_num, field_name in enumerate(header):
            try:
                d[field_name] += [fields[field_num]]
            except:
                #sys.stderr.write('Line (%s) does not have enough fields\n' % (lines_read, ))
                #sys.exit(1)
                d[field_name] += [""]
    for field_name in header:
        print "\t".join([field_name]+d[field_name])


def print_group(fh_r, header, lines_read, group_fields):
    key_names = group_fields.split("|")
    key_ids = []
    for key_name in key_names:
        try:
            key_ids += [header.index(key_name)]
        except:
            sys.stderr.write('Unknown field (%s)\n' % (key_name, ))
            sys.exit(1)
    print "## " + "\t".join(key_names+["<count>"])
    d = {}
    for line_num, line in enumerate(fh_r):
        lines_read += 1
        fields = line.replace("\n","").replace("\r","").split("\t")
        line_key = []
        for key_id in key_ids:
            try:
                line_key += [fields[key_id]]
            except:
                sys.stderr.write('Line (%s) does not have enough fields\n' % (lines_read, ))
                sys.exit(1)
            line_key_str = "\t".join(line_key)
            if line_key_str not in d:
                d[line_key_str] = 0
            d[line_key_str] += 1
    for key in sorted(d.keys()):
        print "\t".join([key, str(d[key])])



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-l", "--list", dest="list",
                      action="store_true", default=False,
                      help="list examples from first 5 records")
    parser.add_option("-g", "--group",
                      action="store", type="string", dest="group",
                      help="count records by the key defined by the field list")

    (options, args) = parser.parse_args()
    
    #fh_r = fileinput.input()
    fh_r = sys.stdin
    header, lines_read = get_header(fh_r)
    if header is None:
        sys.exit(1)

    if options.list:
        print_list(fh_r, header, lines_read)
    elif options.group:
        print_group(fh_r, header, lines_read, options.group)
