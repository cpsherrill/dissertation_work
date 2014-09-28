#!/usr/bin/env python

import networkx
import math
import random

def entropies2cliques(d_entropies, threshold):
    G = networkx.Graph()
    for key in d_entropies:
        G.add_node(key[0])
        G.add_node(key[1])
    for key in d_entropies:
        if d_entropies[key] < threshold:
            G.add_edge(*key)
    #print dir(networkx)
    return ( tuple(sorted(list(c))) for c in networkx.find_cliques(G) if len(c) >=3 )

if __name__ == "__main__":
    d = {}
    for i in range(10):
        for j in range(i,10):
            if i != j:
                d[(i,j)] = random.random()
    print [i for i in entropies2cliques(d, .4)]
