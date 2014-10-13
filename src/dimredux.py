# dimredux.py

from numpy import array, cov, linalg, logical_and, zeros
from PIL import Image
import akando

## ############# PCA
def PCA( mat, D=2 ):
    """Principal Component Analysis"""
    a = mat - mat.mean(0)
    cv = cov( a.transpose() )
    evl, evc = linalg.eig( cv )
    V,H = mat.shape
    cffs = zeros( (V,D) )
    ag = abs(evl).argsort( )
    ag = ag[::-1]
    me = ag[:D]
    for i in range( V ):
        k = 0
        for j in me:
            cffs[i,k] = (mat[i] * evc[:,j]).sum()
            k += 1
    vecs = evc[:,me].transpose()
    print me, evl[me]
    return cffs, vecs
    
def Project( evecs, datavecs ):
    ND = len( datavecs ) # number of data vectors
    NE = len( evecs ) # number of eigenvectors
    cffs= zeros( (ND,NE) )
    for i in range( ND ):
        a = datavecs[i] * evecs
        cffs[i] = a.sum(1)  # NE dot products
    return cffs
