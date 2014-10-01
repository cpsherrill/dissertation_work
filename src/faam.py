# FAAM convert on 9 May 2000
# JM Kinser

from numpy import array, less, dot, not_equal, int8, zeros

class FAAM:
    "Fast Analog Associative Memory"
    
    def __init__( self, maxtrainers, maxr, xdim ):
        self.maxt = maxtrainers
        self.maxr = maxr
        self.xdim = xdim
        # bucket for training vectors
        self.Tx = zeros( (self.maxt,self.xdim), float )
        self.Ty = zeros( self.maxt, int )
        # allocate network innards
        self.Rs = zeros( (self.maxr,self.xdim), float )
        self.Gamma = zeros( self.maxr, float )
        self.Rtable = zeros( (self.maxt, self.maxr), int8 )
        self.lastt, self.lastr, self.tcnt = 0,0,0
    
    def Clear( self ):
        self.Tx = 0
        self.Ty = 0
        self.Gamma = 0
        self.Rtable = 0
        self.tcnt, self.lastt, self.lastr = 0,0,0
    
    def Logic( self, I, J ):
        if( dot( self.Tx[I], self.Rs[J]) > self.Gamma[J]) : return 1
        else: return 0
    
    def SetTrainer( self, X, Y ):
        self.Tx[ self.lastt] = X
        self.Ty[ self.lastt] = Y
        for i in range( self.lastr ):
            self.Rtable[ self.lastt, i ] = self.Logic( self.lastt, i )
        self.lastt = self.lastt+1
        if self.lastt >= self.maxt :
            print "Number of trainers has reached a maximum"
    
    def NewR( self, I, J ):
        psub = self.Tx[I] - self.Tx[J]
        phalf = (self.Tx[I] + self.Tx[J])/2.
        alpha = dot( psub, psub )
        if alpha == 0 : print 'Trainers are equivalent.',I,J
        alpha = dot( phalf,psub)/dot( psub,psub)
        self.Rs[ self.lastr] = alpha * psub
        self.Gamma[ self.lastr] = dot(self.Rs[self.lastr], self.Rs[self.lastr])
        self.lastr = self.lastr + 1
        if self.lastr > self.maxr-2: print 'Maximum R reached.'
        J=self.lastr-1
        for i in range (self.lastt): self.Rtable[i,J] = self.Logic( i,J )
        self.tcnt = self.tcnt + 1
    
    def Compare( self, R1, R2 ):
        ans = 1
        if self.lastr > 0:
            if (not_equal( R1[:self.lastr], R2[:self.lastr] )).sum() > 0: ans =0
        return ans
     
    def Train(self):
        for i in range( self.lastt ):
            for j in range( self.lastr): self.Rtable[i,j] = self.Logic(i,j)
            for j in range( i ):
                if self.Compare(self.Rtable[i], self.Rtable[j])==1 and self.Ty[i] != self.Ty[j]:
                    self.NewR( i,j )
            if self.tcnt >= 10:
                self.Purge()
                self.tcnt = 0
        self.Purge()
    
    def Purge1( self ):
        # eliminate one column at a time.
        i=0
        #while i<self.lastr:

        slr = self.lastr
        for i in xrange(slr-1,-1,-1):
            # i is the column being eliminated
            kill = 1
            # are there any combos that make this fail
            for j in range( self.lastt):
                r1 = self.Rtable[j]+0
                r1[i] = 0
                for k in range( j ):
                    r2 = self.Rtable[k]+0
                    r2[i] = 0
                    a = self.Compare( r1, r2 )
                    if a==1 and self.Ty[j]!= self.Ty[k]: kill = 0
            if kill == 1:
                print 'Removing', i
                self.Rtable[:,i:self.lastr-1] = self.Rtable[:,i+1:self.lastr]
                self.Rtable[:,self.lastr-1] = 0
                self.Gamma[i:self.lastr-1] = self.Gamma[i+1:self.lastr]
                self.Gamma[self.lastr-1] = 0
                self.Rs[i:self.lastr-1] = self.Rs[i+1:self.lastr]
                self.Rs[self.lastr-1] = 0
                self.lastr = self.lastr - 1
            #else:
            #    i = i+1

    def Purge(self):
        i = 0
        slr = self.lastr
        for i in xrange(slr-1,-1,-1):
            ok = 1 # ok to remove this column
            for j in xrange( self.lastt ):
                r1 = self.Rtable[j,:self.lastr] + 0
                r1[i] = 0
                temp = self.Rtable[:self.lastt,:self.lastr] + 0
                temp[:,i] = 0 # remove this column
                a = abs(r1-temp).sum(1)
                ndx = (a==0).nonzero()[0] # ndx of match rows
                ndx = array( ndx )
                # are any of a difft class
                cnt = (self.Ty[j] != self.Ty[ndx]).sum()
                if cnt > 0:
                    ok = 0
                    break
            if ok:
                #print 'Zapping',i
                self.Rtable[:,i:self.lastr-1] = self.Rtable[:,i+1:self.lastr]
                self.Rtable[:,self.lastr-1] = 0
                self.Gamma[i:self.lastr-1] = self.Gamma[i+1:self.lastr]
                self.Gamma[self.lastr-1] = 0
                self.Rs[i:self.lastr-1] = self.Rs[i+1:self.lastr]
                self.Rs[self.lastr-1] = 0
                self.lastr = self.lastr - 1

    def Recall( self, X ):
        bits = zeros( self.maxr, int8 )
        for i in range( self.lastr):
            if dot( X, self.Rs[i]) > self.Gamma[i] : bits[i] = 1
        # is this in the table?
        ans = -1
        for i in range( self.lastt):
            if self.Compare( self.Rtable[i], bits) ==1: ans = self.RTy[i]
        return ans
    
    def Recall2d( self, M ):
        # M is V,H,N were N is length of vecs
        V,H,N = M.shape
        dots = zeros((V,H,self.lastr), int )
        for i in range( self.lastr ):
            dots[:,:,i] = dot( M, self.Rs[i] ) > self.Gamma[i]
        # decode
        answ = zeros((V,H), int )-1
        for i in range( self.lasttr ):
            print i,
            a = abs( dots - self.Rtable[i,:self.lastr]).sum(2)
            mask = a==0 # places were there is an exact match
            answ = (1-mask)*answ + mask*self.RTy[i] 
        return answ
        
    def PurgeRtable( self ):
        dct = {}
        for i in range( self.lastt ):
            k = tuple( self.Rtable[i,:self.lastr] )
            if dct.has_key( k ):
                dct[k].append( self.Ty[i] )
            else:
                dct[k] = [self.Ty[i]]
        # dct is a dict of unique entries.  All data should be same for each entry
        self.Rtable -= self.Rtable # zero it out
        self.RTy = zeros( self.maxt, int )
        ct = 0
        for i in dct.iterkeys():
            self.Rtable[ct,:self.lastr] = i
            self.RTy[ct] = dct[i][0]
            ct += 1
        self.lasttr = len( dct )
    
    def ShowTable( self ):
        # shows entries of Rtable
        print self.Rtable[:self.lastt,:self.lastr]

        
