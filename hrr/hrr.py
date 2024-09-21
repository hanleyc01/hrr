'''Implementation of Holographic Reduced Representations
Code written by Mary Kelly and Eilene Tomkins-Flanagan
Provides HRR objects for use in modelling
For the original HRR paper see Tony Plate's (1995) IEEE paper 
on Holographic Reduced Representations.
Code also provides an inverse permutation helper function invPerm'''

import numpy as np # import linera algebra library
from numpy.fft import fft,ifft
import math # you also need to import basic math because python is silly
from collections.abc import Sequence

# find the inverse permutation given a permutation
def invPerm(perm):
    inv       = np.arange(perm.size) # initialize inv
    inv[perm] = np.arange(perm.size) # create inv from perm
    return inv

# class for Tony Plate's Holographic Reduced Representations
class HRR(Sequence):
    # Generate a vector of values sampled from a normal distribution
    # with a mean of zero and a standard deviation of 1/N
    def __init__(self,N=None,data=None, zero=False,large=0.8,small=0.2):
        self.large = large
        self.small = small
        if data is not None: # the vector is specified rather than random
            if isinstance(data, HRR):
                self.v = data.v
            else:
                self.v=np.array(data,dtype=float)
        elif zero and N is not None:
            self.v = np.zeros(N)
        elif N is not None: # create a random vector of N dims
            sd=1.0/math.sqrt(N)
            self.v=np.random.normal(scale=sd, size=N)
            self.v/=np.linalg.norm(self.v)
        else:
            raise Exception('Must specify size or data for HRR')

        self.scale = np.linalg.norm(self.v)

    # The multiplication-like operation is used to associate or bind vectors.
    def __mul__(self,other):
        if isinstance(other,HRR):
            return HRR(data=ifft(fft(self.v)*fft(other.v)).real)
        else:
            return HRR(data=self.v*other)

    def __rmul__(self,other):
        return self * other

    # exponentiation is used for fractional binding
    def __pow__(self,exponent):
        x=ifft(fft(self.v)**exponent).real
        return HRR(data=x)

    # The addition-like operation is used to superpose vectors or add them to a set.
    def __add__(self,other):
        if isinstance(other,HRR) or isinstance(other, HRRscale):
            return other + self.v
        else:
            return HRR(data=other + self.v)

    # allows us to specify the negative of a vector, -HRR
    def __neg__(self):
        return HRR(data=-self.v)

    # allows subtracting HRR from HRR
    def __sub__(self,other):
        return HRR(data=self.v-other.v)

    # An inverse can be defined such that binding with the inverse unbinds
    def __invert__(self):
        return HRR(data=self.v[np.r_[0,self.v.size-1:0:-1]])

    # Unbinding approximately cancels out binding
    # Unbinding in HRRs is implemented as the binding of the inverse of the cue to the trace.
    def __truediv__(self,other):
        if isinstance(other,HRR):
            return self * ~other
        else: # actually just divide if the divisor is not an HRR
            return HRR(data=self.v/other)

    # retrieve the dimensionality of the vector
    def __len__(self):
        return HRR.v.size

    # index into the vector
    def __getitem__(self, i):
        if isinstance(i,int):
            return self.v[i]
        else:
            return HRR(data=self.v[i])

    def __setitem__(self, key, value):
        self.v[key] = value

    # get the Euclidean length or magnitude of the vector
    # use self.scale most of the time, as it's pre-calculated
    def magnitude(self):
        return math.sqrt(self.v @ self.v)

    # compare two vectors using the vector cosine to measure similarity
    def __eq__(self,other):
        scale = self.scale * other.scale # scaling for normalization
        if scale==0:
            return 0
        return (self.v @ other.v)/scale

    # normalize to the unit vector, i.e., a magnitude or Euclidean length of 1
    def unit(self):
        return HRR(data=self.v / self.scale)

    # dimensionality of our vector-symbol
    # added by Eilene
    def size(self):
        return self.v.size

    # dimensionality of our vector-symbol
    def shape(self):
        return self.v.shape

    # typically the vector dot product
    # operator @
    # added by Eilene
    def __matmul__(self, other):
        if isinstance(other, HRR):
            return self.v @ other.v
        elif isinstance(other, np.ndarray) and len(other.shape) == 2:
            return HRR(data=self.v @ other)
        else:
            return self.v @ other

    # projection of self onto other
    # used in Widdow's PSI model
    # and quantum models of decision making
    def proj(self,other):
        return (self @ other.unit()).item()

    # implements PSI negation
    # returns self "without" the other
    # as self minus the projection of self onto onther
    def reject(self,other):
        return self - (self.proj(other))*other

    # created by Eilene
    # a variant addition operator
    # "a | b" is a if a has a large magnitude
    # otherwise it's b if a has a small magnitude
    # lastly, it could be both a and b if a has a middling magnitude
    def __or__(self,other):
        if self.scale > self.large:
            return self
        elif self.scale < self.small:
            return other
        else:
            return self+other
