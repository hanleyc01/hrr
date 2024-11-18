"""
Implementation of Holographic Reduced Representations
Code written by Mary Kelly and Eilene Tomkins-Flanagan
Provides HRR objects for use in modelling
For the original HRR paper see Tony Plate's (1995) IEEE paper 
on Holographic Reduced Representations.
Code also provides an inverse permutation helper function invPerm
"""
from __future__ import annotations

import math  # you also need to import basic math because python is silly
import typing as t
from collections.abc import Sequence

import numpy as np  # import linear algebra library
import numpy.typing as npt
from numpy.fft import fft, ifft

# holoarray = np.ndarray[t.Any, np.dtype[np.float64]]
holoarray = npt.NDArray[np.float64]
"Type of holographic arrays."

# _any_array = np.ndarray[t.Any, np.dtype[t.Any]]
_any_array = npt.NDArray[np.float64] | npt.NDArray[np.int_] | npt.NDArray[np.bool_]
"Type of any array"


def invPerm(perm: holoarray) -> holoarray:
    """
    Find the inverse permutation given a permutation.
    """
    inv = np.arange(perm.size)  # initialize inv
    inv[perm] = np.arange(perm.size)  # create inv from perm
    return t.cast(holoarray, inv)  # cast the array to our holographic vector type


# class for Tony Plate's Holographic Reduced Representations
class HRR(
    Sequence[float]
):  # Generate a vector of values sampled from a normal distribution
    """
    Class for Tony Plate's Holographic Reduced Representations.
    """

    large: float
    small: float
    v: holoarray
    scale: t.Any
    _shape: t.Any
    _size: int

    # with a mean of zero and a standard deviation of 1/N
    def __init__(
        self,
        N: int | None = None,
        data: HRR | npt.ArrayLike | None = None,
        zero: bool = False,
        large: float = 0.8,
        small: float = 0.2,
    ) -> None:
        self.large = large
        self.small = small
        if data is not None:  # the vector is specified rather than random
            if isinstance(data, HRR):
                self.v = data.v
            else:
                self.v = np.array(data, dtype=float)
        elif zero and N is not None:
            self.v = np.zeros(N)
        elif N is not None:  # create a random vector of N dims
            sd = 1.0 / math.sqrt(N)
            self.v = np.random.normal(scale=sd, size=N)
            self.v /= np.linalg.norm(self.v)
        else:
            raise Exception("Must specify size or data for HRR")

        self._shape = self.v.shape
        self._size = self.v.size
        self.scale = np.linalg.norm(self.v)

    def __mul__(self, other: HRR | _any_array | float | int) -> HRR:
        """
        The multiplication-like operation is used to associate or bind vectors.
        """
        if isinstance(other, HRR):
            return HRR(data=ifft(fft(self.v) * fft(other.v)).real)
        else:
            return HRR(data=self.v * other)

    def __rmul__(self, other: holoarray | float | int) -> HRR:  # type: ignore
        return self * other

    def __pow__(self, exponent: float | int) -> HRR:
        """
        Exponentiation used for fractional binding.
        """
        x = ifft(fft(self.v) ** exponent).real
        return HRR(data=x)

    def __add__(self, other: HRR | holoarray | float | int) -> HRR:
        """
        Addition-like operation used to superpose vectors or add them to a set.
        """
        if isinstance(other, HRR):
            return other + self.v
        else:
            return HRR(data=other + self.v)

    def __neg__(self) -> HRR:
        """
        Allows for the specification of the negative of a vector.
        """
        return HRR(data=-self.v)

    def __sub__(self, other: HRR) -> HRR:
        """
        Allows for subtracting HRR from HRR.
        """
        return HRR(data=self.v - other.v)

    def __invert__(self) -> HRR:
        """
        An inverse can be defined such that the binding with the inverse unbinds.
        """
        return HRR(data=self.v[np.r_[0, self.v.size - 1 : 0 : -1]])

    def __truediv__(self, other: HRR | _any_array | int | float) -> HRR:
        """
        Unbinding approximately cancels out binding.
        Unbinding in HRRs is implemented as the binding of the inverse of the
        cue to the trace.
        """
        if isinstance(other, HRR):
            return self * ~other
        else:  # actually just divide if the divisor is not an HRR
            return HRR(data=self.v / other)

    def __len__(self) -> int:
        """
        Retrieve the dimensionality of the vector.
        """
        return HRR.v.size

    @t.overload
    def __getitem__(self, key: int, /) -> float:
        ...

    @t.overload
    def __getitem__(self, key: slice, /) -> HRR:
        ...

    def __getitem__(self, key: slice | int, /) -> float | HRR:
        """
        Index into the vector.
        """
        if isinstance(key, int):
            return t.cast(float, self.v[key])
        else:
            return HRR(data=self.v[key])

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        self.v[key] = value

    def magnitude(self) -> float:
        """
        Get the Euclidean length or magnitude of the vector.
        Use `self.scale` most of the time, as it is precalculated.
        """
        return math.sqrt(self.v @ self.v)

    def __eq__(self, other: HRR) -> float:  # type: ignore
        """
        Compare two vectors using the vector cosine to measure similarity.
        """
        scale = self.scale * other.scale  # scaling for normalization
        if scale == 0:
            return 0.0
        else:
            return t.cast(float, (self.v @ other.v) / scale)

    def unit(self) -> HRR:
        """
        Normalize to the unit vector, i.e., a magniitude or Euclidean length
        of `1`.
        """
        return HRR(data=self.v / self.scale)

    # added by Eilene
    @property
    def size(self) -> int:
        """
        Dimensionality of our vector-symbol.
        """
        return self._size

    @property
    def shape(self) -> t.Any:
        """
        Shape of our vector-symbol.
        """
        return self._shape

    # typically the vector dot product
    # operator @
    # added by Eilene
    def __matmul__(self, other: HRR | _any_array) -> HRR | npt.ArrayLike:
        """
        Typically the vector dot product operator.
        """
        if isinstance(other, HRR):
            return self.v @ other.v
        elif isinstance(other, np.ndarray) and len(other.shape) == 2:
            return HRR(data=self.v @ other)
        else:
            return self.v @ other

    # projection of self onto other
    # used in Widdow's PSI model
    # and quantum models of decision making
    def proj(self, other: HRR) -> float: 
        """
        Projection of self onto other.  Used in Widdow's PSI model
        and quantum models of decision making.
        """
        return (self @ other.unit()).item() #type: ignore

    # implements PSI negation
    # returns self "without" the other
    # as self minus the projection of self onto onther
    def reject(self, other: HRR) -> HRR:
        """
        PSI negation, returns `self` "without" `other`;
        as self minus the projection of self onto the other.
        """
        return self - (self.proj(other)) * other # type: ignore

    # created by Eilene
    # a variant addition operator
    # "a | b" is a if a has a large magnitude
    # otherwise it's b if a has a small magnitude
    # lastly, it could be both a and b if a has a middling magnitude
    def __or__(self, other: HRR) -> HRR:
        """
        Variant addition operator. 
        `a | b` is `a` if `a` has a large magnitude, otherwise `b` 
        if `a` has a small magnitude, else it is `a + b`.
        """
        if self.scale > self.large:
            return self
        elif self.scale < self.small:
            return other
        else:
            return self + other