import numpy as np
cimport numpy as np
from numpy cimport ndarray, float_t, int_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def kabsch(np.ndarray[float_t, ndim=2] P, np.ndarray[float_t, ndim=2] Q):
    cdef int N, D
    N, D = P.shape[0], P.shape[1]

    cdef np.ndarray[float_t, ndim=2] C = np.dot(P.T, Q)

    cdef np.ndarray[float_t, ndim=2] U
    cdef np.ndarray[float_t, ndim=2] V
    cdef np.ndarray[float_t, ndim=2] W
    cdef float d

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    U = np.dot(V, W)

    return U

@cython.boundscheck(False)
@cython.wraparound(False)
def kabsch_rotate(np.ndarray[float_t, ndim=2] P, np.ndarray[float_t, ndim=2] Q):
    cdef np.ndarray[float_t, ndim=2] U = kabsch(P, Q)
    P = np.dot(P, U)
    return P

@cython.boundscheck(False)
@cython.wraparound(False)
def centroid(np.ndarray[float_t, ndim=2] X):
    return X.mean(axis=0)

@cython.boundscheck(False)
@cython.wraparound(False)
def kabsch_rmsd(np.ndarray[float_t, ndim=2] P, np.ndarray[float_t, ndim=2] Q, **kwargs):
    Q -= centroid(Q)
    P -= centroid(P)
    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)

@cython.boundscheck(False)
@cython.wraparound(False)
def rmsd(np.ndarray[float_t, ndim=2] P, np.ndarray[float_t, ndim=2] Q):
    cdef int N = P.shape[0]
    return np.sqrt((np.sum((P - Q) ** 2)) / N)