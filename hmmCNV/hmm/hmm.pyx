import cython

import numpy as np
cimport numpy as np
np.import_array()
import os


# Declare interface to C code
cdef extern from "utilities.c":
    # forward_backward utilities
    void transposeSquareInPlace(double *out_m, double *in_m, int K)
    void multiplyInPlace(double *result, double *u, double *v, int K)
    double normalizeInPlace(double *A, int N)
    void multiplyMatrixInPlace(double *result, double *trans, double *v, int K)
    void outerProductUVInPlace(double *out, double *u, double *v, int K)
    void componentVectorMultiplyInPlace(double *out, double *u, double *v, int L)

    # viterbi utilities
    void addVectors(double *out, double *u, double *v, int L)
    void setVectorToValue_int(double *A, int value, int L)
    void maxVectorInPlace(double *out_value, int *out_index, double *A, int L)


def get_include():
    """
    Get file directory if C headers
    
    Arguments:
    ---------
        None
    Returns:
    ---------
        str (Directory to header files)
    """

    return os.path.split(os.path.realpath(__file__))[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def forward_backward(np.ndarray[double, ndim=1, mode="fortran"] piZ,
                     np.ndarray[double, ndim=2, mode="fortran"] A,
                     np.ndarray[double, ndim=2, mode="fortran"] py):
    """
    """

    cdef double[:] init_state_distrib = piZ
    cdef double[:,:] transmat = A
    cdef double[:,:] obslik = py

    cdef int K = piZ.size
    cdef double[:] transmatT_data = np.zeros(K*K, dtype=np.double, order="F")
    transposeSquareInPlace(&transmatT_data[0], &transmat[0,0], K)

    cdef int T = py.shape[1]

    cdef double[:] scale_data = np.zeros(T, dtype=np.double, order="F")
    cdef double[:,:] alpha_data = np.zeros((K, T), dtype=np.double, order="F")
    cdef double[:,:] beta_data = np.zeros((K, T), dtype=np.double, order="F")
    cdef double[:,:] gamma_data = np.zeros((K, T), dtype=np.double, order="F")
    cdef double[:] loglik_data = np.zeros(1, dtype=np.double, order="F")

    # ********* FOWARD **********
    cdef int t = 0
    multiplyInPlace(&alpha_data[0,t], &init_state_distrib[0], &obslik[0,t], K)
    scale_data[t] = normalizeInPlace(&alpha_data[0,t], K)

    cdef double[:] m_data = np.zeros(K, dtype=np.double, order="F")

    for t in range(1,T):
        multiplyMatrixInPlace(&m_data[0], &transmatT_data[0], &alpha_data[0,t-1], K)
        multiplyInPlace(&alpha_data[0,t], &m_data[0], &obslik[0,t], K)
        scale_data[t] = normalizeInPlace(&alpha_data[0,t], K)

    loglik_data[0] = 0
    for t in range(T):
        loglik_data[0] += np.log(scale_data[t])

    # ********* BACKWARD **********

    t = T - 1
    for d in range(K):
        beta_data[:,t] = 1
        gamma_data[:,t] = alpha_data[:,t]

    cdef double[:] b_data = np.zeros(K, dtype=np.double, order="F")
    cdef double[:,:,:] eta_data = np.zeros((K,K,T), dtype=np.double, order="F")
    cdef double[:,:] squareSpace_data = np.zeros((K,K), dtype=np.double, order="F")
    
    # We have to remember that the 1:T range in R is 0:(T-1) in C
    for t in range(T-2, -1, -1):

        # setting beta
        multiplyInPlace(&b_data[0], &beta_data[0,t+1], &obslik[0,t+1], K)
        multiplyMatrixInPlace(&m_data[0], &A[0,0], &b_data[0], K)
        normalizeInPlace(&m_data[0], K)

        for d in range(K):
            beta_data[d,t] = m_data[d]
        # using "m" again as valueholder

        # setting eta, whether we want it or not in the output
        outerProductUVInPlace(&squareSpace_data[0,0], &alpha_data[0,t], &b_data[0], K)
        componentVectorMultiplyInPlace(&eta_data[0,0,t], &transmat[0,0], &squareSpace_data[0,0], K * K)
        normalizeInPlace(&eta_data[0,0,t], K * K)

        # Setting gamma
        multiplyInPlace(&m_data[0], &alpha_data[0,t], &beta_data[0,t], K)
        normalizeInPlace(&m_data[0], K)
        for d in range(K):
            gamma_data[d,t] = m_data[d]

    results = {}
    results["rho"] = np.asarray(gamma_data)
    results["alpha"] = np.asarray(alpha_data)
    results["beta"] = np.asarray(beta_data)
    results["xi"] = np.asarray(eta_data)
    results["loglik"] = np.asarray(loglik_data)
    
    return results


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def viterbi(np.ndarray[double, ndim=1, mode="fortran"] piZ,
            np.ndarray[double, ndim=2, mode="fortran"] A,
            np.ndarray[double, ndim=2, mode="fortran"] py):
    """
    """

    cdef int K = piZ.size

    if A.shape[0] != K or A.shape[1] != K:
        raise IndexError("The transition matrix must be of size %d x %d", K, K)
    
    cdef int T = py.shape[1]

    if py.shape[0] != K:
        raise IndexError("The observed likelihoods must have %d rows", K)
        
    cdef double[:,:] delta_data = np.zeros((K, T), dtype=np.double, order="F")
    cdef double[:] d_data = np.zeros(K, dtype=np.double, order="F")
    cdef double[:] loglik_data = np.zeros(1, dtype=np.double, order="F")
    cdef int[:,:] psi_data = np.zeros((K, T), dtype=np.intc, order="F")
    cdef int[:] path_data = np.zeros(T, dtype=np.intc, order="F")

    cdef int t = 0
    #addVectors(delta + t * K, prior, obslik + t * K, K);
    addVectors(&delta_data[0,t], &piZ[0], &py[0,t], K)
    #setVectorToValue_int(psi + t * K, 0, K);
    #setVectorToValue_int(&psi_data[0,t], 0, K)

    # FORWARD
    for t in range(1,T):
        for j in range(K):
            #addVectors(d, delta + (t - 1) * K, transmat + j * K, K);
            addVectors(&d_data[0], &delta_data[0,t-1], &A[0,j], K)
            #maxVectorInPlace(delta + j + t * K, psi + j + t * K, d, K);
            maxVectorInPlace(&delta_data[j,t], &psi_data[j,t], &d_data[0], K)
            #delta[j + t * K] += obslik[j + t * K];
            delta_data[j,t] += py[j,t]
    
    # BACKWARD
    t = T - 1
    #maxVectorInPlace(d, path + t, delta + t * K, K); // Use d[0] as temp variable
    maxVectorInPlace(&d_data[0], &path_data[t], &delta_data[0,t], K) # Use d[0] as temp variable
    loglik_data[0] = d_data[0]

    for t in range(T-2, -1, -1):
        #path[t] = psi[path[t+1] + (t+1)*K]
        path_data[t] = psi_data[path_data[t+1],t+1]
    
    cdef double[:,:] changes_data = np.zeros((4, T), dtype=np.double, order="F")
    changesCounter = 0
    cdef int ind1
    cdef int ind2
    ind1, ind2 = np.unravel_index(changesCounter + 0 * T, (4,T), order="F")
    changes_data[ind1, ind2] = 0
    ind = np.unravel_index(changesCounter + 1 * T, (4,T), order="F")
    changes_data[ind[0], ind[1]] = 0 # overwritten
    ind = np.unravel_index(changesCounter + 2 * T, (4,T), order="F")
    changes_data[ind[0], ind[1]] = path_data[0]
    ind = np.unravel_index(changesCounter + 3 * T, (4,T), order="F")
    changes_data[ind[0], ind[1]] = 0
    changesCounter = 1

    for t in range(1,T):
        if path_data[t] != path_data[t-1]:
            ind = np.unravel_index(changesCounter + 0 * T, (4,T), order="F")
            changes_data[ind[0], ind[1]] = t
            ind = np.unravel_index((changesCounter - 1) + 1 * T, (4,T), order="F")
            changes_data[ind[0], ind[1]] = t - 1
            ind = np.unravel_index(changesCounter + 2 * T, (4,T), order="F")
            changes_data[ind[0], ind[1]] = path_data[t]
            ind = np.unravel_index(changesCounter + 3 * T, (4,T), order="F")
            changes_data[ind[0], ind[1]] = 0
            changesCounter+=1

    ind = np.unravel_index((changesCounter - 1) + 1 * T, (4,T), order="F")
    changes_data[ind[0], ind[1]] = T - 1

    # Reformat to segs
    cdef double[:,:] segs_data = np.zeros((changesCounter, 4), dtype=np.double, order="F")
    for t in range(changesCounter):
        seg_ind = np.unravel_index(t + 0 * changesCounter, (changesCounter, 4), order="F")
        ind = np.unravel_index(t + 0 * T, (4,T), order="F")
        segs_data[seg_ind[0], seg_ind[1]] = changes_data[ind[0], ind[1]] + 1.0

        seg_ind = np.unravel_index(t + 1 * changesCounter, (changesCounter, 4), order="F")
        ind = np.unravel_index(t + 1 * T, (4,T), order="F")
        segs_data[seg_ind[0], seg_ind[1]] = changes_data[ind[0], ind[1]] + 1.0
        
        seg_ind = np.unravel_index(t + 2 * changesCounter, (changesCounter, 4), order="F")
        ind = np.unravel_index(t + 2 * T, (4,T), order="F")
        segs_data[seg_ind[0], seg_ind[1]] = changes_data[ind[0], ind[1]] + 1.0
        
        seg_ind = np.unravel_index(t + 3 * changesCounter, (changesCounter, 4), order="F")
        ind = np.unravel_index(t + 3 * T, (4,T), order="F")
        segs_data[seg_ind[0], seg_ind[1]] = changes_data[ind[0], ind[1]]
    
    results = {}
    results["path"] = np.asarray(path_data)
    results["loglik"] = np.asarray(loglik_data)
    results["seg"] = np.asarray(segs_data)

    return results