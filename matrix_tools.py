import numpy as np


def digaonal_well_seperated(n):
    A = np.diag(1./(1. + np.arange(n))) # diagonal matrix with well-separated maximum eigenvalues
    return A

def diagonal_clustered(n):
    A_clustered = np.diag(1 - 1./(1. + np.arange(n))) # diagonal matrix with clustered maximum eigenvalues
    return A_clustered

def digaonal_dominant(n,sparsity=1E-4):
    '''
    Create a diagonal-dominant mtrix with size n

    Parameters:

        :n: size of the matrix
        :sparsity: probability of being sparse in matrix. 

    '''
    
    A = np.zeros((n,n))
    for i in range(0,n):
        A[i,i] = 1E3*np.random.rand() 
        #A[i,i] = i+1
    A = A + sparsity*np.random.randn(n,n) 
    A = (A.T + A)/2 
    return A

def random_sparse(n,sparsity=1E-4,scale=100):
    A=np.random.rand(n,n)*scale
    idx=np.random.random_integers(low=0,high=n,size=(int(sparsity*n), int(sparsity*n)))
    A[idx]=0
    return A


def diag_non_tda(n,sparsity=1E-4):

    A = digaonal_dominant(n)
    C = sparsity*np.random.rand(n,n)

    return np.block([ [A,C],[-C.T,-A.T] ])

def symmetric_sparse(n,sparsity=1E-4):
    '''
    Build a sparse symmetric matrix 

    Parameters:

        :n: size of the matrix
        :sparsity: probability of being sparse in matrix. 

    '''


    # print('Dimension of the matrix',n,'*',n)

    A = np.zeros((n,n))
    for i in range(0,n) : 
        A[i,i] = i-9
    A = A + sparsity*np.random.randn(n,n)
    A = (A.T + A)/2
    return A

def normalize(v0):
    '''Calculate the norm.'''
    if np.ndim(v0)==2:
        return v0/np.sqrt((np.multiply(v0,v0.conj())).sum(axis=0))
    elif np.ndim(v0)==1:
        return v0/np.norm(v0)
    
def mgs(u,Q,MQ=None,M=None):
    '''
    Modified Gram-Schmidt orthogonalisation,
    The routine MGS orthogonalises the vector u vs. the columns of Q using the modified Gram-Schmidt approach.
    
    Parameters:
        :u: vector, the vector to be orthogonalized.
        :Q: matrix, the search space.
        :M: matrix/None, the matrix, if provided, perform M-orthogonal.
        :MQ: matrix, the matrix of M*Q, if provided, perform M-orthogonal.

    Return:
        vector, orthogonalized vector u.
    '''
    assert(np.ndim(u)==2)
    uH=u.T.conj()
    if MQ is None:
        MQ=M.dot(Q) if M is not None else Q
    for i in range(Q.shape[1]):
        s=uH.dot(MQ[:,i:i+1])
        u=u-s.conj()*Q[:,i:i+1]
    return u