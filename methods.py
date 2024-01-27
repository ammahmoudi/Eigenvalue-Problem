import numpy as np
import matrix_tools as mt
import time

from scipy import sparse as sps
from scipy.linalg import eigh,inv
from scipy.linalg.lapack import dsyev
from numpy.linalg import norm
from scipy.sparse import linalg as lin
from scipy.sparse.linalg import inv as spinv
import pdb,time,warnings
import jd

def power_iteration(A,tol=1e-10,maxiter=1000,use_rayleigh=False,calc_min=False,Sigma=0):
    '''
    Power iteration is used to find the highest eigenvalue one at a time.
    https://github.com/sreeganb/davidson_algorithm/

    '''
    start_time = time.time()
    A=A-Sigma
    if calc_min: 
        # A_orginal=A
        A=np.linalg.pinv(A)

    n=A.shape[0]
    # Build a random trial vector
    B=np.random.rand(n)
    j=0
    norm_mat=np.zeros(2)
    while j<maxiter:
        if use_rayleigh:
            rcoeff = np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
            rmat = rcoeff*np.eye(n)
            C = np.dot(np.linalg.inv(A-rcoeff),B)
        else:
            C = np.dot(A,B)
        norm_c = np.linalg.norm(C)
        B = C/(norm_c)
        j=j+1
        # print(j)
        if j==1: 
            print ('just the first iteration, give me a break')
            norm_mat[0]=norm_c
        else: 
            norm_mat[1] = norm_mat[0]
            norm_mat[0] = norm_c
            diff = abs(norm_mat[1] - norm_mat[0])
            if diff < tol:
                print ('power iteration converged at iteration number:', j)
                break
            else:
                continue

    approx = np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
    if calc_min: approx=1./approx
    approx += Sigma
    end_time = time.time() 
    print ('power iteration dominant eigenvalue=', approx)
    print ('Power itration time:',end_time-start_time)


    w, v = np.linalg.eig(A)
    w=np.sort(w)
    diff = 1./w[-1] - approx
    print ('exact dominant eigenvalue=',1./w[-1] if calc_min else w[-1])
    print ('difference:', diff)

    return approx


def davidson(A,v0=None,tol=1e-10,maxiter=1000):
    '''
    The Davidson's algorithm.
    
    Parameters:
        :A: matrix, the input matrix.
        :v0: 2D array, the initial subspace.
        :tol: float, the tolerence.
        :maxiter: int, the maximum number of iteration times.

    Return:
        tuple of (e,v), e is the eigenvalues and v the eigenvector e is the eigenvalues and v the eigenvectors.
    '''


    start_davidson = time.time() 
    N=A.shape[0]
    # A=A.tocsr()
    DA_diag=A.diagonal()
    if v0 is None:
        v0=np.random.random((N,1))
    elif np.ndim(v0)==1: 
        v0=v0[:,np.newaxis]
    v0=mt.normalize(v0)
    Av=A.dot(v0)
    AV=Av
    V=v0
    #initialise projected matrix.
    G=v0.T.conj().dot(Av)
    for i in range(maxiter):
        ei,vi=eigh(G)
        #compute largest Ritz value theta, and Ritz vector u.
        imax=np.argmax(ei)
        theta,u=ei[imax],V.dot(vi[:,imax:imax+1])
        #get the residual
        r=AV.dot(vi[:,imax:imax+1])-theta*u
        if norm(r)<tol:
            return theta,u
        print ('%s ||r|| = %s, e = %s'%(i,norm(r),theta))
        #compute the correction vector z
        z=-1./(DA_diag-theta)[:,np.newaxis]*r
        z=mt.mgs(z,V)
        z=mt.normalize(z)

        Av=A.dot(z)
        #add z to search space.
        AV=np.concatenate([AV,Av],axis=1)
        #update G, G=UAU.H
        gg=[[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]]
        G=np.bmat([[G,V.T.conj().dot(Av)],[Av.T.conj().dot(V),Av.T.conj().dot(z)]])
        V=np.concatenate([V,z],axis=1)
    end_davidson = time.time()

       # End of block Davidson. Print results.

    print("davidson = ", theta,";",
        end_davidson - start_davidson, "seconds")
    

    return theta,u

def davidson_2(A,k=None,eig=1,tol=1e-10,mmax=1000):
    
    n=A.shape[0]
    if k is None:k=2*eig
    t = np.eye(n,k)			# set of k unit vectors as guess
    V = np.zeros((n,n))		# array of zeros to hold guess vec
    I = np.eye(n)			# identity matrix same dimen as A

    # Begin block Davidson routine

    start_davidson = time.time()

    for m in range(k,mmax,k):
        if m <= k:
            for j in range(0,k):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
            theta_old = 1 
        elif m > k:
            theta_old = theta[:eig]
        V[:,:m],R = np.linalg.qr(V[:,:m])
        T = np.dot(V[:,:m].T,np.dot(A,V[:,:m]))
        THETA,S = np.linalg.eig(T)
        idx = THETA.argsort()
        theta = THETA[idx]
        s = S[:,idx]
        for j in range(0,k):
            w = np.dot((A - theta[j]*I),np.dot(V[:,:m],s[:,j]))
            q = w/(theta[j]-A[j,j])
            V[:,(m+j)] = q
        norm = np.linalg.norm(theta[:eig] - theta_old)
        if norm < tol:
            break

    end_davidson = time.time()

    # End of block Davidson. Print results.

    print("davidson = ", theta[:eig],";",
        end_davidson - start_davidson, "seconds")
    return theta[:eig], s[:eig]

def davidson_3(A,k=None,neig=1,tol=1e-10,mmax=1000):
    '''
    The Block Davidson method ca be used to solve for a number of the lowest or highest few Eigenvalues of a symmetric matrix.
    https://github.com/sreeganb/davidson_algorithm/
    Important: Input matrix must be symmetric
    '''
    #-------------------------------------------------------------------------------
# Attempt at Block Davidson algorithm 
# Sree Ganesh (sreeuci@gmail.com)
# Summer 2017
#-------------------------------------------------------------------------------


    n=A.shape[0]
    # Setup the subspace trial vectors
    if k is None:k=2*neig
    print ('No. of start vectors:',k)
    print ('No. of desired Eigenvalues:',neig)
    t = np.eye(n,k) # initial trial vectors
    v = np.zeros((n,n)) # holder for trial vectors as iterations progress
    I = np.eye(n) # n*n identity matrix
    ritz = np.zeros((n,n))
    f = np.zeros((n,n))
    #-------------------------------------------------------------------------------
    # Begin iterations  
    #-------------------------------------------------------------------------------
    start = time.time()
    iter = 0
    for m in range(k,mmax,k):
        iter = iter + 1
        print ("Iteration no:", iter)
        if iter==1:  # for first iteration add normalized guess vectors to matrix v
            for l in range(m):
                v[:,l] = t[:,l]/(np.linalg.norm(t[:,l]))
        # Matrix-vector products, form the projected Hamiltonian in the subspace
        T = np.linalg.multi_dot([v[:,:m].T,A,v[:,:m]]) # selects fastest evaluation order
        w, vects = np.linalg.eig(T) # Diagonalize the subspace Hamiltonian
        j = 0
        s = w.argsort()
        ss = w[s]
        #***************************************************************************
        # For each eigenvector of T build a Ritz vector, precondition it and check
        # if the norm is greater than a set threshold.
        #***************************************************************************
        for i in range(m): #for each new eigenvector of T
            f = np.diag(1./ np.diag((np.diag(np.diag(A)) - w[i]*I)))
    #        print (f)
            ritz[:,i] = np.dot(f,np.linalg.multi_dot([(A-w[i]*I),v[:,:m],vects[:,i]]))
            if np.linalg.norm(ritz[:,i]) > 1e-7 :
                ritz[:,i] = ritz[:,i]/(np.linalg.norm(ritz[:,i]))
                v[:,m+j] = ritz[:,i]
                j = j + 1
        q, r = np.linalg.qr(v[:,:m+j-1])
        for kk in range(m+j-1):
            v[:,kk] = q[:,kk]
        for i in range(neig):
            print (ss[i])
        if iter==1: 
            check_old = ss[:neig]
            check_new = 1
        elif iter==2:
            check_new = ss[:neig]
        else: 
            check_old = check_new
            check_new = ss[:neig]
        check = np.linalg.norm(check_new - check_old)
        if check < tol:
            print('Block Davidson converged at iteration no.:',iter)
            break
    end = time.time()
    print("davidson = ", ss[:neig],";")
    print ('Block Davidson time:',end-start)
    return ss[:neig]


def get_initial_guess(A,neigen):
    nrows, ncols = A.shape
    d = np.diag(A)
    index = np.argsort(d)
    guess = np.zeros((nrows,neigen))
    for i in range(neigen):
        guess[index[i],i] = 1
    
    return guess
def jacobi_correction(uj,A,thetaj):
    I = np.eye(A.shape[0])
    Pj = I-np.dot(uj,uj.T)
    rj = np.dot((A - thetaj*I),uj) 

    w = np.dot(Pj,np.dot((A-thetaj*I),Pj))
    return np.linalg.solve(w,rj)
def davidson_4(A, n_eigen=1, tol=1E-6, maxiter = 1000, jacobi=False,non_hermitian=False,hamiltonian=False):
    """Davidosn solver for eigenvalue problem
    https://github.com/NLESC-JCER/DavidsonPython/tree/master

    Args :
        A (numpy matrix) : the matrix to diagonalize
        n_eigen (int)     : the number of eigenvalue requied
        tol (float)      : the rpecision required
        maxiter (int)    : the maximum number of iteration
        jacobi (bool)    : do the jacobi correction
    Returns :
        eigenvalues (array) : lowest eigenvalues
        eigenvectors (numpy.array) : eigenvectors
    """
    n = A.shape[0]
    k = 2*n_eigen            # number of initial guess vectors 
    V = np.eye(n,k)         # set of k unit vectors as guess
    I = np.eye(n)           # identity matrix same dimen as A
    Adiag = np.diag(A)
    residuals=[]
    start_davidson = time.time()

    V = get_initial_guess(A,k)
    
    # print('\n'+'='*20)
    # logger.trace("= Davidson Solver ")
    # print('='*20)

    #invA = np.linalg.inv(A)
    #inv_approx_0 = 2*I - A
    #invA2 = np.dot(invA,invA)
    #invA3 = np.dot(invA2,invA)

    norm = np.zeros(k if hamiltonian else n_eigen)

    # Begin block Davidson routine
    # logger.trace("iter size norm"+str(tol))
    for i in range(maxiter):
    
        # QR of V t oorthonormalize the V matrix
        # this uses GrahmShmidtd in the back
        V,R = np.linalg.qr(V)

        # form the projected matrix 
        
        T = np.dot(V.conj().T if hamiltonian or non_hermitian else V.T,np.dot(A,V)) 



        # Diagonalize the projected matrix
        theta,s = np.linalg.eigh(T)

        if hamiltonian or non_hermitian:
            # print(np.diag(T))
            # organize the eigenpairs
            index = np.argsort(theta.real)
            theta  = theta[index]
            s = s[:,index]

        # Ritz eigenvector
        q = np.dot(V,s)

        # compute the residual append append it to the 
        # set of eigenvectors
        if hamiltonian:
            ind0 = np.where(theta>0, theta, np.inf).argmin()
        
        for _j in range(k if hamiltonian else n_eigen):
            j = ind0+_j-int(0.25*k) if hamiltonian else +_j

            # residue vetor
            res = np.dot((A - theta[j]*I),q[:,j]) 
            norm[_j] = np.linalg.norm(res)

            # correction vector
            if(jacobi):
            	delta = jacobi_correction(q[:,j],A,theta[j])
            else:
            	delta = res / (theta[j]-Adiag+1E-16)
                #C = inv_approx_0 + theta[j]*I
                #delta = -np.dot(C,res)

            delta /= np.linalg.norm(delta)

            # expand the basis
            V = np.hstack((V,delta.reshape(-1,1)))

        # comute the norm to se if eigenvalue converge
        # logger.trace(str(i)+" "+str(V.shape[1])+" "+ str(np.max(norm)))
        residuals.append(np.max(norm))
        if np.all(norm < tol):
            logger.info("Davidson_4 has converged in iteration number = "+str(i))
            break
    end_davidson = time.time()
    if not hamiltonian: ind0=0
    logger.success("davidson_4 = "+ str(theta[ind0:n_eigen+ind0])+"; time = "+str(end_davidson-start_davidson)+" seconds.")


    return theta[ind0:ind0+n_eigen], q[:,ind0:ind0+n_eigen],residuals
    

def numpy_eigen(A,l,u):
      # Begin Numpy diagonalization of A

    start_numpy = time.time()

    E,Vec = np.linalg.eig(A)
    idx=np.argsort(E)


    end_numpy = time.time()
    E = E[idx]
    Vec=Vec[idx]

    # End of Numpy diagonalization. Print results.

    print("numpy = ", E[l:u],";",
        end_numpy - start_numpy, "seconds")
    return E[l:u],Vec[l:u]
    
def main():
    # A=mt.random_sparse(1600,1e-3)
    A=mt.digaonal_dominant(1600,1e-3)
    # B=mt.symmetric_sparse(1000)
    # jd.davidson_Basic(A)
    # davidson_4(A,maxiter=200,tol=1e-5,neigen=4)
    # davidson_4(A,maxiter=200,tol=1e-5,neigen=4,jacobi=True)

    # davidson_2(A)
    # davidson_3(A,mmax=20,tol=1e-9)
    # davidson_2(A,mmax=20,tol=1e-9)
    a=power_iteration(A,calc_min=True)
    print(a)
    power_iteration(A,a,calc_min=True)
    numpy_eigen(A,0,1)
    # davidson_2(A,k=16,eig=8)



if __name__=='__main__':
    main()
