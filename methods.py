import numpy as np
import matrix_tools as mt
import time

from scipy import sparse as sps
from scipy.linalg import eigh,inv,lu_factor,lu_solve
from scipy.linalg.lapack import dsyev
from numpy.linalg import norm
from scipy.sparse import linalg as lin
from scipy.sparse.linalg import inv as spinv
import pdb,time,warnings

#logging
import datetime
from loguru import logger
import sys        # <!- add this line
logger.remove()             # <- add this line
logger.add(sys.stdout, level="INFO")   # <- add this line
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
log_path=".\logs\log-"+str(datetime.datetime.now()).replace(" ","-").replace(".","-").replace(":","-")+".log"
logger.add(log_path, level="TRACE", format=log_format, colorize=False, backtrace=True, diagnose=True)

def block_power_method(A, k, tol=1e-6, max_iter=100):
    # A is a symmetric matrix
    # k is the number of smallest eigenvalues and eigenvectors to find
    # tol is the tolerance for convergence
    # max_iter is the maximum number of iterations
    
    # Get the dimension of A
    n = A.shape[0]
    
    # Check if k is valid
    if k < 1 or k > n:
        print("Invalid value of k")
        return None
    start_time=time.time()
    # Initialize a random matrix Q of size n x k
    Q = np.random.rand(n, k)
    
    # Orthonormalize Q using QR decomposition
    Q, _ = np.linalg.qr(Q)
    
    # Initialize a variable to store the eigenvalues
    lambdas = np.zeros(k)
    
    # Initialize a counter for iterations
    iter = 0
    
    # Loop until convergence or maximum iterations
    while iter < max_iter:
        # Perform the matrix multiplication AQ
        Z = A @ Q
        
        # Solve the linear system Q^T Z = Q^T A Q = Lambda
        # Lambda is a diagonal matrix of eigenvalues
        # We can use np.linalg.solve to find Lambda
        Lambda = np.linalg.solve(Q.T, Z.T).T
        
        # Extract the diagonal elements of Lambda
        lambdas_new = np.diag(Lambda)
        
        # Check the relative change of eigenvalues
        if np.linalg.norm(lambdas - lambdas_new) < tol:
            # Converged
            break
        
        # Update the eigenvalues
        lambdas = lambdas_new
        
        # Orthonormalize Z using QR decomposition
        Q, _ = np.linalg.qr(Z)
        
        # Increment the iteration counter
        iter += 1

    end_time=time.time()
    logger.success("Block Power method  = "+ str(lambdas)+"; time = "+str(end_time-start_time)+" seconds.")

    # Return the eigenvalues and eigenvectors
    return lambdas, Q

def subspace_iteration(A, k=1, Y0=None, maxiter=100):
    start_time=time.time()
    n=A.shape[0]

    if Y0 is None:
        Y0 = np.random.random((n, k))

    Y0, _ = np.linalg.qr(Y0)
    Y = Y0.copy()
    Y_old = Y0.copy()
    err = []
    for i in range(maxiter):
        B = A.dot(Y)
        Y, E = np.linalg.qr(B)
        err.append(np.linalg.norm(Y_old - Y.dot(Y.T.dot(Y_old))))
        Y_old = Y.copy()
        end_time=time.time()

    # approx = np.dot(Y.T,np.dot(A,Y))/np.linalg.norm(Y)

    logger.success("Subspace iteration = "+ str(np.diag(E))+"; time = "+str(end_time-start_time)+" seconds.")

    return Y, B, err

def subspace_iteration_2(A, k=1, V0=None, maxiter=1000,tol=1e-4):
    start_time=time.time()
    n=A.shape[0]

    if V0 is None:
        V0 = np.random.random((n, k))

    V=V0
    residuals = []
    err=100
    i=0
    while err>tol:
        B = A.dot(V)
        Q, R = np.linalg.qr(B)
        V=Q[:, :k]
        E=R[:k, :]
        err=np.linalg.norm(A.dot(V)-V.dot(E))
        residuals.append(err)
        i+=1
        if(err<tol): 
            logger.info('Subspace_2 converged at iteration number = '+ str(i))
            break

    end_time=time.time()


    logger.success("Subspace iteration_2 = "+ str(np.diag(E))+"; time = "+str(end_time-start_time)+" seconds.")

    return E, V, residuals

def power_iteration(A,tol=1e-10,maxiter=1000,use_rayleigh=False,calc_min=False,use_inverse=True,Sigma=0):
    '''
    Power iteration is used to find the highest eigenvalue one at a time.
    https://github.com/sreeganb/davidson_algorithm/

    '''
    residuals=[]
    A_orginal=A
    start_time = time.time()
    A=A-Sigma
    # if not calc_min and use_inverse:
    #     logger.warning('Ignoring use_inverse. Use_inverse is an option when calc_min is set to True.')
    if calc_min: 
        if use_inverse:
            try :
                A=np.linalg.inv(A)
            except np.linalg.LinAlgError as exc:
                logger.exception(exc)
                logger.warning("Matrix is Singuar.Trying to use Psudo inverse of A. Also you can turn off use_inverse flag for using lu_factor and lu_solver to solve the linear system. ")
                A=np.linalg.pinv(A)
        else:
            LU,piv=lu_factor(A)
            

            

    n=A.shape[0]
    # Build a random trial vector
    B=np.random.rand(n)
    j=0
    norm_mat=np.zeros(2)
    while j<maxiter:
        if calc_min and not use_inverse:
            C=lu_solve((LU,piv),B)
            idx=np.argmax(np.abs(C))

        elif use_rayleigh:
            rcoeff = np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
            rmat = rcoeff*np.eye(n)

            C =  np.dot(A-rcoeff,B)  if calc_min and use_inverse else np.dot(np.linalg.inv(A-rcoeff),B)
        else:
            C = np.dot(A,B)

        
        norm_c = C[idx] if ( not use_inverse and calc_min) else np.linalg.norm(C) 
        B = C/(norm_c)
        j=j+1
        # print(j)
        if j==1: 
            logger.trace('just the first iteration, give me a break')
            norm_mat[0]=norm_c
        else: 
            norm_mat[1] = norm_mat[0]
            norm_mat[0] = norm_c
            diff = abs(norm_mat[1] - norm_mat[0])
            residuals.append(diff)
            if diff < tol:
                logger.info('Power iteration converged at iteration number = '+ str(j))
                break
            else:
                continue

    approx = 1/norm_c if (calc_min and not use_inverse) else np.dot(B.T,np.dot(A,B))/np.linalg.norm(B)
    if calc_min and use_inverse: approx=1./approx
    approx += Sigma
    end_time = time.time() 
    logger.success('Power iteration = '+ str(approx)+'; time = '+str(end_time-start_time)+ " seconds.")



    w, v = np.linalg.eig(A_orginal)
    w=np.sort(w)
    diff = (w[0] if calc_min else w[-1])- approx
    logger.trace('exact eigenvalue='+(str(w[0]) if calc_min else str(w[-1])))
    logger.trace('Residual = '+ str(diff))

    return approx,B,residuals


def davidson_1(A,v0=None,tol=1e-10,maxiter=1000):
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
        ei,vi=np.linalg.eig(G)
        #compute largest Ritz value theta, and Ritz vector u.
        imax=np.argmax(ei)
        theta,u=ei[imax],V.dot(vi[:,imax:imax+1])
        #get the residual
        r=AV.dot(vi[:,imax:imax+1])-theta*u
        if norm(r)<tol:
            break
        
        if(i%20==0):logger.trace(str(i)+' ||r|| = '+ str(norm(r))+', eigen value = '+str(theta))
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

    logger.success("davidson_1 = "+ str(theta)+"; time = "+
        str(end_davidson - start_davidson)+ " seconds.")
    

    return theta,u

def davidson_2(A,k=None,n_eigen=1,tol=1e-10,maxiter=1000):
    ''' Block Davidson, Joshua Goings (2013)

    Block Davidson method for finding the first few
	lowest eigenvalues of a large, diagonally dominant,
    sparse Hermitian matrix (e.g. Hamiltonian)
'''
    n=A.shape[0]
    maxiter = n//2				# Maximum number of iterations	

    if k is None:k=2*n_eigen
    t = np.eye(n,k)			# set of k unit vectors as guess
    V = np.zeros((n,n))		# array of zeros to hold guess vec
    I = np.eye(n)			# identity matrix same dimen as A
    residuals=[]
    # Begin block Davidson routine

    start_davidson = time.time()

    for m in range(k,maxiter,k):
        if m <= k:
            for j in range(0,k):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
            theta_old = 1 
        elif m > k:
            theta_old = theta[:n_eigen]
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
        norm = np.linalg.norm(theta[:n_eigen] - theta_old)
        residuals.append(norm)
        if norm < tol:
            break

    end_davidson = time.time()

    # End of block Davidson. Print results.

    logger.success("davidson_2 = "+ str(theta[:n_eigen])+"; time = "+
       str(end_davidson - start_davidson)+" seconds.")
    return theta[:n_eigen], s[:n_eigen],residuals

def davidson_3(A,k=None,n_eigen=1,tol=1e-10,maxiter=1000):
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
    if k is None:k=2*n_eigen
    # logger.trace('No. of start vectors:'+str(k))
    # logger.trace('No. of desired Eigenvalues:'+str(n_eigen))
    t = np.eye(n,k) # initial trial vectors
    v = np.zeros((n,n)) # holder for trial vectors as iterations progress
    I = np.eye(n) # n*n identity matrix
    ritz = np.zeros((n,n))
    f = np.zeros((n,n))
    residuals=[]
    #-------------------------------------------------------------------------------
    # Begin iterations  
    #-------------------------------------------------------------------------------
    start = time.time()
    iter = 0
    for m in range(k,maxiter,k):
        iter = iter + 1
        # logger.trace("Iteration no:"+ str(iter))
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
    #        logger.trace(f)
            ritz[:,i] = np.dot(f,np.linalg.multi_dot([(A-w[i]*I),v[:,:m],vects[:,i]]))
            if np.linalg.norm(ritz[:,i]) > 1e-7 :
                ritz[:,i] = ritz[:,i]/(np.linalg.norm(ritz[:,i]))
                v[:,m+j] = ritz[:,i]
                j = j + 1
        q, r = np.linalg.qr(v[:,:m+j-1])
        for kk in range(m+j-1):
            v[:,kk] = q[:,kk]
        # for i in range(n_eigen):
        #     logger.trace(ss[i])
        if iter==1: 
            check_old = ss[:n_eigen]
            check_new = 1
        elif iter==2:
            check_new = ss[:n_eigen]
        else: 
            check_old = check_new
            check_new = ss[:n_eigen]
        check = np.linalg.norm(check_new - check_old)
        residuals.append(check)
        if check < tol:
            logger.info('Block Davidson converged at iteration number = '+str(iter))
            break
    end = time.time()
    logger.success("davidson_3 = "+ str(ss[:n_eigen])+"; time = "+str(end-start)+" seconds.")

    return ss[:n_eigen],v[:n_eigen],residuals


def get_initial_guess(A,n_eigen):
    nrows, ncols = A.shape
    d = np.diag(A)
    index = np.argsort(d)
    guess = np.zeros((nrows,n_eigen))
    for i in range(n_eigen):
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
