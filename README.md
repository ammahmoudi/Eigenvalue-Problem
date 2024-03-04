# A Brief Overview of Iterative Methods for Eigenvalue Problems

Supervisior: Dr. Mohammadreza Razvan

Matrix Calculations course 2024

Sharif University of Technology

Department of Mathematical Sciences

## Description

The eigenvalue problem is one of linear algebra's most essential and valuable problems. It deals with finding a square matrix's special values and vectors that satisfy a certain equation. The equation is of the form $Ax = \lambda x$, where $A$ is a given matrix, $x$ is an unknown vector, and $\lambda$ is an unknown scalar. The vector $x$ is called an eigenvector of $A$, and the scalar $\lambda$ is called an eigenvalue of $A$. The word "eigen" comes from the German word for "own" or "characteristic", meaning that the eigenvectors and eigenvalues capture some essential properties of the matrix.

The eigenvalue problem arises in many applications, such as modeling vibrations, stability, and data analysis. For example, suppose we have a system of masses and springs, and we want to find the natural frequencies and modes of vibration of the system. We can represent the system by a matrix $A$, where each entry $a_{ij}$ indicates the stiffness of the spring connecting the $i$-th and the $j$-th mass. Then, the equation $Ax = \lambda x$ describes how the displacements of the masses $x$ change over time, and the eigenvalues $\lambda$ are the squares of the natural frequencies, and the eigenvectors $x$ are the modes of vibration. By finding the eigenvalues and eigenvectors of $A$, we can understand how the system behaves and how to control it.

To solve the eigenvalue problem, we must find all the possible values of $\lambda$ and the corresponding vectors of $x$ that make the equation true. There are various methods to do this, depending on the type and the size of the matrix $A$. One of the most common methods is finding the roots of the characteristic polynomial of $A$, defined as $p(\lambda) = \det(\lambda I - A)$. The determinant of $\lambda I - A$ is a polynomial function of $\lambda$, and its roots are precisely the eigenvalues of $A$. Once we find the eigenvalues, we can find them by plugging them into the equation $Ax = \lambda x$ and solving for $x$. However, this method can be difficult or impossible to carry out by hand, especially for large or complicated matrices. Therefore, we often use numerical algorithms, such as the power method, the QR algorithm, or the Jacobi method, to approximate the eigenvalues and eigenvectors of $A$.

The eigenvalue problem reveals important properties and insights about the matrix and the system it represents. For example, the number of eigenvalues of $A$ is equal to the dimension of $A$, and the sum of the eigenvalues of $A$ is equal to the trace of $A$, which is the sum of the diagonal entries of $A$. The product of the eigenvalues of $A$ equals the determinant of $A$, which measures the volume change of $A$. The eigenvalues and eigenvectors of $A$ also determine the rank, the nullity, the inverse, and the powers of $A$. Moreover, some special matrices have special eigenvalues and eigenvectors. For example, a symmetric matrix $A$ has only real eigenvalues and orthogonal eigenvectors, meaning that the eigenvectors are perpendicular. A positive definite matrix $A$ has only positive eigenvalues and positive definite eigenvectors, meaning that the eigenvectors point in the same direction as $A$. These properties make the eigenvalue problem a powerful tool for analyzing and solving linear systems.

A common way to find eigenvalues in basic linear algebra courses is to solve a characteristic polynomial. However, this method has a drawback: the solutions of the characteristic polynomial can change drastically if the polynomial coefficients are slightly altered, even for eigenvalues that are not ill-conditioned. Instead of using this approach, we will use different techniques.
Iterative methods help solve eigenvalue problems for large matrices, which are matrices that have a high number of rows and columns. Large matrices are often sparse, meaning that most entries are zero and have unique structures, such as symmetry, positive definiteness, or diagonal dominance. These properties can be exploited by iterative methods to reduce the computational cost and storage requirements of finding the eigenvalues and eigenvectors of large matrices.

Some of the most common iterative methods for eigenvalue problems are:

- Power method: This is the simplest iterative method. It computes the largest eigenvalue in absolute value and its corresponding eigenvector by repeatedly multiplying a random vector by the matrix and normalizing it. The power method is easy to implement and only requires matrix-vector multiplication, but it converges slowly and may fail if there are multiple eigenvalues of the same magnitude.
- Inverse iteration method: This is a variant of the power method, which computes the eigenvalue closest to a given value and its corresponding eigenvector by repeatedly solving a linear system with the matrix shifted by the given value and normalizing the solution. The inverse iteration method can be used to find any eigenvalue, but it requires solving a linear system at each step, which can be costly and unstable.
- Rayleigh quotient iteration method: This is another variant of the power method, which computes the eigenvalue and eigenvector of a symmetric matrix by using the Rayleigh quotient, which is the ratio of $x^T A x$ and $x^T x$, as the shift value for the inverse iteration method. The Rayleigh quotient iteration method can converge faster than the inverse iteration method, but it is not guaranteed to converge to the desired eigenvalue and may oscillate or diverge.
= Arnoldi method: This is a generalization of the power method, which computes a few eigenvalues and eigenvectors of a matrix by constructing an orthogonal basis for a Krylov subspace, which is the span of successive powers of the matrix applied to a random vector, and then finding the eigenvalues and eigenvectors of a smaller matrix that preserves the action of the original matrix on the subspace. The Arnoldi method can find eigenvalues of any magnitude and multiplicity, but it requires storing and orthogonalizing the basis vectors, which can be expensive and ill-conditioned.
- Davidson method: This is an improvement of the Arnoldi method, which computes a few eigenvalues and eigenvectors of a symmetric matrix by using a preconditioner, which is a matrix that approximates the inverse of the matrix, to accelerate the convergence of the Krylov subspace and reduce the size of the smaller matrix. The Davidson method can be more efficient and robust than the Arnoldi method, but it depends on the choice of the preconditioner, which can be challenging to construct and apply.
- Jacobi-Davidson method: This is a further improvement of the Davidson method, which computes a few eigenvalues and eigenvectors of a symmetric matrix by using a correction equation, which is a linear system that updates the approximate eigenvector by minimizing the residual, to refine the Krylov subspace and the preconditioner. The Jacobi-Davidson method can be more accurate and flexible than the Davidson method, but it requires solving a correction equation at each step, which can be challenging and time-consuming.
- Lanczos method: This is a special case of the Arnoldi method, which computes a few eigenvalues and eigenvectors of a symmetric matrix by constructing a tridiagonal matrix that preserves the action of the original matrix on the Krylov subspace. The Lanczos method can be faster and more stable than the Arnoldi method, but it may suffer from loss of orthogonality and spurious eigenvalues due to round-off errors.

## Implementation
various iterative methods for finding eigenvalues and eigenvectors of matrices in Python are presented. The following methods are implemented:

	- Power method
	- Shifted inverse power method
	- RQ iteration method
	- Block power method
	- Lanczos method
	- Davidson method
	- Subspace iteration method
	- Jacobi-Davidson method


The performance and accuracy of the implementation are tested, and the results are compared with the built-in functions in SciPy and numpy.

## Results

![Smallest Eigenvalue Problem Real Residuals](/smallest_real_diff.svg)
![Smallest Eigenvalue Problem Residuals](/smallest_residuals.svg)
![Biggest Eigenvalue Problem Real Residuals](/biggest_real_diff.svg)
![Biggest Eigenvalue Problem Residuals](/biggest_residuals.svg)

