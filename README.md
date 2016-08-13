# neural-network
Neural Network implementation in C++:\n
Recently Updated Using Matrices\n
Matrix definitions handled by template library Armadillo\n
Matrix math handled by linked BLAS implementation\n

Currently Compiles With Following BLAS implementations:
Mac OSX   : Accelerate Framework
GNU/Linux : Netlib BLAS and LAPACK, OpenBLAS, Intel MKL, AMD AMCL

Notes:
Makefile currently linked using Accelerate Framework for Mac OSX.
If intending to compile on GNU/Linux please link your own BLAS framework to armadillo.
