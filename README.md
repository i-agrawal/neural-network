# neural-network
Neural Network implementation in C++.<br />
Recently Updated Using Matrices.<br />
Matrix definitions handled by template library Armadillo.<br />
Matrix math handled by linked BLAS implementation.<br />

Currently Compiles With Following BLAS implementations:<br />
Mac OSX   : Accelerate Framework<br />
GNU/Linux : Netlib BLAS and LAPACK, OpenBLAS, Intel MKL, AMD AMCL<br />

Notes:<br />
Makefile currently linked using Accelerate Framework for Mac OSX.<br />
If intending to compile on GNU/Linux please link your own BLAS framework to armadillo.<br />
