# neural-network
#### Description
Neural Network implementation in C++.<br />
Matrix definitions handled by template library Armadillo.<br />
Matrix math handled by linked BLAS implementation.<br />

#### Currently Compiles With Following BLAS implementations:
Mac   : Accelerate Framework
Linux : BLAS/LAPACK, OpenBLAS, MKL, AMCL

Notes:<br />
Makefile currently linked using Accelerate Framework for Mac OSX.<br />
If intending to compile on GNU/Linux please link your own BLAS framework to armadillo.<br />
