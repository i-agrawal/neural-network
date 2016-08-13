#ifndef PROJECT_NEURALNETWORK_H
#define PROJECT_NEURALNETWORK_H

#include <armadillo>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>

class neuralnetwork {
public:
	typedef arma::Mat<double> matrix;
	unsigned nlayers;
	unsigned * sizes;
	matrix 	 * theta;
	matrix	 * grads;
	matrix	 * louts;
	matrix	 * sigma;

public:
	matrix sigmoid(const matrix &) const;
	matrix vectorise(const matrix *, unsigned) const;
	matrix * reshape(const matrix &) const;
	double costfunction(const matrix &, const matrix &, double &, double &) const;

public:
	neuralnetwork(unsigned, const unsigned *, double = 0);
	~neuralnetwork();
	matrix feedforward(const matrix &) const;
	matrix * gradientcheck(const matrix &, const matrix &, double = 0) const;

};

#endif
