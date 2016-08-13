#ifndef PROJECT_NEURALNETWORK_H											//place holder name called project nn
#define PROJECT_NEURALNETWORK_H											//place holder name called project nn

#include <armadillo>													//include of matrix template library
#include <stdlib.h>														//include stdlib for malloc
#include <assert.h>														//include assertions for parameter checking
#include <math.h>														//include math.h for math
#include <stdio.h>														//include stdio.h for verbose output during training

class neuralnetwork {													//neuralnetwork class definition
private:																//private member data
	typedef arma::Mat<double> matrix;									//using to define matrix to make it easier to read
	unsigned nlayers;													//the number of layers in the neural network
	unsigned * sizes;													//the size of each layer in the network
	matrix 	 * theta;													//holds the weights between each layer
	matrix	 * grads;													//holds the gradients calculated by costfunction
	matrix	 * louts;													//holds the outputs in each layer
	matrix	 * sigma;													//holds the difference calculated during backpropagation

private:																//private member functions
	matrix sigmoid(const matrix &) const;								//sigmoid needed as activation function
	matrix vectorise(const matrix *, unsigned) const;					//changes an array of matrices into a vector
	matrix * reshape(const matrix &) const;								//changes a vector into an array of matrices
	double costfunction(const matrix &, const matrix &, double &) const;//calculates costs and gradients of inputed examples
	matrix * gradientcheck(const matrix &, const matrix &, double = 0) const;	//calculates an approximation of gradients (dont use as actual calculation)

public:																	//public member functions
	neuralnetwork(unsigned, const unsigned *, double = 0);				//constructor
	~neuralnetwork();													//destructor
	matrix feedforward(const matrix &) const;							//produces hypothesis based off of inputs and weights
	void batchgradientdescent(const matrix &, const matrix &, unsigned &, double &, double &);	//trains neural network based off user input
	void reinitialize(double = 0);										//rerandomizes weights

};

#endif
