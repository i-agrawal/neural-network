#ifndef __ann_h__
#define __ann_h__

#include <iostream>
#include <armadillo>
#include <assert.h>
#include <math.h>
#include <random>
#include <time.h>

using namespace std;
using namespace arma;

class ann{
public:
	//constructor
	ann(Mat<int> &, double epsilon = 0);	//constructs a artificial neural network
											//through the parameter layers that stores the size
											//of each layer
	//public functions
	Mat<double> feed(Mat<double> &);	//function that predicts output given
										//an input matrix of the features

	void train(Mat<double> &, Mat<double> &, int, double,					//trains the neural network given by the input data
			   double training = 0.6, double cv = 0.2, double test = 0.2);	//and the output data provided by the user
																			//will split data into training set, cv set, and test set

private:
	//private data
	Mat<int> layer_sizes;			//stores the size of each layer
									//is equivalent to the parameter passed through in constructor

	Mat<double> theta; 				//size = number of output by number of input + 1 for bias
				 					//number of thetas = number of layers - 1
	//private functions
	Mat<double> sigmoid(Mat<double>);	//produces a matrix equivalent to performing the
										//sigmoid function 1 / (1 + e^(-t)) on each element
										//of the input matrix

	Mat<double> sigmoidgradient(Mat<double>);	//produces a matrix equivalent to performing the
												//sigmoid function derivative sigmoid(t)(1 - sigmoid(t)) on each element
												//of the input matrix

	field<Mat<double> > unrolledtheta(Mat<double> &, Mat<int> &);	//helper function used to reshape vectorised matrices
																	//because many vectors are vectorised

	double costfunction(Mat<double> &, Mat<double> &, Mat<double> &, Mat<double> &, double);	//produces a cost and a gradient given by
																								//the X, y, and lambda provided by train function
																								//it uses both feedforward and backpropagation

	Mat<double> gradientdescent(Mat<double> &, Mat<double> &, Mat<double> &, int, double, double alpha = 10);	//estimates the parameters theta that produce the minimum cost
																												//given the X and y from the function train
																												//and the cost produced by the costfunction
};

#endif