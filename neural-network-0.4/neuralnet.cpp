#include "neuralnet.h"

neuralnetwork::neuralnetwork(unsigned unum, const unsigned usizes[], double epsilon): nlayers(unum) {
	#ifndef PROJECT_NO_DEBUG											//if the user wishs for no parameter checking define PROJECT_NO_DEBUG
		assert(unum > 2);												//neural network must have 2 or more layers
		assert(usizes[0] > 0);											//layers must have 1 or more neurons
	#endif																//end check for PROJECT_NO_DEBUG
	sizes = (unsigned *)malloc(unum * sizeof(unsigned));				//initialization of our unsigned array to hold the sizes of layers
	theta = (matrix *)malloc((unum-1) * sizeof(matrix));				//initialization of our theta array to hold the values of weights

	if(epsilon == 0) {													//check if the user has a defined epsilon value
		epsilon = 1.0 / sqrt(static_cast<double>(usizes[0]));			//otherwise use the standard 1 / sqrt(inputs)
	}																	//end epsilon creation

	unsigned i;															//for loop variable
	for (i = 0; i != unum; i++) {										//loop to randomly initialize weights
		assert(usizes[i] > 0);											//layers must have 1 or more neurons
		sizes[i] = usizes[i];											//record the layer size in our layer sizes array
		if (i != 0) {													//there are weights in front of every layer except the first one
			theta[i-1] = arma::randu(sizes[i], sizes[i-1]+1) * 2 * epsilon - epsilon;	//create randomized weights
		}																//end of the if statement
	}																	//end loop for initialization

	louts = (matrix *)malloc(nlayers * sizeof(matrix));					//create an array to hold the outputs from each layer
	sigma = (matrix *)malloc(nlayers * sizeof(matrix));					//create an array to hold the sigma for each layers
	grads = (matrix *)malloc((nlayers-1) * sizeof(matrix));				//create an array to hold the gradients for each theta

}

neuralnetwork::~neuralnetwork() {
	free(sizes);														//free the pre-allocated space for layer sizes
	free(theta);														//free the pre-allocated space for theta matrices
	free(grads);														//free the pre-allocated space for gradients
	free(louts);														//free the pre-allocated space for layer outputs
	free(sigma);														//free the pre-allocated space for sigma difference in each layer
}

void neuralnetwork::reinitialize(double epsilon) {
	if (epsilon == 0) {													//check if user defined epsilon
		epsilon = 1.0 / sqrt(static_cast<double>(sizes[0]));			//epsilon estimation = 1 / sqrt(n_input neurons)
	}																	//end check

	unsigned i;															//for loop variable
	for (i = 0; i != nlayers-1; i++) {									//loop to randomly initialize weights
		theta[i-1] = arma::randu(sizes[i+1], sizes[i]+1) * 2 * epsilon - epsilon;	//create randomized weights
	}																	//end loop for initialization
}

arma::Mat<double> neuralnetwork::feedforward(const matrix &input) const {
	#ifndef PROJECT_NO_DEBUG											//if the user wishs for no parameter checking define PROJECT_NO_DEBUG
		assert(input.n_cols == sizes[0]);								//number of features and number of inputs must match
	#endif																//end check for PROJECT_NO_DEBUG
	matrix output = input;												//the first layer's output is simply the input later
	output = join_horiz(arma::ones(input.n_rows, 1), output);			//add the bias unit
	unsigned i;															//for loop variable
	for (i = 0; i != nlayers-1; i++) {									//for each layer onward
		output = output * theta[i].t();									//the output is the pervious layer * thetatransposed
		output = sigmoid(output);										//the output is then sigmoided
		if (i != nlayers-2) {											//if it is not the output layer output
			output = join_horiz(arma::ones(input.n_rows, 1), output);	//add the bias unit
		}																//end the if statement
	}																	//end of for loop
	return output;														//return output
}

double neuralnetwork::costfunction(const matrix &input, const matrix &output, double &lambda) const {
	unsigned m = input.n_rows;											//number of examples

	#ifndef PROJECT_NO_DEBUG											//if the user wishs for no parameter checking define PROJECT_NO_DEBUG
		assert(m == output.n_rows);										//the number of input examples must match the number of output examples
		assert(input.n_cols == sizes[0]);								//the number of input features must equal the number of inputs
		assert(output.n_cols == sizes[nlayers-1]);						//the number of output values must equal the number of outputs
		assert(lambda >= 0);											//the regularization parameter can not be negative
	#endif																//end check for PROJECT_NO_DEBUG

	louts[0] = input;													//input layer is unchanged
	louts[0] = join_horiz(arma::ones(m, 1), louts[0]);					//add the bias unit

	unsigned i;															//loop variable
	for (i = 1; i != nlayers; i++) {									//loop to calculate layer inputs and outputs
		louts[i] = sigmoid(louts[i-1] * theta[i-1].t());				//the output to the next layer is the output from last layer * thetatransposed
		if (i != nlayers-1) {											//if it is not the output layer
			louts[i] = join_horiz(arma::ones(m, 1), louts[i]);			//add the bias units
		}																//end of if statement
	}																	//end of for loop

	matrix foutput = louts[nlayers-1];									//final output also the neural network output
	matrix vals = output % log(foutput) + (1 - output) % log(1 - foutput);	//the measure of how off each neural network output is
	double cost = -arma::accu(vals) / static_cast<double>(m);			//calculate the average cost of the function

	if (lambda != 0) {													//if we have regularization
		double lambdacost = 0;											//calculate the regularization cost of the function's weights
		for (i = 0; i != nlayers-1; i++) {								//loop to go through each theta
			lambdacost += arma::accu(theta[i].cols(1, theta[i].n_cols-1) % theta[i].cols(1, theta[i].n_cols-1));	//the cost is theta^2 without bias
		}																//end loop
		lambdacost = lambda * lambdacost / (2.0 * static_cast<double>(m));//lambdacost total = lambda * sum(theta^2 without bias) / (2m)
		cost += lambdacost;												//add the regularization cost
	}																	//end if statement

	sigma[nlayers-1] = output - foutput;								//the sigma for the output layer is actual - predicted
	for (i = nlayers-2; i != 0; i--) {									//for loop to calculate sigma in the rest of the layers
		sigma[i] = (sigma[i+1] * theta[i]) % (louts[i] % (1.0 - louts[i]));	//sigma for other layers
	}																	//end for loop

	//if we were to calculate delta in its own loop
	/*

		for (i = 0; i != nlayers-1; i++) {
			delta[i] = sigma[i+1].t() * louts[i];
		}

	*/

	matrix cursigma;													//holds the current sigma
	for (i = 0; i != nlayers-1; i++) {									//loop through all thetas
		if (i != nlayers-2) {											//if not the last layer
			cursigma = sigma[i+1].cols(1, sigma[i+1].n_cols-1);			//remove the bias
		}																//end if
		else {															//else
			cursigma = sigma[i+1];										//there is no bias
		}																//end else
		grads[i] = (cursigma.t() * louts[i]) / static_cast<double>(m);	//the gradient is delta / m
		if (lambda != 0) {												//if we have regularization
			grads[i] = -grads[i];										//gradients are calculated negative
			grads[i] += lambda * join_horiz(arma::zeros(theta[i].n_rows, 1), theta[i].cols(1, theta[i].n_cols-1)) / static_cast<double>(m);//the regularization addition is lambda * theta without bias
		}																//end if statement
	}																	//end loop through all thetas

	return cost;														//return final cost
}

void neuralnetwork::batchgradientdescent(const matrix &input, const matrix &output, unsigned &maxiters, double &alpha, double &lambda) {
	unsigned i, j;														//loop variables
	double cost;														//holds cost of current batch
	for (i = 0; i != maxiters; i++) {									//perform gradient addition iter number of times
		cost = costfunction(input, output, lambda);						//calculate cost and gradient
		#ifndef PROJECT_NO_VERBOSE										//if the user wishs no text output define PROJECT_NO_VERBOSE
			printf("Iteration #%i | error cost = %g\n", i+1, cost);		//print the iteration and error cost of current iteration
		#endif															//end of if
		for (j = 0; j != nlayers-1; j++) {								//loop through all thetas
			theta[j] = theta[j] - alpha * grads[j];						//adjust for calculated gradient
		}																//end gradient loop
	}																	//end loop through thetas
}

arma::Mat<double> * neuralnetwork::gradientcheck(const matrix &input, const matrix &output, double lambda) const {
	unsigned i, j, k, l, m = input.n_rows;								//loop variables and number of examples
	double epsilon = 0.0001;											//the amount of difference to find approx derivative
	matrix * gradcheck = (matrix *)malloc((nlayers-1) * sizeof(matrix));//holds approximate derivatives
	matrix foutput, vals;												//holds matrices for variables
	double cost, lambdacost;											//holds the cost and lambdacost addition
	for (i = 0; i != nlayers-1; i++) {									//loop through all thetas
		gradcheck[i] = matrix(theta[i].n_rows, theta[i].n_cols);		//make the matrix the same size as theta
		for (j = 0; j != theta[i].n_rows; j++) {						//loops through each row of theta values
			for (k = 0; k != theta[i].n_cols; k++) {					//loops through each theta value in a row
				theta[i](j, k) += epsilon;								//add epsilon
				foutput = feedforward(input);							//find our output
				vals = output % log(foutput) + (1 - output) % log(1 - foutput); //difference between expected and calculated
				cost = -arma::accu(vals) / static_cast<double>(m);		//cost = diff / n_examples

				if (lambda != 0) {										//if there is regularization
					lambdacost = 0;										//calculate lambda cost
					for (l = 0; l != nlayers-1; l++) {					//loop through all thetas
						lambdacost += arma::accu(theta[l].cols(1, theta[l].n_cols-1) % theta[l].cols(1, theta[l].n_cols-1)); //increases cost based off theta^2 without bias values
					}													//end loop through all thetas
					lambdacost = lambda * lambdacost / (2.0 * static_cast<double>(m)); //additional cost is multiplied by lambda and divided by 2*n_examples
					cost += lambdacost;									//add the additional cost
				}
				gradcheck[i](j, k) = cost;								//record cost

				theta[i](j, k) -= 2*epsilon;							//subtract epsilon from original theta
				foutput = feedforward(input);							//find our output
				vals = output % log(foutput) + (1 - output) % log(1 - foutput); //difference between expected and calculated
				cost = -arma::accu(vals) / static_cast<double>(m);		//cost = diff / n_examples

				if (lambda != 0) {										//if there is regularization
					lambdacost = 0;										//calculate lambda cost
					for (l = 0; l != nlayers-1; l++) {					//loop through all thetas
						lambdacost += arma::accu(theta[l].cols(1, theta[l].n_cols-1) % theta[l].cols(1, theta[l].n_cols-1)); //increases cost based off theta^2 without bias values
					}													//end loop through all thetas
					lambdacost = lambda * lambdacost / (2.0 * static_cast<double>(m)); //additional cost is multiplied by lambda and divided by 2*n_examples
					cost += lambdacost;									//add the additional cost
				}

				gradcheck[i](j, k) = (gradcheck[i](j, k) - cost) / (2.0 * epsilon);	//approx derivative is J(T+e) - J(T-e) / 2e
				theta[i](j, k) += epsilon;								//correct to original theta value
			}															//end looping through each theta value in row
		}																//end looping through each row in theta
	}																	//end looping through each theta
	return gradcheck;													//return the approximated gradient
}

arma::Mat<double> neuralnetwork::sigmoid(const matrix &tosig) const {
	return 1 / (1 + exp(-tosig));										//sigmoid of x = 1 / (1 + e^-x)
}

arma::Mat<double> neuralnetwork::vectorise(const matrix * toroll, unsigned size) const {
	matrix rolled(0, 0);												//will hold array of matrices vectorised
	unsigned i;															//variable for loop
	for (i = 0; i != size; i++) {										//go through each matrix in array
		if (rolled.n_rows == 0) {										//if this is the first matrix in the array initialize the matrix
			rolled = arma::vectorise(toroll[i]);						//to a vectorised version of the first matrix
		}																//go to else
		else {															//if it is not the first matrix
			rolled = join_vert(rolled, arma::vectorise(toroll[i]));		//join the vectorised version of the matrix to current rolled matrix
		}																//end the check
	}																	//end loop through all matrices in the array
	return rolled;														//calculate the rolled
}

arma::Mat<double> * neuralnetwork::reshape(const matrix &tounroll) const {
	matrix * unrolled = (matrix *)malloc((nlayers-1) * sizeof(matrix));	//will turn a vectorised theta into unrolled theta
	unsigned i;															//variable for loop
	int start, end = 0;													//hold the start and end of current matrix
	for (i = 1; i != nlayers; i++) {									//for each theta
		start = end;													//start was the last end
		end = start + sizes[i] * (sizes[i-1] + 1);						//and end is the number of variables plus start
		unrolled[i-1] = arma::reshape(tounroll.rows(start, end-1), sizes[i], sizes[i-1]+1); //unroll the matrix and put it in index
	}																	//end loop through thetas
	return unrolled;													//return the created matrix
}
