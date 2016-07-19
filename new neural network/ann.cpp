#include "ann.h"

//construct the artificial neural network by recording layer infos and randomizing theta
ann::ann(Mat<int> &layers, double epsilon) {
	//begin initializing neural network
	assert(layers.n_cols > 1);					//make sure there is enough layers for 1 input and 1 output
	assert(layers(0, 0) != 0);					//make sure thers i more than 0 inputs

	if (epsilon == 0)							//epsilon is the range for which initial values should be randomized
		epsilon = 1.0 / sqrt(layers(0, 0));		//it is 1 / sqrt(number of inputs) under the idea the data is normalized

	layer_sizes = layers;						//record the layer size for future reference

	//begin initializing theta
	field<Mat<double> > thetaf = field<Mat<double> >(layers.n_cols - 1, 1);			//initialize theta which is equivalent to the weights
	for (unsigned i = 0; i < layers.n_cols - 1; i++) {			//create theta for each layer after input
		assert(layers(0, i) != 0);								//make sure there is more than 0 neurons in a layer
		Mat<double> randtheta(layers(0, i+1), layers(0, i)+1);	//initialize the theta from layer x to x + 1, the extra + 1 is to account for the bias
		randtheta.randu();										//randomize them between 0 and 1
		randtheta = randtheta * (2 * epsilon);					//multiply by 2 * eplsion and subtract epsilon to get values between -epsilon and epsilon
		randtheta = randtheta - epsilon;						
		thetaf(i, 0) = randtheta;								//set the theta for it
		theta = join_vert(theta, vectorise(thetaf(i, 0)));
	}
	//finished initializing theta, initial neural network has been set up
	//finsih intializing neural network
}

//trains the neural network given by the input data
//and the output data provided by the user
//will split data into training set, cv set, and test set
void ann::train(Mat<double> &X, Mat<double> &y, int iters, double lambda, double training, double cv, double test) {
	assert(training + cv + test == 1);		//make sure the partitioning equals 1
	default_random_engine generator;		//make an engine to perform uniform random distribution
	uniform_real_distribution<double> distribution(0.0, 1.0); //create a uniform real distribtuion form 0 to 1
	double number;							//create placeholder for random number

	Mat<double> trainingsetX, trainingsetY, cvsetX, cvsetY, testsetX, testsetY; //intialize sets X for input y for output
	for (unsigned i = 0; i < X.n_rows; i++) {					//go through all the examples
		number = distribution(generator);						//get a uniformly distributed random number
		if (number < training) {								//if it is within training set range
			trainingsetX = join_vert(trainingsetX, X.row(i));	//add it to training set
			trainingsetY = join_vert(trainingsetY, y.row(i));
		}
		else if (number < training + cv) {						//if it is within cross validations set range
			cvsetX = join_vert(cvsetX, X.row(i));				//add it to cross validation set
			cvsetY = join_vert(cvsetY, y.row(i));
		}
		else {													//otherwise it is in test set range
			testsetX = join_vert(testsetX, X.row(i));			//add it to test set
			testsetY = join_vert(testsetY, y.row(i));
		}
	}

	clock_t t;			//create clock to find time elapsed
	t = clock();		//get time before starts
	theta = gradientdescent(theta, trainingsetX, trainingsetY, iters, lambda);	//train the network through gradient descent
	t = clock() - t;	//get time difference
	cout << "Time Elapsed Training\t\t\t| " << static_cast<double>(t) / CLOCKS_PER_SEC << " Seconds" << endl;	//log the time elapsed

	Mat<double> pred = feed(cvsetX);											//test trained network on cross validation set
	double averror = accu((pred - cvsetY) % (pred - cvsetY)) / cvsetY.n_rows;	//calculate the average squared error
	cout << "Average Error On Cross-Validation Set\t| " << averror << endl;		//log the average squared error for cv set

	pred = feed(testsetX);														//test trained network on test set
	averror = accu((pred - testsetY) % (pred - testsetY)) / testsetY.n_rows;	//calculate the average squared error
	cout << "Average Error On Test Set\t\t| " << averror << endl;				//log the average squared error for test set
}

Mat<double> ann::feed(Mat<double> &X) {
	int nlayers = layer_sizes.n_cols;	//get the number of layers

	//reshape theta
	field<Mat<double> > thetaf = unrolledtheta(theta, layer_sizes);
	//end reshape

	//feedforward storing zs and as
	assert(X.n_cols == layer_sizes(0, 0));	//make sure the number of input features is equal to number of inputs
	field<Mat<double> > A(nlayers, 1);		//number of zs and as is equal to number of layers
	field<Mat<double> > Z(nlayers, 1);
	A(0, 0) = X;							//the first layer is special because it lacks a z so I just stored it as X
	Z(0, 0) = X;

	for (unsigned i = 0; i < thetaf.n_rows; i++) {
		Mat<double> wbias = join_horiz(ones<Mat<double> >(A(i, 0).n_rows, 1), A(i, 0));	//add the bias unit to the previous a
		Mat<double> curZ = wbias * thetaf(i, 0).t();									//znew = aold * weights
		Mat<double> curA = sigmoid(curZ);												//anew = sigmoid(znew)
		A(i+1, 0) = curA;																//store the calculated a and z
		Z(i+1, 0) = curZ;
	}

	Mat<double> pred = A(nlayers - 1, 0);	//the predicted values from feed forward
	//end of feedforward

	return pred;	//return predicted matrix of output
}

//produces a cost and a gradient given by
//the X, y, and lambda provided by train function
//it uses both feedforward and backpropagation
double ann::costfunction(Mat<double> &itheta, Mat<double> &X, Mat<double> &y, Mat<double> &grad, double lambda) {
	//costfunction calculae gradient and cost
	int nlayers = layer_sizes.n_cols;		//get the number of layers

	//reshape theta
	field<Mat<double> > thetaf = unrolledtheta(itheta, layer_sizes);
	//end reshape

	//feedforward storing zs and as
	assert(X.n_cols == layer_sizes(0, 0));	//make sure the number of input features is equal to number of inputs
	field<Mat<double> > A(nlayers, 1);		//number of zs and as is equal to number of layers
	field<Mat<double> > Z(nlayers, 1);
	A(0, 0) = X;							//the first layer is special because it lacks a z so I just stored it as X
	Z(0, 0) = X;

	for (unsigned i = 0; i < thetaf.n_rows; i++) {
		Mat<double> wbias = join_horiz(ones<Mat<double> >(A(i, 0).n_rows, 1), A(i, 0));	//add the bias unit to the previous a
		Mat<double> curZ = wbias * thetaf(i, 0).t();										//znew = aold * weights
		Mat<double> curA = sigmoid(curZ);												//anew = sigmoid(znew)
		A(i+1, 0) = curA;																//store the calculated a and z
		Z(i+1, 0) = curZ;
	}

	Mat<double> pred = A(nlayers - 1, 0);	//the predicted values from feed forward
	//end of feedforward

	//time to calculate cost = (1/m)*sum(y * log(predictedy) - (1 - y) * log(1 - predictedy)) + (0.5/m)*sum(theta^2)
	assert(size(y) == size(pred));
	double cost = 0;
	Mat<double> cost_each = y % log(pred) + (1 - y) % log(1 - pred);	//calculate cost difference for each example
	cost = -sum(sum(cost_each)) / X.n_rows;								//total cost divided by number of examples for average cost
	double lcost = 0;													//variable to hold cost for regularization using lmabda
	for (unsigned i = 0; i < thetaf.n_rows; i++) {						//go through every theta for which we have to square
		Mat<double> wobias = thetaf(i, 0).cols(1, thetaf(i, 0).n_cols-1);//remove the bias by going second col to end
		Mat<double> sq = wobias % wobias;								//square each theta value for square
		lcost += sum(sum(sq));											//then sum it so the cost increases
	}
	lcost = lambda * lcost / (2 * X.n_rows);							//average it by dividing by 2m and multiplying by factor
	cost += lcost;														//add it original cost
	//end of cost caluclation

	//backpropagation storing gradient matrices for theta change
	field<Mat<double> > gradient(nlayers-1, 1);		//stored the gradients for each theta
	field<Mat<double> > delta(nlayers, 1);			//stores the delta for each layer
	field<Mat<double> > reg(nlayers-1, 1);			//stores the delta for each layer

	delta(nlayers-1, 0) = pred - y;					//first delta is calulated by predicted - actual
	for (int i = nlayers - 2; i >= 0; i--) {		//calculating the delta, regularization, and gradient for each

		//calculate current delta
		Mat<double> wbias = join_horiz(ones<Mat<double> >(Z(i, 0).n_rows, 1), Z(i, 0));	//get z with the bias value
		Mat<double> curdelta = delta(i+1, 0) * thetaf(i, 0) % sigmoidgradient(wbias);	//current delta is calculated
		curdelta = curdelta.cols(1, curdelta.n_cols-1);									//remove bias
		delta(i, 0) = curdelta;															//store the current delta
		//end of current delta calculations

		//calculate current regularization
		Mat<double> curreg = thetaf(i, 0);							//get theta without bias weights
		curreg.col(0) = zeros<Mat<double> >(curreg.n_rows, 1);		//set bias column to 0
		curreg = lambda * curreg * (1.0 / X.n_rows);				//get curreg by multiplying by regularization factor and dividing my number of examples
		//end of current regularization calculations

		//calculate current gradient
		Mat<double> curgrad = delta(i+1, 0).t() * join_horiz(ones<Mat<double> >(A(i, 0).n_rows, 1), A(i, 0));	//the current gradient is calculated by muliplying delta of next layer by current values
		gradient(i, 0) = curgrad * (1.0 / X.n_rows); + curreg;			//divide my number of examples and add regularization
		//end of current gradient calculations
	}

	grad.clear();
	for (unsigned i = 0; i < gradient.n_rows; i++) {
		grad = join_vert(grad, vectorise(gradient(i, 0)));
	}
	//end of backpropagation

	return cost; //return the final cost
	//end of cost function
}

//estimates the parameters theta that produce the minimum cost
//given the X and y from the function train
//and the cost produced by the costfunction
Mat<double> ann::gradientdescent(Mat<double> &itheta, Mat<double> &input, Mat<double> &output, int iters, double lambda) {
	Mat<double> grad, ctheta = itheta;		//set current theta to the initial theta
	double cost, alpha = 1;					//intialize cost and set alpha to 1
	for (unsigned i = 0; i < iters; i++) {	//for each iteration
		cost = costfunction(ctheta, input, output, grad, lambda); 	//calculate the cost and gradient
		ctheta = ctheta - alpha * grad;								//make ctheta equal to new theta
	}
	return ctheta;	//return the final product
}

//produces a matrix equivalent to performing the
//sigmoid function = 1 / (1 + e^(-t)) on each element
//of the input matrix
Mat<double> ann::sigmoid(Mat<double> m) {
	m = 1 / (1 + exp(-m));	//calculate sigmoid
	return m;				//return sigmoid
}

//produces a matrix equivalent to performing the
//sigmoid function derivative = sigmoid(t)(1 - sigmoid(t))
//on each element of the input matrix
Mat<double> ann::sigmoidgradient(Mat<double> m) {
	m = sigmoid(m) % (1 - sigmoid(m));	//calculate derivative
	return m;							//return derivative
}

//helper function used to reshape vectorised matrices
//because many vectors are vectorised
field<Mat<double> > ann::unrolledtheta(Mat<double> &m, Mat<int> &layers) {
	field<Mat<double> > thetaf(layers.n_cols - 1, 1);	//create a holder for each theta to be created 
	int begin = 0, end = 0;								//create holder for beginning index and end index
	for (unsigned i = 0; i < thetaf.n_rows; i++) {								//for each theta
		begin = end;															//set beginning to the old ending
		end = begin + (layers(0, i) + 1) * layers(0, i+1);						//set ending to be old layersize + 1 for bias times next layersize
		thetaf(i, 0) = m.rows(begin, end - 1);									//get the values between the beginnign and end intervals
		thetaf(i, 0) = reshape(thetaf(i, 0), layers(0, i+1), layers(0, i)+1);	//reshape the vectorised version into a matrix
	}
	return thetaf;	//return the reshaped matrices
}
