#include "ann.h"

ann::ann(const Mat &l, double epsilon) {
	layers = l;
	assert(layers.cols() > 1);
	assert(layers(0, 0) != 0);
	if (epsilon == 0) {
		epsilon = layers(0, 0) / 3.0;
	}

	theta = Mat(0, 1);
	for (unsigned i = 1; i < layers.cols(); i++) {
		assert(layers(0, i) != 0);
		int size = layers(0, i) * (layers(0, i - 1) + 1);
		Mat initial = Mat::Random(size, 1); //output layer x input layer + bias
		initial	= initial * epsilon;
		Mat temp = theta;
		theta.resize(theta.rows() + size, 1);
		theta << temp, initial;
	}
}

void ann::train(const Mat &input, const Mat &output, double lambda, bool verbose) {
	assert(input.cols() == layers(0, 0));
	assert(output.cols() == layers(0, layers.cols() - 1));
	assert(input.rows() == output.rows());

	double cost;
	Mat gradient;
	for (unsigned i = 0; i < 10000; i++) {
		cost = costfunction(input, output, lambda, gradient);
		std::cout << "[ Iteration " << i+1 << " ] | " << cost << std::endl;
		theta = theta - gradient;
	}
}

Mat ann::feed(const Mat &input, bool verbose) {
	assert(input.cols() == layers(0, 0));
	Mat * A = new Mat[layers.cols()];
	Mat * Z = new Mat[layers.cols()];
	Mat * T = reshape(theta);
	Z[0] = input;
	A[0] = input;
	for (unsigned i = 1; i < layers.cols(); i++) {
		Mat wbias(A[i-1].rows(), A[i-1].cols() + 1);
		wbias << Mat::Constant(A[i-1].rows(), 1, 1), A[i-1];
		Z[i] = wbias * T[i-1].transpose();
		A[i] = sigmoid(Z[i]);
	}
	return A[layers.cols() - 1];
}

double ann::costfunction(const Mat &input, const Mat &output, double lambda, Mat &gradient) {
	assert(input.cols() == layers(0, 0));
	assert(output.cols() == layers(0, layers.cols() - 1));
	assert(input.rows() == output.rows());

	//begin feed forward calculation
	double m = input.rows();
	Mat * A = new Mat[layers.cols()];
	Mat * B = new Mat[layers.cols()];
	Mat * Z = new Mat[layers.cols()];
	Mat * T = reshape(theta);
	Z[0] = input;
	A[0] = input;
	for (unsigned i = 1; i < layers.cols(); i++) {
		Mat wbias(A[i-1].rows(), A[i-1].cols() + 1);
		wbias << Mat::Constant(A[i-1].rows(), 1, 1), A[i-1];
		B[i-1] = wbias;
		Z[i] = B[i-1] * T[i-1].transpose();
		A[i] = sigmoid(Z[i]);
	}
	Mat wbias(A[layers.cols() - 1].rows(), A[layers.cols() - 1].cols() + 1);
	wbias << Mat::Constant(A[layers.cols() - 1].rows(), 1, 1), A[layers.cols() - 1];
	B[layers.cols() - 1] = wbias;
	//end feed forward calculation

	//begin cost calculation
	Mat pred = A[layers.cols() - 1];
	Mat costeach = mult(output, log(pred)) + mult((1 - output), log(1 - pred));
	double cost = -costeach.sum() / m;
	Mat * wobias = T;
	for (unsigned i = 0; i < layers.cols() - 1; i++) {
		wobias[i].col(0) = Mat::Constant(wobias[i].rows(), 1, 0);
		cost += (lambda * pow(wobias[i], 2).sum() / (2 * m));
	}
	//end cost clauclation

	//begin backpropagation calculation
	Mat * S = new Mat[layers.cols()];
	Mat * D = new Mat[layers.cols() - 1];
	S[layers.cols() - 1] = A[layers.cols() - 1] - output;
	for (unsigned i = layers.cols() - 2; i > 0; i--) {
		S[i] = mult(S[i+1] * T[i].block(0, 1, T[i].rows(), T[i].cols() - 1),
					sigmoidgradient(Z[i]));
	}
	for (unsigned i = 0; i < layers.cols() - 1; i++) {
		D[i] = (S[i+1].transpose() * B[i]) / m + (wobias[i] * lambda / m);
	}

	Mat grad(0, 1);
	for (unsigned i = 0; i < layers.cols() - 1; i++) {
		Mat temp = grad;
		Mat vect = Eigen::Map<Mat>(D[i].data(), D[i].cols() * D[i].rows(), 1);
		grad.resize(grad.rows() + vect.rows(), 1);
		grad << temp, vect;
	}
	gradient = grad;
	//end backpropagation calculation

	return cost;
}

Mat ann::sigmoid(Mat input){
	return 1.0 / (1.0 + exp(-input));
}

Mat ann::sigmoidgradient(Mat input){
	return mult(sigmoid(input), (1 - sigmoid(input)));
}

Mat * ann::reshape(Mat m) {
	assert(m.cols() == 1);
	Mat * T = new Mat[layers.cols() - 1];
	int rows, cols, begin, end = 0;
	for (unsigned i = 1; i < layers.cols(); i++) {
		rows = layers(0, i);
		cols = layers(0, i-1)+1;
		begin = end;
		end = begin + rows * cols;
		Mat b = m.block(begin, 0, rows * cols, 1);
		Mat r = Eigen::Map<Mat>(b.data(), rows, cols);
		T[i-1] = r;
	}
	return T;
}

Mat ann::size(Mat m) {
	Mat sz(1, 2);
	sz(0, 0) = m.rows();
	sz(0, 1) = m.cols();
	return sz;
}
