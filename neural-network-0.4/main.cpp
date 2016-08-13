#include "neuralnet.h"
#include <time.h>
#include <iostream>
#include <armadillo>
#include <random>

#define PROJECT_NO_DEBUG

int main(int argc, char ** argv) {
	arma::Mat<double> in(5000, 2), out(5000, 1), hyp;

	unsigned i;
	int i1, i2;

	for (i = 0; i < 5000; i++) {
		i1 = rand() % 2;
		i2 = rand() % 2;
		in(i, 0) = i1;
		in(i, 1) = i2;
		if (i1 == i2) {
			out(i, 0) = 1;
		}
		else {
			out(i, 0) = 0;
		}
	}


	unsigned arr[3] = {2, 3, 1};
	double lambda = 0.1, alpha = 10, cost;
	neuralnetwork ann(3, arr);

	clock_t t = clock();

	for (i = 0; i != 1000; i++) {
		cost = ann.costfunction(in, out, lambda, alpha);
		std::cout << "Training Iteration " << i << " | " << cost << std::endl;
	}

	hyp = ann.feedforward(in);

	std::cout << join_horiz(hyp, out) << std::endl;

	t = clock() - t;
	printf("TIme Elapsed | %g\n", (double)t / CLOCKS_PER_SEC);
	return 0;
}
