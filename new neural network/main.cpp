#include <iostream>
#include <armadillo>
#include <time.h>
#include "ann.h"

using namespace std;
using namespace arma;

int main() {
	Mat<int> lsize;
	lsize << 2 << 3 << 1 << endr;
	ann nn(lsize);

	int num_data = 100;
	Mat<double> X(num_data, 2);
	Mat<double> y(num_data, 1);
	for (unsigned i = 0; i < num_data; i++) {
		int one = rand() & 1;
		int two = rand() & 1;
		int three;
		if(one == two)
			three = 1;
		else
			three = 0;
		X(i, 0) = one;
		X(i, 1) = two;
		y(i, 0) = three;
	}

	nn.train(X, y, 3000, 1);

	nn.save("ann");

	return 0;
}