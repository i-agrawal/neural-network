#include "helper.h"

Mat operator+(const Mat m, double d) {
	return m + Mat::Constant(m.rows(), m.cols(), d);
}

Mat operator+(double d, const Mat m) {
	return m + Mat::Constant(m.rows(), m.cols(), d);
}

Mat operator-(const Mat m, double d) {
	return m + Mat::Constant(m.rows(), m.cols(), -d);
}

Mat operator-(double d, const Mat m) {
	return Mat::Constant(m.rows(), m.cols(), d) - m;
}

Mat operator/(double d, const Mat m) {
	Mat output = m;
	for (unsigned i = 0; i < m.rows(); i++) {
		for (unsigned j = 0; j < m.cols(); j++) {
			output(i, j) = d / output(i, j);
		}
	}
	return output;
}

Mat exp(const Mat m) {
	Mat output = m;
	for (unsigned i = 0; i < m.rows(); i++) {
		for (unsigned j = 0; j < m.cols(); j++) {
			output(i, j) = exp(output(i, j));
		}
	}
	return output;
}

Mat pow(const Mat m, double d) {
	Mat output = m;
	for (unsigned i = 0; i < m.rows(); i++) {
		for (unsigned j = 0; j < m.cols(); j++) {
			output(i, j) = pow(output(i, j), d);
		}
	}
	return output;
}

Mat pow(double d, const Mat m) {
	Mat output = m;
	for (unsigned i = 0; i < m.rows(); i++) {
		for (unsigned j = 0; j < m.cols(); j++) {
			output(i, j) = pow(d, output(i, j));
		}
	}
	return output;
}

Mat log(const Mat m) {
	Mat output = m;
	for (unsigned i = 0; i < m.rows(); i++) {
		for (unsigned j = 0; j < m.cols(); j++) {
			output(i, j) = log(output(i, j));
		}
	}
	return output;
}

Mat mult(const Mat m1, const Mat m2) {
	Mat output = m1;
	for (unsigned i = 0; i < m1.rows(); i++) {
		for (unsigned j = 0; j < m1.cols(); j++) {
			output(i, j) = m1(i, j) * m2(i, j);
		}
	}
	return output;
}

Mat div(const Mat m1, const Mat m2) {
	Mat output = m1;
	for (unsigned i = 0; i < m1.rows(); i++) {
		for (unsigned j = 0; j < m1.cols(); j++) {
			output(i, j) = m1(i, j) / m2(i, j);
		}
	}
	return output;
}
