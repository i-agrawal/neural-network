#ifndef __ann_h__
#define __ann_h__

#include <Eigen/Dense>
#include "helper.h"
#include <assert.h>
#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mat;

class ann {
public:
	Mat layers, theta;
	ann(const Mat &, double epsilon = 0);
	void train(const Mat &, const Mat &, double, bool verbose = true);
	Mat feed(const Mat &, bool verbose = true);
	Mat sigmoid(const Mat);
	Mat sigmoidgradient(const Mat);
	Mat * reshape(Mat);
	Mat size(Mat);
	double costfunction(const Mat &, const Mat &, double lambda, Mat &);

private:

};

#endif
