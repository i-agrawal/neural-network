#ifndef __helper_h__
#define __helper_h__

#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mat;

Mat operator+(const Mat, double);
Mat operator+(double, const Mat);
Mat operator-(const Mat, double);
Mat operator-(double, const Mat);
Mat operator/(double, const Mat);
Mat exp(const Mat);
Mat pow(const Mat, double);
Mat pow(double, const Mat);
Mat log(const Mat);
Mat mult(const Mat, const Mat);
Mat div(const Mat, const Mat);


#endif
