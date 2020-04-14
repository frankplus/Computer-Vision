#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "filter.h"

using namespace std;
using namespace cv;

// constructor
Filter::Filter(cv::Mat input_img, int size) {

	input_image = input_img;
	if (size % 2 == 0)
		size++;
	filter_size = size;
}

// for base class do nothing (in derived classes it performs the corresponding filter)
void Filter::doFilter() {

	// it just returns a copy of the input image
	result_image = input_image.clone();

}

// get output of the filter
cv::Mat Filter::getResult() {
	return result_image;
}

//set window size (it needs to be odd)
void Filter::setSize(int size) {

	if (size % 2 == 0)
		size++;
	filter_size = size;
}

//get window size 
int Filter::getSize() {

	return filter_size;
}


void MedianFilter::doFilter() {
	medianBlur(input_image, result_image, getSize());
}

void GaussianFilter::doFilter() {
	Size kernel_size = Size(getSize(), getSize());
	GaussianBlur(input_image, result_image, kernel_size, sigma, 0);
}

void GaussianFilter::setSigma(double sigma) {
	this->sigma = sigma;
}

void BilateralFilter::doFilter() {
	bilateralFilter(input_image, result_image, 9, sigma_range, sigma_space);
}

void BilateralFilter::setSigmaRange(double sigma_range) {
	this->sigma_range = sigma_range;
}

void BilateralFilter::setSigmaSpace(double sigma_space) {
	this->sigma_space = sigma_space;
}