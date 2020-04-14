#ifndef __HOMEWORK3__
#define __HOMEWORK3__

#include <opencv2/core.hpp>

void main_homework_3();

/**
 * Generate histograms for the three channels of an image
 * @param img_channels Vector of 3 image channels
 * @param winname Name of the window
 */
void generate_show_histograms(std::vector<cv::Mat> img_channels, std::string winname);

#endif