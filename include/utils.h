#ifndef __UTILS__
#define __UTILS__

#include <opencv2/core.hpp>

/**
 * Show images in a collage.
 * 
 * @param src vector of images
 */
void show_collage(const std::vector<cv::Mat> &src, std::string winname = "collage");

/**
 * Make collage of images
 * 
 * @param src vector of images
 * @param dst destination image matrix
 * @param grid_x number of columns
 * @param grid_y number of rows
 */
void tile(const std::vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y);

/**
 * hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
 * e.g.: hists[0] = cv:mat of size 256 with the red histogram
 *       hists[1] = cv:mat of size 256 with the green histogram
 *       hists[2] = cv:mat of size 256 with the blue histogram
 */
void show_histogram(std::vector<cv::Mat> &hists, std::string winname = "");

#endif