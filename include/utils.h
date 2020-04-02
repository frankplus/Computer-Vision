#ifndef __UTILS__
#define __UTILS__

#include <opencv2/core.hpp>

/**
 * Show images in a collage.
 * 
 * @param src vector of images
 */
void show_collage(const std::vector<cv::Mat> &src);

/**
 * Make collage of images
 * 
 * @param src vector of images
 * @param dst destination image matrix
 * @param grid_x number of columns
 * @param grid_y number of rows
 */
void tile(const std::vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y);

#endif