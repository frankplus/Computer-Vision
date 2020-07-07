#ifndef __FINALPROJECT__
#define __FINALPROJECT__

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;

struct DetectionParams
{
    CascadeClassifier tree_cascade;
    Mat input_image;
    Mat filtered_image;
    Range hue_range{96, 128};
    int filter_sigma = 100;
    int min_neighbors = 5;
    int min_size = 50;
    int group_thresh = 0;
    int group_eps = 20;
};

void main_finalproject();
void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range);
void detect_and_display( DetectionParams *params );

#endif