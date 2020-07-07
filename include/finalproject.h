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
    Range hue_range{36, 195};
    Range value_range{89, 256};
    int filter_sigma = 61;
    int min_neighbors = 4;
    int min_size = 90;
    int group_thresh = 0;
    int group_eps = 30;
};

void main_finalproject();
void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range, Range value_range);
void detect_and_display( DetectionParams *params );
void print_parameters(DetectionParams *filter_params);

#endif