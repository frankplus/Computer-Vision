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
    Range hue_range{0, 195};
    Range value_range{92, 256};
    int filter_sigma = 61;
    int min_neighbors = 7;
    int min_size = 90;
    int delta_brightness = 45;
};

void main_finalproject();
void preprocess_image(Mat input, Mat &result, double sigma, Range hue_range, Range value_range, int delta_brightness);
void detect_and_display( DetectionParams *params );
void print_parameters(DetectionParams *filter_params);

#endif